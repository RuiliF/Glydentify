import os
import time
import json
import random
import re
import sys
import numpy as np
import torch
from Bio.PDB import PDBParser, MMCIFParser

from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score, roc_auc_score, matthews_corrcoef
from sklearn.metrics import precision_recall_curve, auc

def compute_metrics(targets, probs):
    labels = np.array(targets)
    preds = (probs > 0.5).astype(int)

    def pr_auc_score(labels, probs):
        precision, recall, _ = precision_recall_curve(labels.flatten(), probs.flatten())
        return auc(recall, precision)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    cm = confusion_matrix(labels, preds)
    try:
        tn, fp, fn, tp = cm.ravel()
    except:
        tn, fp, fn, tp = 0, 0, 0, 0
        # print(cm)
        # print(labels)
    return {
        "accuracy": accuracy_score(labels, preds),  # strict exact match across all labels
        "f1": f1.item() if isinstance(f1, np.float64) else f1,
        "precision": precision.item() if isinstance(precision, np.float64) else precision,
        "recall": recall.item() if isinstance(recall, np.float64) else recall,
        "roc_auc": roc_auc_score(labels, probs, average="micro"),
        "pr_auc": pr_auc_score(labels, probs),
        "mcc": matthews_corrcoef(labels.flatten(), preds.flatten()),
        "tn": tn.item(),
        "fp": fp.item(),
        "fn": fn.item(),
        "tp": tp.item()
    }

def compute_multilabel_metrics(probs, labels, label2id, threshold=0.5):
    """
    Computes global (micro/macro) and per-class metrics for multilabel classification.
    
    Args:
        probs (np.ndarray): Shape (N_samples, N_classes) or (N_samples, N_classes). Probabilities.
        labels (np.ndarray): Shape (N_samples, N_classes). Ground truth binary labels.
        label2id (dict): Mapping from label name to index.
        threshold (float): Threshold for binary prediction.
        
    Returns:
        results (dict): Dictionary with keys like "f1_micro", "label_precision", etc.
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    preds = (probs > threshold).astype(int)
    
    # micro/macro metrics
    p_mi, r_mi, f1_mi, _ = precision_recall_fscore_support(labels, preds, average="micro")
    p_ma, r_ma, f1_ma, _ = precision_recall_fscore_support(labels, preds, average="macro")
    
    try:
        roc_mi = roc_auc_score(labels, probs, average="micro")
    except ValueError:
        roc_mi = 0.0

    # PR-AUC on flattened array
    prec, rec, _ = precision_recall_curve(labels.flatten(), probs.flatten())
    pr_mi = auc(rec, prec)
    mcc = matthews_corrcoef(labels.flatten(), preds.flatten())
    acc = accuracy_score(labels, preds)

    results = {
        "accuracy": acc,
        "f1_micro": f1_mi, "precision_micro": p_mi, "recall_micro": r_mi,
        "f1_macro": f1_ma, "precision_macro": p_ma, "recall_macro": r_ma,
        "roc_auc_micro": roc_mi,
        "pr_auc_micro": pr_mi,
        "mcc": mcc
    }

    # Per-class metrics
    for label, i in label2id.items():
        # Reuse compute_metrics for single class (binary problem)
        # We assume compute_metrics handles 1D arrays correctly
        metrics = compute_metrics(labels[:, i], probs[:, i])
        
        # Unpack useful ones into results with prefix
        results[f"{label}_precision"] = metrics["precision"]
        results[f"{label}_recall"]    = metrics["recall"]
        results[f"{label}_f1"]        = metrics["f1"]
        results[f"{label}_roc_auc"]   = metrics["roc_auc"]
        results[f"{label}_pr_auc"]    = metrics["pr_auc"]
        results[f"{label}_mcc"]       = metrics["mcc"]

    return results

def report_metrics(all_labels, all_probs, label2id):
    results = {}
    major_classes = {'classes':[], 'y_true':[], 'y_pred':[]}
    all_classes = {'classes':[], 'y_true':[], 'y_pred':[]}
    for label, i in label2id.items():
        n_pos = sum(all_labels[:,i])
        if n_pos > 10:
            major_classes['classes'].append(label)
            major_classes['y_true'].append(all_labels[:,i])
            major_classes['y_pred'].append(all_probs[:,i])

        metrics = compute_metrics(all_labels[:,i], all_probs[:,i])
        metrics['n_pos'] = sum(all_labels[:,i]).item() if isinstance(sum(all_labels[:,i]), (float, int)) else sum(all_labels[:,i]).item()

        all_classes['classes'].append(label)
        all_classes['y_true'].append(all_labels[:,i])
        all_classes['y_pred'].append(all_probs[:,i])
        results[label] = metrics

    major_metrics = compute_metrics(np.concatenate(major_classes['y_true'], axis=0), np.concatenate(major_classes['y_pred'], axis=0))
    results['major_classes'] = major_metrics
    all_metrics = compute_metrics(np.concatenate(all_classes['y_true'], axis=0), np.concatenate(all_classes['y_pred'], axis=0))
    results['all_classes'] = all_metrics
    return results

def set_seed(seed):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)

# Add current path to sys.path if needed, though usually not recommended in utils
sys.path.append(".")


# Get structural seqs from pdb file
def get_struc_seq(foldseek,
                  path,
                  chains: list = None,
                  process_id: int = 0,
                  plddt_mask: bool = "auto",
                  plddt_threshold: float = 70.,
                  foldseek_verbose: bool = False) -> dict:
    """
    This function is adapted from SaProt to convert a protein structure into a structure-aware sequence using Foldseek.
    For Foldseek binary installation and download, please refer to the `README.md`.

    Args:
        foldseek: Binary executable file of foldseek

        path: Path to pdb file

        chains: Chains to be extracted from pdb file. If None, all chains will be extracted.

        process_id: Process ID for temporary files. This is used for parallel processing.

        plddt_mask: If True, mask regions with plddt < plddt_threshold. plddt scores are from the pdb file.

        plddt_threshold: Threshold for plddt. If plddt is lower than this value, the structure will be masked.

        foldseek_verbose: If True, foldseek will print verbose messages.

    Returns:
        seq_dict: A dict of structural seqs. The keys are chain IDs. The values are tuples of
        (seq, struc_seq, combined_seq).
    """
    assert os.path.exists(foldseek), f"Foldseek not found: {foldseek}"
    assert os.path.exists(path), f"PDB file not found: {path}"
    
    tmp_save_path = f"get_struc_seq_{process_id}_{time.time()}.tsv"
    if foldseek_verbose:
        cmd = f"{foldseek} structureto3didescriptor --threads 1 --chain-name-mode 1 {path} {tmp_save_path}"
    else:
        cmd = f"{foldseek} structureto3didescriptor -v 0 --threads 1 --chain-name-mode 1 {path} {tmp_save_path}"
    os.system(cmd)
    
    # Check whether the structure is predicted by AlphaFold2
    if plddt_mask == "auto":
        with open(path, "r") as r:
            plddt_mask = True if "alphafold" in r.read().lower() else False
    
    seq_dict = {}
    name = os.path.basename(path)
    with open(tmp_save_path, "r") as r:
        for i, line in enumerate(r):
            desc, seq, struc_seq = line.split("\t")[:3]
            
            # Mask low plddt
            if plddt_mask:
                try:
                    plddts = extract_plddt(path)
                    assert len(plddts) == len(struc_seq), f"Length mismatch: {len(plddts)} != {len(struc_seq)}"
                    
                    # Mask regions with plddt < threshold
                    indices = np.where(plddts < plddt_threshold)[0]
                    np_seq = np.array(list(struc_seq))
                    np_seq[indices] = "#"
                    struc_seq = "".join(np_seq)
                
                except Exception as e:
                    print(f"Error: {e}")
                    print(f"Failed to mask plddt for {name}")
            
            name_chain = desc.split(" ")[0]
            chain = name_chain.replace(name, "").split("_")[-1]
            
            if chains is None or chain in chains:
                if chain not in seq_dict:
                    combined_seq = "".join([a + b.lower() for a, b in zip(seq, struc_seq)])
                    seq_dict[chain] = (seq, struc_seq, combined_seq)
    
    os.remove(tmp_save_path)
    os.remove(tmp_save_path + ".dbtype")
    return seq_dict


def extract_plddt(pdb_path: str) -> np.ndarray:
    """
    Extract plddt scores from pdb file.
    Args:
        pdb_path: Path to pdb file.

    Returns:
        plddts: plddt scores.
    """

    # Initialize parser
    if pdb_path.endswith(".cif"):
        parser = MMCIFParser()
    elif pdb_path.endswith(".pdb"):
        parser = PDBParser()
    else:
        raise ValueError("Invalid file format for plddt extraction. Must be '.cif' or '.pdb'.")
    
    structure = parser.get_structure('protein', pdb_path)
    model = structure[0]
    chain = model["A"]

    # Extract plddt scores
    plddts = []
    for residue in chain:
        residue_plddts = []
        for atom in residue:
            plddt = atom.get_bfactor()
            residue_plddts.append(plddt)
        
        plddts.append(np.mean(residue_plddts))

    plddts = np.array(plddts)
    return plddts


def transform_pdb_dir(foldseek: str, pdb_dir: str, seq_type: str, save_path: str):
    """
    Transform a directory of pdb files into a fasta file.
    Args:
        foldseek: Binary executable file of foldseek.
        
        pdb_dir: Directory of pdb files.
        
        seq_type: Type of sequence to be extracted. Must be "aa" or "foldseek"
        
        save_path: Path to save the fasta file.
    """
    assert os.path.exists(foldseek), f"Foldseek not found: {foldseek}"
    assert seq_type in ["aa", "foldseek"], f"seq_type must be 'aa' or 'foldseek'!"
    
    tmp_save_path = f"get_struc_seq_{time.time()}.tsv"
    cmd = f"{foldseek} structureto3didescriptor --chain-name-mode 1 {pdb_dir} {tmp_save_path}"
    os.system(cmd)
    
    with open(tmp_save_path, "r") as r, open(save_path, "w") as w:
        for line in r:
            protein_id, aa_seq, foldseek_seq = line.strip().split("\t")[:3]
            
            if seq_type == "aa":
                w.write(f">{protein_id}\n{aa_seq}\n")
            else:
                w.write(f">{protein_id}\n{foldseek_seq.lower()}\n")
    
    os.remove(tmp_save_path)
    os.remove(tmp_save_path + ".dbtype")
    

if __name__ == '__main__':
    foldseek = "/sujin/bin/foldseek"
    # test_path = "/sujin/Datasets/PDB/all/6xtd.cif"
    test_path = "/sujin/Datasets/FLIP/meltome/af2_structures/A0A061ACX4.pdb"
    plddt_path = "/sujin/Datasets/FLIP/meltome/af2_plddts/A0A061ACX4.json"
    res = get_struc_seq(foldseek, test_path, plddt_path=plddt_path, plddt_threshold=70.)
    print(res["A"][1].lower())
