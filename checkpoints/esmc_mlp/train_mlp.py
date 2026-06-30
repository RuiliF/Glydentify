from transformers import AutoTokenizer, EsmModel
from train_add_residue_freeze_both import GTDonorDataset, get_criterion, eval_model, collate_fn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, precision_recall_curve, auc, matthews_corrcoef, accuracy_score
import wandb
import os
import json
import argparse
import pandas as pd
import ast
import pickle
from datetime import datetime
from losses import *
from esm.models.esmc import ESMC
from esm.utils import encoding
from esm.utils.misc import stack_variable_length_tensors
from tqdm import tqdm
from fix_random_seed import set_seed

class Classifier(nn.Module):
    def __init__(self, checkpoint: str, num_labels: int, train_encoder: bool = False):
        super().__init__()
        self.seq_encoder = ESMC.from_pretrained(checkpoint)
        if checkpoint == "esmc_600m":
            self.proj_dim = 1152
        else:
            raise ValueError("Not recognized checkpoint!")

        self.num_labels = num_labels
        if not train_encoder:
            for p in self.seq_encoder.parameters():
                p.requires_grad = False
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.proj_dim, self.proj_dim),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(self.proj_dim, num_labels),
        )
    
    def forward(self, sequence_tokens):
        output = self.seq_encoder(sequence_tokens)
        embeddings = output.embeddings.to(torch.float32)

        seq_valid_mask = (sequence_tokens > 2).to(torch.float32).unsqueeze(-1)
        sum_repr = (embeddings * seq_valid_mask).sum(dim=1)               # [B, H]
        counts   = seq_valid_mask.sum(dim=1).clamp(min=1)
        seq_repr_mean = sum_repr / counts

        x = seq_repr_mean
        x = self.classifier(x)
        return x, seq_repr_mean, embeddings

    def save_checkpoint(self, save_path):
        if os.path.isdir(save_path):
            save_path = os.path.join(save_path, "state_dict.pt")
        
        trainable_params = {
            name: param.detach().cpu()
            for name, param in self.named_parameters() if param.requires_grad
        }
        torch.save(trainable_params, save_path)

    def load_checkpoint(self, save_path):
        if os.path.isdir(save_path):
            save_path = os.path.join(save_path, "state_dict.pt")
        self.load_state_dict(torch.load(save_path, weights_only=True), strict=False)

    @classmethod
    def from_pretrained(cls, checkpoint_path):
        if os.path.isdir(checkpoint_path):
            checkpoint_path = os.path.join(checkpoint_path, "state_dict.pt")
        config = json.load(open(checkpoint_path.replace("state_dict.pt", "config.json")))
        train_encoder = config["train_encoder"]
        label2id = config["label2id"]
        model = cls(config.get("ckpt_name", "esmc_600m"), num_labels=len(label2id), train_encoder=train_encoder)
        state_dict = torch.load(checkpoint_path, weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        model.load_checkpoint(checkpoint_path)
        return model

def train_model(df_train, df_val, label2id, criterion,
                ckpt_name="esmc_600m",
                lr=5e-5,
                weight_decay=1e-4,
                batch_size=12,
                epochs=10,
                max_seq_len=1024,
                train_encoder=False,
                checkpoint_dir='./checkpoints/freeze_encoder_mlp',
                data_parallel=False,
                seed=None):
    
    model = Classifier(
        ckpt_name, 
        num_labels=len(label2id), 
        train_encoder=train_encoder)

    tokenizer = model.seq_encoder.tokenizer

    train_dataset   = GTDonorDataset(df_train, "Sequence", 
                                     tokenizer, max_seq_len, 
                                     label2id, label_column="Donor"
                                     )
    val_dataset     = GTDonorDataset(df_val, "Sequence", 
                                     tokenizer, max_seq_len, 
                                     label2id, label_column="Donor"
                                     )

    generator = torch.Generator()
    if seed is not None:    
        generator.manual_seed(seed)
    dl_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn(tokenizer), generator=generator)
    dl_val = DataLoader(val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn(tokenizer))


    model.to(torch.device("cuda:0"))
    if data_parallel:
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)

    def compute_metrics_all(logits, labels):
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        y = labels.cpu().numpy()
        preds = (probs>0.5).astype(int)

        # micro/macro f1, etc.
        p_mi, r_mi, f1_mi, _ = precision_recall_fscore_support(y, preds, average="micro")
        p_ma, r_ma, f1_ma, _ = precision_recall_fscore_support(y, preds, average="macro")
        roc_mi = roc_auc_score(y, probs, average="micro")
        # PR‐AUC on flattened array
        prec, rec, _ = precision_recall_curve(y.flatten(), probs.flatten())
        pr_mi = auc(rec, prec)
        mcc = matthews_corrcoef(y.flatten(), preds.flatten())
        acc = accuracy_score(y, preds)
        per_class_metrics = {}
        results = {
            "accuracy": acc,
            "f1_micro":   f1_mi, "precision_micro": p_mi, "recall_micro": r_mi,
            "f1_macro":   f1_ma, "precision_macro": p_ma, "recall_macro": r_ma,
            "roc_auc_micro": roc_mi,
            "pr_auc_micro":  pr_mi,
            "mcc":           mcc
        }
        
        for label, id in label2id.items():
            p, r, f1, _ = precision_recall_fscore_support(y[:,id], preds[:,id], average="binary")
            roc = roc_auc_score(y[:,id], probs[:,id])
            # PR‐AUC on flattened array
            prec, rec, _ = precision_recall_curve(y[:,id], probs[:,id])
            pr = auc(rec, prec)
            mcc = matthews_corrcoef(y[:,id], preds[:,id]) 
            results[f"{label}_precision"] = p
            results[f"{label}_recall"] = r
            results[f"{label}_f1"] = f1
            results[f"{label}_roc_auc"] = roc
            results[f"{label}_pr_auc"] = pr
            results[f"{label}_mcc"] = mcc
        return results

    train_config = {
        "label2id": label2id,
        "lr": lr,
        "batch_size": batch_size,
        "epochs": epochs,
        "max_seq_len": max_seq_len,
        "checkpoint_dir": checkpoint_dir,
        "train_encoder": train_encoder
    }
    
    if os.path.isfile(checkpoint_dir):
        log_path = os.path.dirname(checkpoint_dir) + "train_log.csv"
        with open(os.path.dirname(checkpoint_dir) + "config.json", "w") as f:
            json.dump(train_config, f, indent=4)
    elif os.path.isdir(checkpoint_dir):
        log_path = os.path.join(checkpoint_dir, "train_log.csv")
        with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
            json.dump(train_config, f, indent=4)
    elif checkpoint_dir.endswith(".pth"):
        os.makedirs(os.path.dirname(checkpoint_dir), exist_ok=True)
        log_path = os.path.join(os.path.dirname(checkpoint_dir), "train_log.csv")
        with open(os.path.join(os.path.dirname(checkpoint_dir), "config.json"), "w") as f:
            json.dump(train_config, f, indent=4)
    else:
        os.makedirs(checkpoint_dir, exist_ok=True)
        log_path = os.path.join(checkpoint_dir, "train_log.csv")
        with open(os.path.join(checkpoint_dir, "config.json"), "w") as f:
            json.dump(train_config, f, indent=4)

    with open(log_path, "w") as f:
        f.write("epoch,train_loss,roc_auc_micro,pr_auc_micro,mcc\n")


    best_val_pr_auc = 0.0
    for i in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(dl_train, desc=f"Training {i}"):
            batch = {k:v.to(torch.device("cuda:0")) for k,v in batch.items()}
            labels = batch.pop("labels")
            optimizer.zero_grad()
            outputs,_,_ = model(**batch)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_train_loss = total_loss / len(dl_train)
        wandb.log({"train_loss": avg_train_loss, "learning_rate": optimizer.param_groups[0]["lr"]}, step=i)

        # Validation
        model.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(dl_val, desc=f"Validation {i}"):
                batch = {k:v.to(torch.device("cuda:0")) for k,v in batch.items()}
                labels = batch.pop("labels")
                outputs,_,_ = model(**batch)
                loss = criterion(outputs, labels)
                total_loss += loss.item()
                all_logits.append(outputs.detach().cpu())
                all_labels.append(labels.detach().cpu())
                del outputs, labels, batch
                torch.cuda.empty_cache() # clear cache
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        metrics = compute_metrics_all(all_logits, all_labels)
        wandb.log({"val_" + k: v for k,v in metrics.items()}, step=i)

        with open(log_path, "a") as f:
            f.write(f"{i},{avg_train_loss},{metrics['roc_auc_micro']},{metrics['pr_auc_micro']},{metrics['mcc']}\n")

        if metrics["pr_auc_micro"] > best_val_pr_auc:
            best_val_pr_auc = metrics["pr_auc_micro"]
            if data_parallel:
                model.module.save_checkpoint(checkpoint_dir)
            else:
                model.save_checkpoint(checkpoint_dir)
            wandb.run.summary["best_pr_auc"] = best_val_pr_auc
            wandb.run.summary["best_roc_auc"] = metrics["roc_auc_micro"]
            wandb.run.summary["best_epoch"] = i

        print(f"Epoch {i}: train_loss={avg_train_loss:.4f}  " +
              "  ".join(f"{k}={v:.4f}" for k,v in metrics.items()))


        if data_parallel:
            model.module.load_checkpoint(checkpoint_dir)
        else:
            model.load_checkpoint(checkpoint_dir)
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=str, required=True, help="Path to the dataset.")
    parser.add_argument("--sequence_checkpoint", type=str, default="esmc_600m")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--use_alpha", action="store_true", default=False)
    parser.add_argument("--use_pos_weight", action="store_true", default=False)
    parser.add_argument("--criterion", type=str, default="asl", choices=["bce", "weighted_bce", "focal", "asl"])
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--gamma_pos", type=float, default=0.0)
    parser.add_argument("--clip", type=float, default=0.05)
    parser.add_argument("--continue_training", type=str, default=None)
    parser.add_argument("--train_encoder", action="store_true", default=False)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--data_parallel", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.seed is not None:
        set_seed(args.seed)
        torch.use_deterministic_algorithms(True)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # load your CSVs as before
    df_tr = pd.read_csv(f"../data/{args.fold}/train.csv")
    df_tr["Donor"] = df_tr["Nucleotide_Sugars"].apply(ast.literal_eval)
    # df_te = pd.read_csv(f"5k_test_{fold}_w_saprot.csv")
    df_te = pd.read_csv(f"../data/{args.fold}/test.csv")
    df_te["Donor"] = df_te["Nucleotide_Sugars"].apply(ast.literal_eval)

    # df_exp = pd.read_csv('GT41/donor_labels.csv')
    # df_exp["Donor"] = df_exp["Donor"].apply(ast.literal_eval)

    actual_max_len = df_tr["Sequence"].apply(len).max()
    if actual_max_len < args.max_seq_len:
        args.max_seq_len = actual_max_len

    train_df_exp = df_tr.explode("Donor").groupby("Donor").size().reset_index(name="count")
    test_df_exp = df_te.explode("Donor").groupby("Donor").size().reset_index(name="count")
    train_donor = train_df_exp["Donor"].tolist()
    test_donor = test_df_exp["Donor"].tolist()
    # remove donors with count < 10
    # train_df_exp = train_df_exp[train_df_exp["count"] >= 20]
    all_labels = sorted([donor for donor in train_donor if donor in test_donor])
    label2id = {l:i for i,l in enumerate(all_labels)}
    pos_counts = train_df_exp["count"].values  # shape (C,)

    if args.criterion == "weighted_bce" and args.use_pos_weight:
        pos_counts = torch.as_tensor(pos_counts, dtype=torch.float32)
        N = len(df_tr)
        neg_counts = N - pos_counts
        pos_weight = torch.tensor(neg_counts / (pos_counts + 1e-8),
                                dtype=torch.float32)
        pos_weight   = torch.clamp(pos_weight, max= 10.0, min=0.25)
    else:
        pos_weight = None

    if args.use_alpha:
        alpha = class_alpha_from_counts(pos_counts)
    else:
        alpha = None
    criterion = get_criterion(args.criterion, alpha, pos_weight, args.gamma, args.gamma_pos)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = f"checkpoints/mlp-{'freeze_encoder' if not args.train_encoder else 'train_encoder'}/{args.fold}/" + f"{args.sequence_checkpoint.split('/')[-1]}_" + timestamp
    os.makedirs(checkpoint_dir, exist_ok=True)
    with open(checkpoint_dir + "/label2id.json", "w") as f:
        json.dump(label2id, f)


    config = {
        "label2id": label2id,
        "ckpt_name": args.sequence_checkpoint,
        "train_encoder": args.train_encoder,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size":args.batch_size,
        "criterion": args.criterion,
        "use_alpha": args.use_alpha,
        "epochs": args.epochs,
        "max_seq_len": args.max_seq_len,
        "checkpoint_dir": checkpoint_dir,
        "seed": args.seed
    }
    with open(checkpoint_dir + "/config.pkl", "wb") as f:
        pickle.dump(config, f)
  
    
    wandb.init(project="esmc_unimol", name=f"{'gta' if 'gta' in args.fold else 'gtb'} mlp {'freeze_encoder' if not args.train_encoder else 'train_encoder'}" + timestamp, config=config)
    config.pop("criterion")
    config.pop("use_alpha")
    model = train_model(df_tr, df_te, criterion=criterion,
                        **config, data_parallel=args.data_parallel)

    eval_model(model, df_te, config)
