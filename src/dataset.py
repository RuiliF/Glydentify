import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import random
from transformers import AutoTokenizer
from esm.utils import encoding
from esm.utils.misc import stack_variable_length_tensors


class GTDonorDataset(Dataset):
    def __init__(self, df: pd.DataFrame, seq_column: str,
                 tokenizer,
                 max_seq_length: int=None,
                 label2id: dict=None,
                 label_column: str=None,
                 is_esmc: bool=False):
        """
        df: contains columns [seq_column, 'Donor'] where Donor is a list of donor‐labels
        label2id: donor_label -> integer index
        """
        self.df = df.reset_index(drop=True)
        self.seq_column = seq_column
        self.label_column = label_column
        self.label2id = label2id
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.is_esmc = is_esmc

        # Precompute all donor embeddings once and freeze
        self.num_labels = len(self.label2id) if self.label2id else 0

    def _encode_labels(self, donor_list):
        assert self.label2id is not None
        vec = np.zeros(self.num_labels, dtype=np.float32)
        for d in donor_list:
            if d in self.label2id:
                vec[self.label2id[d]] = 1.
        return torch.from_numpy(vec)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = row[self.seq_column]
        
        # tokenize with padding/truncation
        if self.seq_column == "SaProt Sequence":
            seq_len = len(seq)//2
            if seq_len > self.max_seq_length:
                start = random.randint(0, seq_len - self.max_seq_length)
                seq = seq[start*2:(start + self.max_seq_length)*2]
        else:
            if len(seq) > self.max_seq_length:
                start = random.randint(0, len(seq) - self.max_seq_length)
                seq = seq[start:start + self.max_seq_length]

        if self.is_esmc:
            toks = encoding.tokenize_sequence(seq, self.tokenizer, add_special_tokens=True)
            ret = {"sequence_tokens": toks}
        else:
            toks = self.tokenizer(seq,
                                  padding=False,
                                  truncation=True,
                                  max_length=self.max_seq_length+2,
                                  return_tensors="pt")
            # drop batch dim
            ret = {k:v.squeeze(0) for k,v in toks.items()}

        if self.label_column:
            if self.label2id:
                labels = self._encode_labels(row[self.label_column])
            else:
                labels = row[self.label_column]
            ret["labels"] = labels
        # Else: do not add "labels" key to ret to avoid default_collate failure on NoneType
            
        return ret

def get_collate_fn(tokenizer, is_esmc=False):
    if is_esmc:
        def collate(batch):
            sequence_tokens = stack_variable_length_tensors(
                [item["sequence_tokens"] for item in batch],
                constant_value=tokenizer.pad_token_id,
            )
            if "labels" in batch[0] and batch[0]["labels"] is not None:
                labels = torch.stack([item["labels"] for item in batch])
                return {"sequence_tokens": sequence_tokens, "labels": labels}
            else:
                return {"sequence_tokens": sequence_tokens}
        return collate
    else:
        # Dynamic padding: pad to the longest sequence in each batch
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        def collate(batch):
            input_ids = [item["input_ids"] for item in batch]
            max_len = max(x.size(0) for x in input_ids)
            padded_ids = torch.stack([
                torch.nn.functional.pad(x, (0, max_len - x.size(0)), value=pad_id)
                for x in input_ids
            ])
            attention_mask = torch.stack([
                torch.nn.functional.pad(torch.ones(x.size(0), dtype=torch.long), (0, max_len - x.size(0)), value=0)
                for x in input_ids
            ])
            result = {"input_ids": padded_ids, "attention_mask": attention_mask}
            if "token_type_ids" in batch[0]:
                token_type_ids = [item["token_type_ids"] for item in batch]
                result["token_type_ids"] = torch.stack([
                    torch.nn.functional.pad(x, (0, max_len - x.size(0)), value=0)
                    for x in token_type_ids
                ])
            if "labels" in batch[0] and batch[0]["labels"] is not None:
                result["labels"] = torch.stack([item["labels"] for item in batch])
            return result
        return collate
