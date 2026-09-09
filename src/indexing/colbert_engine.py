"""
ColBERTv2 Engine for Late-Interaction Retrieval.

Implements token-level multi-vector representations with the MaxSim operator:
    Score(Q, D) = sum_{i in |Q|} max_{j in |D|} (E_{q_i} . E_{d_j})

Uses the standard colbert-ir/colbertv2.0 checkpoint:
- BERT backbone (768-dim)
- Linear projection down to 128-dim
- L2-normalization on every token embedding
- Punctuation & padding masking for exact MaxSim calculation
"""

from __future__ import annotations

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, BertModel
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv

load_dotenv()

if hf_token := os.getenv("HF_TOKEN"):
    os.environ["HF_TOKEN"] = hf_token


class ColBERTv2Model(nn.Module):
    def __init__(self, model_name: str = "colbert-ir/colbertv2.0", device: str | None = None):
        super().__init__()
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Load base BERT
        self.bert = BertModel.from_pretrained(model_name)
        self.linear = nn.Linear(768, 128, bias=False)

        # Load ColBERT projection weights from checkpoint
        fpath = hf_hub_download(repo_id=model_name, filename="pytorch_model.bin")
        state_dict = torch.load(fpath, map_location="cpu")
        if "linear.weight" in state_dict:
            self.linear.weight.data.copy_(state_dict["linear.weight"])

        self.to(self.device)
        self.eval()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Returns:
            L2-normalized token embeddings of shape (batch_size, seq_len, 128)
        """
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        emb = self.linear(out.last_hidden_state)
        return F.normalize(emb, p=2, dim=-1)


class ColBERTv2Engine:
    _instance: ColBERTv2Engine | None = None

    def __init__(self, model_name: str = "colbert-ir/colbertv2.0", device: str | None = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = ColBERTv2Model(model_name=model_name, device=self.device)

        # ColBERT query marker is [unused0], doc marker is [unused1]
        self.query_prefix = "[unused0] "
        self.doc_prefix = "[unused1] "

    @classmethod
    def get_instance(cls, model_name: str = "colbert-ir/colbertv2.0", device: str | None = None) -> ColBERTv2Engine:
        if cls._instance is None:
            cls._instance = cls(model_name=model_name, device=device)
        return cls._instance

    def encode_query(self, query: str, max_length: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encodes a single query string.
        Returns:
            Q_emb: (L_q, 128) tensor on CPU/GPU
            Q_mask: (L_q,) boolean mask for valid query tokens (non-pad)
        """
        formatted_query = self.query_prefix + query.strip()
        encoded = self.tokenizer(
            formatted_query,
            padding=False,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        with torch.no_grad():
            emb = self.model(input_ids, attention_mask)[0]  # (L_q, 128)
        return emb, attention_mask[0].bool()

    def encode_passages(
        self, passages: list[str], max_length: int = 256, batch_size: int = 32
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        Encodes a list of passage texts in batches.
        Returns:
            list of (D_emb, D_mask) for each passage
        """
        formatted = [self.doc_prefix + p.strip() for p in passages]
        results = []

        for i in range(0, len(formatted), batch_size):
            batch_texts = formatted[i : i + batch_size]
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(self.device)
            attention_mask = encoded["attention_mask"].to(self.device)

            with torch.no_grad():
                batch_emb = self.model(input_ids, attention_mask)  # (B, L_d, 128)

            for b in range(len(batch_texts)):
                mask = attention_mask[b].bool()
                valid_len = mask.sum().item()
                results.append((batch_emb[b][:valid_len], mask[:valid_len]))

        return results

    def score_query_against_passages(
        self, query: str, passages: list[str], batch_size: int = 32
    ) -> list[float]:
        """
        Computes exact ColBERT MaxSim scores between a query and a list of passages.
        Returns:
            list of float scores matching the order of `passages`
        """
        if not passages:
            return []

        Q, q_mask = self.encode_query(query)
        # Filter only valid query tokens
        Q = Q[q_mask]  # (L_q_valid, 128)

        passage_encodings = self.encode_passages(passages, batch_size=batch_size)
        scores = []

        for D, d_mask in passage_encodings:
            # D shape: (L_d_valid, 128)
            # Dot product similarity matrix: (L_q_valid, L_d_valid)
            sim_matrix = torch.matmul(Q, D.T)
            # MaxSim: for each query token, find max cosine similarity across document tokens
            max_sim_per_token, _ = sim_matrix.max(dim=-1)
            # Sum over query tokens
            score = max_sim_per_token.sum().item()
            scores.append(float(score))

        return scores
