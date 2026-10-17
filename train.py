#!/usr/bin/env python3
# train.py
# Unified Research Trainer for ReLU-KAN & Hyperspherical VQ Language Models

import os
import gc
import math
import time
import json
import copy
import random
import shutil
import argparse
from pathlib import Path
from array import array

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ---------------------------------------------------------------------------
# Default Configuration
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "experiment_name": "a4_kan_vq_head",
    "model": {
        "vocab_size": 16384,
        "embed_dim": 384,
        "num_layers": 8,
        "nhead": 6,
        "ffn_type": "kan",                # "kan" or "swiglu"
        "mlp_ratio": 2.0,                 # Use 2.0 for kan (grid=4), 3.5 for swiglu to match params!
        "max_seq_len": 512,
        "dropout": 0.0,
        "grid_size": 4,
        "grid_range": [-1.5, 1.5],
        "vq_mode": "head",                # "none", "head", "bottleneck", or "both"
        "bottleneck_layer": 4,            # Layer index after which latent residual VQ is applied
        "bottleneck_codebook_size": 4096, # Codebook size for mid-layer bottleneck
        "bottleneck_alpha": 0.25,         # Residual mixing weight: h = h + alpha * z_q
        "commitment_cost": 0.05,
        "vq_init_std": 0.02,
        "logit_scale": 16.0,
        "tie_embeddings": False
    },
    "data": {
        "local_data_file": None,
        "dataset_name": "HuggingFaceFW/fineweb-edu",
        "dataset_config": "sample-10BT",
        "dataset_split": "train",
        "text_column": "text",
        "data_dir": "./data_cache",
        "max_articles": 1200000,
        "min_char_length": 120,
        "val_fraction": 0.005,
        "tokenize_chunk_size": 4000,
        "tokenizer_sample_docs": 50000,
        "reuse_tokenized_cache": True,
        "special_tokens": ["<|endoftext|>", "<unk>", "<pad>"]
    },
    "training": {
        "seed": 1337,
        "device": "cuda",
        "precision": "bf16",
        "use_compile": False,
        "use_checkpointing": False,
        "batch_size": 32,
        "grad_accum_steps": 2,
        "max_steps": 25000,
        "warmup_steps": 500,
        "rewarm_steps": 500,
        "max_lr": 4e-4,
        "min_lr": 4e-5,
        "continuation_lr": 2e-4,
        "weight_decay": 0.01,
        "betas": [0.9, 0.95],
        "eps": 1e-8,
        "grad_clip": 1.0,
        "log_interval": 50,
        "eval_interval": 500,
        "eval_iters": 25,
        "sample_interval": 2500,
        "checkpoint_interval": 2500,
        "min_free_disk_gb": 2.5
    },
    "inference": {
        "max_new_tokens": 100,
        "temperature": 0.7,
        "top_k": 40,
        "top_p": 0.9,
        "repetition_penalty": 1.15,
        "prompts": [
            "Photosynthesis is the biological process by which",
            "In computer science, a hash table works by",
            "The history of ancient Rome began when"
        ]
    },
    "paths": {
        "checkpoint_dir": "./checkpoints/a4_kan_vq_head",
        "checkpoint_name": "model",
        "config_path": "config.json",
        "plot_filename": "training_curves.png",
        "metrics_jsonl": "metrics.jsonl",
        "plot_dpi": 200
    }
}


# ---------------------------------------------------------------------------
# Crash-Proof Atomic Saver & JSONL Logger
# ---------------------------------------------------------------------------

def cleanup_stale_tmp(ckpt_dir):
    try:
        for p in Path(ckpt_dir).glob("*.tmp"):
            try:
                p.unlink()
            except OSError:
                pass
    except Exception:
        pass


def atomic_torch_save(obj, target_path, retries=5, min_free_gb=2.5):
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        free_gb = shutil.disk_usage(str(target_path.parent)).free / (1024 ** 3)
        if free_gb < min_free_gb:
            print(f"\n[warning] Low disk space ({free_gb:.2f} GB). Skipping checkpoint '{target_path.name}'.\n")
            return False
    except Exception:
        pass

    tmp_path = target_path.with_name(f"{target_path.name}.tmp")
    try:
        torch.save(obj, tmp_path)
        with open(tmp_path, "rb+") as f:
            os.fsync(f.fileno())
        for attempt in range(retries):
            try:
                os.replace(tmp_path, target_path)
                return True
            except OSError:
                time.sleep(0.2 * (attempt + 1))
        os.replace(tmp_path, target_path)
        return True
    except OSError as e:
        print(f"\n[error] Checkpoint write failed ({e}). Continuing safely.\n")
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        return False


def append_jsonl(filepath, record):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def compute_codebook_perplexity(indices, codebook_size):
    """Computes exact information-theoretic codebook perplexity: exp(-sum p_i log p_i)."""
    flat = indices.reshape(-1)
    counts = torch.bincount(flat, minlength=codebook_size).float()
    probs = counts / counts.sum().clamp(min=1.0)
    mask = probs > 0
    entropy = -torch.sum(probs[mask] * torch.log(probs[mask]))
    return torch.exp(entropy).item(), int(mask.sum().item())


# ---------------------------------------------------------------------------
# Model Components: RoPE, FlashAttention, ReLU-KAN, SwiGLU, and VQ Modules
# ---------------------------------------------------------------------------

def _rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_len=4096):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        self.register_buffer("cos2", torch.cat([cos, cos], dim=-1).view(1, 1, max_len, dim), persistent=False)
        self.register_buffer("sin2", torch.cat([sin, sin], dim=-1).view(1, 1, max_len, dim), persistent=False)

    def forward(self, q, k):
        T = q.shape[2]
        cos = self.cos2[:, :, :T].to(dtype=q.dtype, device=q.device)
        sin = self.sin2[:, :, :T].to(dtype=q.dtype, device=q.device)
        return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


class FlashRoPECausalAttention(nn.Module):
    def __init__(self, dim, n_heads=8, max_len=4096, dropout=0.0):
        super().__init__()
        assert dim % n_heads == 0, f"embed_dim ({dim}) must be divisible by nhead ({n_heads})"
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.dropout = dropout
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, max_len=max_len)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2).contiguous()
        q, k = self.rope(q, k)
        drop_p = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop_p, is_causal=True)
        return self.proj(out.transpose(1, 2).reshape(B, T, C))


class ReLUKANLinear(nn.Module):
    def __init__(self, in_features, out_features, grid_size=4, grid_range=(-1.5, 1.5)):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size

        self.base_linear = nn.Linear(in_features, out_features, bias=False)
        grid = torch.linspace(grid_range[0], grid_range[1], grid_size).view(1, 1, 1, grid_size)
        self.register_buffer("grid", grid, persistent=False)

        std = 1.0 / math.sqrt(in_features * grid_size)
        self.kan_weight = nn.Parameter(torch.randn(out_features, in_features * grid_size) * std)

    def forward(self, x):
        base_out = self.base_linear(F.silu(x))
        grid = self.grid.to(dtype=x.dtype, device=x.device)
        x_norm = torch.tanh(x).unsqueeze(-1)
        basis = torch.relu(x_norm - grid).square()
        kan_out = F.linear(basis.flatten(2), self.kan_weight)
        return base_out + kan_out


class GatedKANFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, grid_size=4, grid_range=(-1.5, 1.5), dropout=0.0):
        super().__init__()
        self.gate_kan = ReLUKANLinear(dim, hidden_dim, grid_size=grid_size, grid_range=grid_range)
        self.up_linear = nn.Linear(dim, hidden_dim, bias=False)
        self.down_linear = nn.Linear(hidden_dim, dim, bias=False)
        self.post_norm = nn.LayerNorm(hidden_dim, eps=1e-5)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x):
        g1 = self.gate_kan(x)
        g2 = F.silu(self.up_linear(x))
        gated = self.post_norm(g1.float() * g2.float()).to(dtype=x.dtype)
        return self.dropout(self.down_linear(gated))


class SwiGLUFeedForward(nn.Module):
    """Standard Llama-style SwiGLU FFN for iso-parameter baseline comparisons."""
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.w_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.w_up = nn.Linear(dim, hidden_dim, bias=False)
        self.w_down = nn.Linear(hidden_dim, dim, bias=False)
        self.post_norm = nn.LayerNorm(hidden_dim, eps=1e-5)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x):
        g1 = F.silu(self.w_gate(x))
        g2 = self.w_up(x)
        gated = self.post_norm(g1.float() * g2.float()).to(dtype=x.dtype)
        return self.dropout(self.w_down(gated))


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, ffn_type="kan", mlp_ratio=2.0, grid_size=4,
                 grid_range=(-1.5, 1.5), dropout=0.0, total_layers=8, max_len=4096):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim, eps=1e-5)
        self.attn = FlashRoPECausalAttention(dim, n_heads=n_heads, max_len=max_len, dropout=dropout)
        self.ln2 = nn.LayerNorm(dim, eps=1e-5)
        hidden_dim = int(round(dim * mlp_ratio / 64.0)) * 64  # Multiple of 64 for Tensor Cores
        if ffn_type == "swiglu":
            self.ffn = SwiGLUFeedForward(dim, hidden_dim, dropout=dropout)
        else:
            self.ffn = GatedKANFeedForward(
                dim, hidden_dim, grid_size=grid_size, grid_range=grid_range, dropout=dropout
            )
        self.res_scale = 1.0 / math.sqrt(2.0 * total_layers)

    def forward(self, x):
        x = torch.add(x, self.attn(self.ln1(x)), alpha=self.res_scale)
        x = torch.add(x, self.ffn(self.ln2(x)), alpha=self.res_scale)
        return x


class ResidualHypersphereBottleneckVQ(nn.Module):
    """
    Architecture 2: Mid-Network Residual Hypersphere Vector Quantizer.
    Quantizes normalized hidden states onto a discrete codebook on S^{d-1} and mixes
    them back into the residual stream via Straight-Through Estimator:
        h_out = h + alpha * proj(z_q_ste)
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.05, alpha=0.25, init_std=0.02):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.alpha = alpha

        self.ln = nn.LayerNorm(embedding_dim, eps=1e-5)
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim, bias=False)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=init_std)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=init_std)

    def forward(self, h):
        B, T, C = h.shape
        z = self.ln(h)
        flat_z = F.normalize(z.reshape(-1, C), p=2, dim=-1)
        norm_emb = F.normalize(self.embedding.weight, p=2, dim=-1)

        sim = torch.matmul(flat_z, norm_emb.t())
        indices = torch.argmax(sim, dim=1)
        quantized = norm_emb[indices].view(B, T, C)

        z_norm = flat_z.view(B, T, C)
        e_loss = F.mse_loss(quantized.detach(), z_norm)
        q_loss = F.mse_loss(quantized, z_norm.detach())
        vq_loss = q_loss + self.commitment_cost * e_loss

        quantized_ste = z_norm + (quantized - z_norm).detach()
        h_out = h + self.alpha * self.out_proj(quantized_ste)
        return h_out, vq_loss, indices.view(B, T)


class HypersphereVQOutputHead(nn.Module):
    """
    Architecture 1: Fused Unit-Hypersphere VQ Regularizer & Cosine Classification Head.
    If use_vq=False, acts as a pure Cosine-Scaled Linear Classifier (for ablation A1/A2).
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.05,
                 init_std=0.02, logit_scale=16.0, use_vq=True):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.logit_scale = logit_scale
        self.use_vq = use_vq

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=init_std)

    def forward(self, z):
        B, T, C = z.shape
        flat_z = F.normalize(z.reshape(-1, C), p=2, dim=-1)
        norm_emb = F.normalize(self.embedding.weight, p=2, dim=-1)

        sim = torch.matmul(flat_z, norm_emb.t())
        indices = torch.argmax(sim, dim=1)
        logits = (sim * self.logit_scale).view(B, T, self.num_embeddings)

        if not self.use_vq:
            zero_vq = torch.zeros((), device=z.device, dtype=z.dtype)
            return logits, zero_vq, indices.view(B, T)

        quantized = norm_emb[indices].view(B, T, C)
        z_norm = flat_z.view(B, T, C)
        e_loss = F.mse_loss(quantized.detach(), z_norm)
        q_loss = F.mse_loss(quantized, z_norm.detach())
        vq_loss = q_loss + self.commitment_cost * e_loss
        return logits, vq_loss, indices.view(B, T)


class KAN_VQTransformer(nn.Module):
    def __init__(self, m_cfg, use_checkpointing=False):
        super().__init__()
        self.vocab_size = m_cfg["vocab_size"]
        self.embed_dim = m_cfg["embed_dim"]
        self.max_seq_len = m_cfg["max_seq_len"]
        self.vq_mode = m_cfg.get("vq_mode", "head").lower()
        self.bottleneck_layer = m_cfg.get("bottleneck_layer", m_cfg["num_layers"] // 2)
        self.use_checkpointing = use_checkpointing

        self.token_embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)

        num_layers = m_cfg["num_layers"]
        ffn_type = m_cfg.get("ffn_type", "kan").lower()
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=self.embed_dim,
                n_heads=m_cfg["nhead"],
                ffn_type=ffn_type,
                mlp_ratio=m_cfg.get("mlp_ratio", 2.0),
                grid_size=m_cfg.get("grid_size", 4),
                grid_range=tuple(m_cfg.get("grid_range", [-1.5, 1.5])),
                dropout=m_cfg.get("dropout", 0.0),
                total_layers=num_layers,
                max_len=self.max_seq_len * 2
            )
            for _ in range(num_layers)
        ])

        self.use_bottleneck_vq = self.vq_mode in ("bottleneck", "both")
        self.use_head_vq = self.vq_mode in ("head", "both")

        if self.use_bottleneck_vq:
            bn_codes = m_cfg.get("bottleneck_codebook_size", 4096)
            self.bottleneck_vq = ResidualHypersphereBottleneckVQ(
                num_embeddings=bn_codes,
                embedding_dim=self.embed_dim,
                commitment_cost=m_cfg.get("commitment_cost", 0.05),
                alpha=m_cfg.get("bottleneck_alpha", 0.25),
                init_std=m_cfg.get("vq_init_std", 0.02)
            )
        else:
            self.bottleneck_vq = None

        self.ln_final = nn.LayerNorm(self.embed_dim, eps=1e-5)
        self.output_head = HypersphereVQOutputHead(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embed_dim,
            commitment_cost=m_cfg.get("commitment_cost", 0.05),
            init_std=m_cfg.get("vq_init_std", 0.02),
            logit_scale=m_cfg.get("logit_scale", 16.0),
            use_vq=self.use_head_vq
        )

        if m_cfg.get("tie_embeddings", False):
            self.output_head.embedding.weight = self.token_embedding.weight

    def forward(self, idx, return_hidden=False):
        h = self.token_embedding(idx)
        total_vq_loss = torch.zeros((), device=idx.device, dtype=h.dtype)
        tracked_indices = None

        for layer_idx, block in enumerate(self.blocks, start=1):
            if self.use_checkpointing and self.training:
                h = checkpoint(block, h, use_reentrant=False)
            else:
                h = block(h)

            if self.use_bottleneck_vq and layer_idx == self.bottleneck_layer:
                h, bn_vq_loss, bn_indices = self.bottleneck_vq(h)
                total_vq_loss = total_vq_loss + bn_vq_loss
                tracked_indices = bn_indices

        h_final = self.ln_final(h)
        logits, head_vq_loss, head_indices = self.output_head(h_final)
        total_vq_loss = total_vq_loss + head_vq_loss

        if tracked_indices is None or self.use_head_vq:
            tracked_indices = head_indices

        if return_hidden:
            return logits, total_vq_loss, tracked_indices, h_final
        return logits, total_vq_loss, tracked_indices


# ---------------------------------------------------------------------------
# Data Streaming & Memory-Mapped Cache Builder
# ---------------------------------------------------------------------------

def load_dataset_and_tokenize(cfg):
    from datasets import load_dataset
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder

    d_cfg = cfg["data"]
    m_cfg = cfg["model"]
    vocab_size = m_cfg["vocab_size"]

    data_dir = Path(d_cfg.get("data_dir", "./data_cache"))
    data_dir.mkdir(parents=True, exist_ok=True)

    tok_file = data_dir / f"tokenizer_bpe_{vocab_size}.json"
    max_arts = d_cfg.get("max_articles", 1200000)
    ds_tag = Path(d_cfg["local_data_file"]).stem if d_cfg.get("local_data_file") else d_cfg["dataset_name"].replace("/", "_")
    train_bin = data_dir / f"{ds_tag}_train_v{vocab_size}_n{max_arts}.bin"
    val_bin = data_dir / f"{ds_tag}_val_v{vocab_size}_n{max_arts}.bin"

    dtype = np.uint16 if vocab_size < 65535 else np.int32

    if d_cfg.get("reuse_tokenized_cache", True) and train_bin.exists() and val_bin.exists() and tok_file.exists():
        print(f"[data] Reusing cached binary token streams from {data_dir}:")
        tokenizer = Tokenizer.from_file(str(tok_file))
        train_mmap = np.memmap(train_bin, dtype=dtype, mode="r")
        val_mmap = np.memmap(val_bin, dtype=dtype, mode="r")
        print(f"[data] Memory-mapped {len(train_mmap):,} train | {len(val_mmap):,} val tokens.")
        return train_mmap, val_mmap, tokenizer

    def doc_iterator(limit=None):
        count = 0
        min_len = d_cfg.get("min_char_length", 120)
        if d_cfg.get("local_data_file"):
            with open(d_cfg["local_data_file"], "r", encoding="utf-8") as f:
                for line in f:
                    text = line.strip()
                    if len(text) >= min_len:
                        yield text
                        count += 1
                        if limit and count >= limit:
                            break
        else:
            ds = load_dataset(
                d_cfg["dataset_name"],
                name=d_cfg.get("dataset_config"),
                split=d_cfg.get("dataset_split", "train"),
                streaming=True
            )
            col = d_cfg.get("text_column", "text")
            for item in ds:
                text = (item.get(col) or "").strip()
                if len(text) >= min_len:
                    yield text
                    count += 1
                    if limit and count >= limit:
                        break

    special_tokens = d_cfg.get("special_tokens", ["<|endoftext|>", "<unk>", "<pad>"])
    if tok_file.exists():
        tokenizer = Tokenizer.from_file(str(tok_file))
    else:
        sample_limit = d_cfg.get("tokenizer_sample_docs", 50000)
        print(f"[tokenizer] Streaming {sample_limit:,} docs to train BPE tokenizer (vocab={vocab_size})...")
        sample_docs = list(doc_iterator(limit=sample_limit))
        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = ByteLevel()
        tokenizer.decoder = ByteLevelDecoder()
        trainer = BpeTrainer(vocab_size=vocab_size, min_frequency=2, special_tokens=special_tokens)
        tokenizer.train_from_iterator(sample_docs, trainer)
        tokenizer.save(str(tok_file))
        del sample_docs
        gc.collect()

    eot_id = tokenizer.token_to_id("<|endoftext|>") or 0
    val_interval = int(1.0 / max(d_cfg.get("val_fraction", 0.005), 1e-5))
    chunk_size = d_cfg.get("tokenize_chunk_size", 4000)
    arr_code = "H" if dtype == np.uint16 else "i"

    print(f"[data] Streaming and tokenizing up to {max_arts:,} articles to {data_dir}...")
    chunk_articles = []
    articles_collected = 0
    total_train_tokens, total_val_tokens = 0, 0
    t0 = time.time()

    with open(train_bin, "wb") as f_train, open(val_bin, "wb") as f_val:
        for text in doc_iterator(limit=max_arts):
            chunk_articles.append(text)
            articles_collected += 1

            if len(chunk_articles) >= chunk_size:
                encs = tokenizer.encode_batch(chunk_articles)
                train_chunk, val_chunk = array(arr_code), array(arr_code)
                for doc_i, enc in enumerate(encs):
                    target = val_chunk if (doc_i % val_interval == 0) else train_chunk
                    target.extend(enc.ids)
                    target.append(eot_id)
                if train_chunk:
                    f_train.write(train_chunk.tobytes())
                    total_train_tokens += len(train_chunk)
                if val_chunk:
                    f_val.write(val_chunk.tobytes())
                    total_val_tokens += len(val_chunk)
                chunk_articles.clear()

                if articles_collected % (chunk_size * 10) == 0:
                    rate = (total_train_tokens + total_val_tokens) / max(1e-4, time.time() - t0)
                    print(f"[data] {articles_collected:,} docs | {total_train_tokens + total_val_tokens:,} tok ({rate:,.0f} tok/s)")

        if chunk_articles:
            encs = tokenizer.encode_batch(chunk_articles)
            train_chunk, val_chunk = array(arr_code), array(arr_code)
            for doc_i, enc in enumerate(encs):
                target = val_chunk if (doc_i % val_interval == 0) else train_chunk
                target.extend(enc.ids)
                target.append(eot_id)
            if train_chunk:
                f_train.write(train_chunk.tobytes())
                total_train_tokens += len(train_chunk)
            if val_chunk:
                f_val.write(val_chunk.tobytes())
                total_val_tokens += len(val_chunk)
            chunk_articles.clear()

    print(f"[data] Completed: {total_train_tokens:,} train | {total_val_tokens:,} val tokens.")
    return np.memmap(train_bin, dtype=dtype, mode="r"), np.memmap(val_bin, dtype=dtype, mode="r"), tokenizer


# ---------------------------------------------------------------------------
# Batching, Evaluation & Plotting
# ---------------------------------------------------------------------------

def get_batch(data, batch_size, seq_len, device):
    window_count = len(data) - seq_len - 1
    starts = torch.randint(0, window_count, (batch_size,)).tolist()
    batch = np.empty((batch_size, seq_len + 1), dtype=np.int64)
    for i, s in enumerate(starts):
        batch[i] = data[s : s + seq_len + 1]
    seq_tensor = torch.from_numpy(batch)
    if device.type == "cuda":
        seq_tensor = seq_tensor.pin_memory()
    seq_dev = seq_tensor.to(device, non_blocking=True)
    return seq_dev[:, :-1].contiguous(), seq_dev[:, 1:].contiguous()


@torch.inference_mode()
def evaluate(model, raw_model, val_tokens, batch_size, max_len, device, amp_dtype, use_amp, eval_iters=25):
    was_training = model.training
    model.eval()
    ce_losses, vq_losses = [], []
    all_indices = []
    cb_size = (
        raw_model.bottleneck_vq.num_embeddings
        if (raw_model.use_bottleneck_vq and not raw_model.use_head_vq)
        else raw_model.vocab_size
    )

    for _ in range(eval_iters):
        x, y = get_batch(val_tokens, batch_size, max_len, device)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            logits, vq_loss, indices = model(x)
            ce_loss = F.cross_entropy(logits.view(-1, raw_model.vocab_size), y.view(-1))
        if not (torch.isnan(ce_loss) or torch.isinf(ce_loss)):
            ce_losses.append(ce_loss.item())
            vq_losses.append(vq_loss.item())
            all_indices.append(indices.cpu())

    model.train(was_training)
    mean_ce = sum(ce_losses) / len(ce_losses) if ce_losses else float("inf")
    mean_vq = sum(vq_losses) / len(vq_losses) if vq_losses else 0.0
    val_ppl = math.exp(min(mean_ce, 20.0))
    if all_indices:
        cat_indices = torch.cat(all_indices, dim=0)
        cb_ppl, active_codes = compute_codebook_perplexity(cat_indices, cb_size)
    else:
        cb_ppl, active_codes = 0.0, 0
    return mean_ce, val_ppl, mean_vq, cb_ppl, active_codes


def save_metrics_plot(history_steps, train_ce, val_steps, val_ce, history_cb_ppl, output_path, dpi=200):
    if not history_steps:
        return
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1.plot(history_steps, train_ce, color="#1f77b4", alpha=0.65, linewidth=1.5, label="Train CE Loss")
    if val_steps and val_ce:
        ax1.plot(val_steps, val_ce, color="#ff7f0e", linewidth=2.0, marker="o", markersize=3, label="Val CE Loss")
    ax1.set_ylabel("Cross-Entropy Loss (Pure)", fontsize=11)
    ax1.set_title("Iso-Parameter Training & Validation Dynamics", fontsize=13, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper right")

    ax2.plot(history_steps, history_cb_ppl, color="#2ca02c", linewidth=1.8, label="Batch Codebook Perplexity")
    ax2.set_xlabel("Training Steps", fontsize=11)
    ax2.set_ylabel("Codebook Perplexity (exp(H))", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)
    plt.close()


# ---------------------------------------------------------------------------
# Config & CLI Loader
# ---------------------------------------------------------------------------

def deep_merge(base, override):
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config():
    p = argparse.ArgumentParser(description="Unified KAN-VQ Research Trainer")
    p.add_argument("--config", type=str, default="config.json", help="Path to config JSON")
    p.add_argument("--fresh", action="store_true", help="Start from step 1 ignoring existing checkpoints")
    p.add_argument("--resume-from", type=str, default=None, help="Explicit checkpoint path")
    args = p.parse_args()

    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg_path = Path(args.config)
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = deep_merge(cfg, json.load(f))
    return cfg, args.fresh, args.resume_from


# ---------------------------------------------------------------------------
# Main Loop
# ---------------------------------------------------------------------------

def main():
    cfg, fresh, resume_from = load_config()
    m_cfg, t_cfg, p_cfg = cfg["model"], cfg["training"], cfg["paths"]

    device = torch.device(t_cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    seed = t_cfg.get("seed", 1337)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    prec = t_cfg.get("precision", "bf16").lower()
    use_amp = (device.type == "cuda") and (prec in ("bf16", "bfloat16", "fp16", "float16"))
    if prec in ("bf16", "bfloat16") and device.type == "cuda" and torch.cuda.is_bf16_supported():
        amp_dtype, use_scaler = torch.bfloat16, False
    elif prec in ("fp16", "float16") and device.type == "cuda":
        amp_dtype, use_scaler = torch.float16, True
    else:
        amp_dtype, use_scaler = torch.float32, False

    ckpt_dir = Path(p_cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with open(ckpt_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    ckpt_name = p_cfg.get("checkpoint_name", "model")
    best_path = ckpt_dir / f"{ckpt_name}_best.pt"
    latest_path = ckpt_dir / f"{ckpt_name}_latest.pt"
    plot_path = ckpt_dir / p_cfg.get("plot_filename", "training_curves.png")
    jsonl_path = ckpt_dir / p_cfg.get("metrics_jsonl", "metrics.jsonl")

    cleanup_stale_tmp(ckpt_dir)
    train_tokens, val_tokens, tokenizer = load_dataset_and_tokenize(cfg)

    grad_accum = t_cfg.get("grad_accum_steps", 1)
    batch_size = t_cfg["batch_size"]
    seq_len = m_cfg["max_seq_len"]
    tokens_per_step = batch_size * grad_accum * seq_len
    max_steps = t_cfg["max_steps"]

    model = KAN_VQTransformer(m_cfg, use_checkpointing=t_cfg.get("use_checkpointing", False)).to(device)
    raw_model = model
    if t_cfg.get("use_compile", False) and hasattr(torch, "compile"):
        model = torch.compile(model)

    total_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    cb_size = (
        raw_model.bottleneck_vq.num_embeddings
        if (raw_model.use_bottleneck_vq and not raw_model.use_head_vq)
        else raw_model.vocab_size
    )

    print("=" * 80)
    print(f" Experiment   : {cfg.get('experiment_name', 'unnamed')} | FFN: {m_cfg.get('ffn_type', 'kan').upper()} | VQ Mode: {m_cfg.get('vq_mode', 'head').upper()}")
    print(f" Parameters   : {total_params:,} (~{total_params * 4 / (1024 ** 2):.2f} MB fp32)")
    print(f" Throughput   : SeqLen={seq_len} | Batch={batch_size}x{grad_accum} ({tokens_per_step:,} tok/step) | Steps={max_steps:,}")
    print("=" * 80)

    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        lr=t_cfg["max_lr"],
        weight_decay=t_cfg.get("weight_decay", 0.01),
        betas=tuple(t_cfg.get("betas", [0.9, 0.95])),
        eps=t_cfg.get("eps", 1e-8),
        fused=(device.type == "cuda")
    )
    scaler = torch.amp.GradScaler("cuda", init_scale=32768.0, enabled=use_scaler)

    start_step = 1
    best_val_ce = float("inf")
    tokens_seen = 0
    history_steps, history_train_ce, history_cb_ppl = [], [], []
    val_steps, history_val_ce = [], []

    target_resume = Path(resume_from) if resume_from else (latest_path if not fresh and latest_path.exists() else None)
    if target_resume and target_resume.exists():
        ckpt = torch.load(target_resume, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        start_step = ckpt.get("step", 0) + 1
        best_val_ce = ckpt.get("val_ce", float("inf"))
        tokens_seen = ckpt.get("tokens_seen", 0)
        print(f"[resume] Resumed from Step {start_step} (Best Val CE: {best_val_ce:.4f})")

    t0 = time.time()
    last_log_time = time.time()
    accum_ce, accum_vq = 0.0, 0.0
    steps_since_log = 0
    warmup_steps = t_cfg.get("warmup_steps", 500)
    max_lr, min_lr = t_cfg["max_lr"], t_cfg["min_lr"]

    model.train()
    for step in range(start_step, max_steps + 1):
        if step < warmup_steps:
            cur_lr = max_lr * (step / max(1, warmup_steps))
        else:
            prog = min(1.0, max(0.0, (step - warmup_steps) / max(1, max_steps - warmup_steps)))
            cur_lr = min_lr + (max_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * prog))
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        optimizer.zero_grad(set_to_none=True)
        step_ce, step_vq = 0.0, 0.0
        cb_ppl, unique_codes = 0.0, 0
        skip_step = False

        for micro_i in range(grad_accum):
            x, y = get_batch(train_tokens, batch_size, seq_len, device)
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                logits, vq_loss, indices = model(x)
                ce_loss = F.cross_entropy(logits.view(-1, raw_model.vocab_size), y.view(-1))
                total_loss = ce_loss + vq_loss

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                skip_step = True
                break

            scaler.scale(total_loss / grad_accum).backward()
            step_ce += ce_loss.item() / grad_accum
            step_vq += vq_loss.item() / grad_accum

            if micro_i == grad_accum - 1:
                cb_ppl, unique_codes = compute_codebook_perplexity(indices.detach(), cb_size)

        if skip_step:
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            last_log_time = time.time()
            continue

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=t_cfg.get("grad_clip", 1.0))
        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            last_log_time = time.time()
            continue

        scaler.step(optimizer)
        scaler.update()

        tokens_seen += tokens_per_step
        accum_ce += step_ce
        accum_vq += step_vq
        steps_since_log += 1

        if step % t_cfg["log_interval"] == 0 or step == start_step:
            if device.type == "cuda":
                torch.cuda.synchronize()
            now = time.time()
            tok_s = (steps_since_log * tokens_per_step) / max(1e-5, now - last_log_time)
            avg_ce = accum_ce / steps_since_log
            avg_vq = accum_vq / steps_since_log
            train_ppl = math.exp(min(avg_ce, 20.0))

            history_steps.append(step)
            history_train_ce.append(avg_ce)
            history_cb_ppl.append(cb_ppl)

            vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if device.type == "cuda" else 0.0
            print(
                f"Step {step:05d}/{max_steps} | CE: {avg_ce:.4f} (PPL: {train_ppl:6.2f}) | VQ: {avg_vq:.4f} | "
                f"CB_PPL: {cb_ppl:6.1f} ({unique_codes:4d} codes) | VRAM: {vram_mb:.0f}MB | {tok_s:,.0f} tok/s"
            )
            append_jsonl(jsonl_path, {
                "type": "train", "step": step, "tokens_seen": tokens_seen,
                "train_ce": round(avg_ce, 5), "train_ppl": round(train_ppl, 3),
                "train_vq": round(avg_vq, 5), "cb_ppl": round(cb_ppl, 2),
                "active_codes": unique_codes, "tok_s": round(tok_s, 1), "vram_mb": round(vram_mb, 1)
            })
            accum_ce, accum_vq, steps_since_log = 0.0, 0.0, 0
            last_log_time = time.time()

        if step % t_cfg["eval_interval"] == 0 or step == max_steps:
            val_ce, val_ppl, val_vq, val_cb_ppl, val_codes = evaluate(
                model, raw_model, val_tokens, batch_size, seq_len, device,
                amp_dtype, use_amp, eval_iters=t_cfg.get("eval_iters", 25)
            )
            val_steps.append(step)
            history_val_ce.append(val_ce)

            is_best = (val_ce < best_val_ce) and not math.isnan(val_ce)
            if is_best:
                best_val_ce = val_ce
                atomic_torch_save({
                    "step": step, "model_state": raw_model.state_dict(), "model_config": m_cfg,
                    "val_ce": val_ce, "val_ppl": val_ppl, "val_vq": val_vq,
                    "tokens_seen": tokens_seen, "config": cfg
                }, best_path)

            status = "NEW BEST" if is_best else f"Best: {best_val_ce:.4f}"
            print(f"\n>>> [EVAL Step {step}] Val CE: {val_ce:.4f} | Val PPL: {val_ppl:.2f} | Val VQ: {val_vq:.4f} | CB_PPL: {val_cb_ppl:.1f} ({status}) <<<\n")
            append_jsonl(jsonl_path, {
                "type": "val", "step": step, "tokens_seen": tokens_seen,
                "val_ce": round(val_ce, 5), "val_ppl": round(val_ppl, 3),
                "val_vq": round(val_vq, 5), "val_cb_ppl": round(val_cb_ppl, 2),
                "val_active_codes": val_codes, "is_best": is_best
            })
            save_metrics_plot(history_steps, history_train_ce, val_steps, history_val_ce, history_cb_ppl, plot_path)
            last_log_time = time.time()

        if step % t_cfg["checkpoint_interval"] == 0 or step == max_steps:
            atomic_torch_save({
                "step": step, "model_state": raw_model.state_dict(),
                "optimizer_state": optimizer.state_dict(), "model_config": m_cfg,
                "val_ce": best_val_ce, "tokens_seen": tokens_seen, "config": cfg
            }, latest_path)
            last_log_time = time.time()


if __name__ == "__main__":
    main()