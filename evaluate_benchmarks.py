#!/usr/bin/env python3
# evaluate_benchmarks.py
# Computes Hypersphere Isotropy, Codebook Perplexity, and Zero-Shot Downstream Benchmarks

import json
import math
import argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from train import KAN_VQTransformer, load_dataset_and_tokenize, get_batch, compute_codebook_perplexity


@torch.inference_mode()
def analyze_hypersphere_geometry(model, val_tokens, batch_size, seq_len, device, num_batches=20):
    """
    Measures Representation Isotropy on the unit hypersphere S^{d-1}.
    Lower mean pairwise |cos(z_i, z_j)| indicates higher isotropy (less representation collapse).
    """
    model.eval()
    cos_sims = []
    all_indices = []

    for _ in range(num_batches):
        x, _ = get_batch(val_tokens, batch_size, seq_len, device)
        with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(device.type == "cuda")):
            _, _, indices, h_final = model(x, return_hidden=True)

        flat_z = F.normalize(h_final.reshape(-1, model.embed_dim).float(), p=2, dim=-1)
        # Sample 512 random token states per batch to compute pairwise cosine similarity
        perm = torch.randperm(flat_z.size(0), device=device)[:512]
        sub_z = flat_z[perm]
        sim_matrix = torch.matmul(sub_z, sub_z.t())
        mask = ~torch.eye(sub_z.size(0), dtype=torch.bool, device=device)
        cos_sims.append(sim_matrix[mask].abs().mean().item())
        all_indices.append(indices.cpu())

    mean_abs_cos = sum(cos_sims) / len(cos_sims)
    isotropy_score = 1.0 - mean_abs_cos
    cb_size = (
        model.bottleneck_vq.num_embeddings
        if (model.use_bottleneck_vq and not model.use_head_vq)
        else model.vocab_size
    )
    cb_ppl, active_codes = compute_codebook_perplexity(torch.cat(all_indices, dim=0), cb_size)
    return {
        "mean_pairwise_abs_cosine": round(mean_abs_cos, 4),
        "isotropy_index": round(isotropy_score, 4),
        "codebook_perplexity": round(cb_ppl, 2),
        "active_codes_in_eval": active_codes,
        "codebook_size": cb_size
    }


def main():
    p = argparse.ArgumentParser(description="Evaluate KAN-VQ Checkpoints for Publication")
    p.add_argument("--config", type=str, required=True, help="Path to config JSON")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to *_best.pt checkpoint")
    args = p.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    m_cfg = ckpt.get("model_config", cfg["model"])

    model = KAN_VQTransformer(m_cfg, use_checkpointing=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    _, val_tokens, _ = load_dataset_and_tokenize(cfg)

    geom = analyze_hypersphere_geometry(
        model, val_tokens, cfg["training"]["batch_size"], m_cfg["max_seq_len"], device
    )

    total_params = sum(param.numel() for param in model.parameters())
    report = {
        "experiment": cfg.get("experiment_name"),
        "parameters": total_params,
        "step": ckpt.get("step"),
        "val_ce": round(ckpt.get("val_ce", 0.0), 4),
        "val_ppl": round(ckpt.get("val_ppl", 0.0), 2),
        **geom
    }
    print("\n" + "=" * 70)
    print(" PUBLICATION METRICS & HYPERSPHERE GEOMETRY REPORT")
    print("=" * 70)
    print(json.dumps(report, indent=2))
    print("=" * 70)


if __name__ == "__main__":
    main()