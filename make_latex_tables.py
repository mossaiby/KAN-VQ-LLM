#!/usr/bin/env python3
# make_latex_tables.py
# Reads completed ablation checkpoints and prints LaTeX rows for Table 1

import json
from pathlib import Path
import torch

EXPERIMENTS = [
    ("a1_baseline_swiglu",  r"\texttt{A1}", r"SwiGLU Baseline ($\rho=3.5$, No VQ)"),
    ("a2_kan_only",         r"\texttt{A2}", r"Gated ReLU-KAN ($\rho=2.0$, No VQ)"),
    ("a3_swiglu_vq_head",   r"\texttt{A3}", r"SwiGLU + VQ Output Head"),
    ("a4_kan_vq_head",      r"\texttt{A4}", r"Gated ReLU-KAN + VQ Output Head"),
    ("a5_kan_vq_bottleneck",r"\texttt{A5}", r"Gated ReLU-KAN + Residual VQ Bottleneck"),
    ("a6_kan_vq_both",      r"\texttt{A6}", r"\textbf{Gated ReLU-KAN + Both (Head + BN)}"),
]

print("% --- Copy-paste into Table 1 of main.tex ---")
for folder, code_lbl, desc in EXPERIMENTS:
    ckpt_dir = Path("checkpoints") / folder
    best_pt = ckpt_dir / "model_best.pt"
    jsonl_file = ckpt_dir / "metrics.jsonl"

    if not best_pt.exists():
        print(f"% [missing] {best_pt}")
        continue

    ckpt = torch.load(best_pt, map_location="cpu", weights_only=False)
    n_params = sum(p.numel() for p in ckpt["model_state"].values()) / 1e6
    val_ce = ckpt.get("val_ce", 0.0)
    val_ppl = ckpt.get("val_ppl", 0.0)

    # Read median tok/s and final val_cb_ppl from metrics.jsonl
    tok_s_list, val_cb_ppl = [], "N/A"
    if jsonl_file.exists():
        with open(jsonl_file, "r") as f:
            for line in f:
                row = json.loads(line)
                if row.get("type") == "train" and "tok_s" in row:
                    tok_s_list.append(row["tok_s"])
                elif row.get("type") == "val" and row.get("is_best"):
                    if ckpt["config"]["model"]["vq_mode"] != "none":
                        val_cb_ppl = f"{row.get('val_cb_ppl', 0.0):.1f}"

    mean_tok_s = f"{int(sum(tok_s_list[-20:]) / max(1, len(tok_s_list[-20:]))):,}" if tok_s_list else "--"
    print(f"{code_lbl} & {desc} & {n_params:.1f}M & {val_ce:.4f} & {val_ppl:.2f} & {val_cb_ppl} & -- & {mean_tok_s} \\\\")