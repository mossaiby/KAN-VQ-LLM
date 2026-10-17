# `KAN-VQ-LLM`: Scaling ReLU-KAN Language Models with Hyperspherical Vector Quantization

Official PyTorch implementation for training, ablating, and evaluating autoregressive language models using **Tensor-Core Flattened Gated ReLU-KAN** blocks paired with **Hyperspherical Vector Quantization (VQ)**. 

This repository supports controlled iso-parameter and iso-FLOP comparisons between standard Llama-style **SwiGLU** baselines and **Gated ReLU-KAN** architectures across four quantization regimes (`none`, `head`, `bottleneck`, and `both`) on `FineWeb-Edu` and `TinyStories`.

---

## Architectural Highlights

```text
Input Tokens: x_{1:T}
    │
    ▼
Token Embedding ──► Latent dim d (RoPE applied inside Attention)
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ Transformer Blocks 1 .. l_b                                     │
│   ├── LayerNorm ──► FlashRoPECausalAttention                    │
│   └── LayerNorm ──► FFN: GatedKANFeedForward OR SwiGLU          │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼  [Optional: vq_mode in ("bottleneck", "both")]
┌─────────────────────────────────────────────────────────────────┐
│ ResidualHypersphereBottleneckVQ (at Layer l_b = L // 2)         │
│   ├── Project hidden state u onto unit hypersphere S^{d-1}      │
│   ├── Quantize against latent codebook C (M = 4,096 codes)      │
│   └── Straight-Through Residual Mix: h = h + α · W_bn(q_STE)    │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ Transformer Blocks (l_b + 1) .. L                               │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│ HypersphereVQOutputHead [vq_mode in ("head", "both", "none")]   │
│   ├── L2-normalize final state ẑ and output codebook Ê          │
│   ├── Single Cosine GEMM: sim = ẑ · Êᵀ                          │
│   ├── Classification Logits: logits = 16.0 · sim                │
│   └── Optional VQ Commitment Loss on argmax(sim)                │
└─────────────────────────────────────────────────────────────────┘
```

### Core Innovations
1. **Tensor-Core Flattened Gated ReLU-KAN (`GatedKANFeedForward`):** Evaluates squared-ReLU basis functions $\phi_k(x) = \text{ReLU}(\tanh(x) - g_k)^2$ and flattens the basis tensor into a contiguous 2D matrix multiplication (`F.linear`), bypassing slow 4D `torch.einsum` reductions and saturating GPU Tensor Cores.
2. **Fused Hyperspherical VQ Output Head (`vq_mode = "head"`):** Shares a single normalized cosine similarity matrix $\hat{E}\hat{z}$ between discrete codebook commitment and temperature-scaled Cross-Entropy classification—adding zero extra matrix multiplications.
3. **Mid-Network Residual VQ Bottleneck (`vq_mode = "bottleneck"`):** Quantizes intermediate layer representations onto a discrete hyperspherical codebook via Straight-Through Estimation (STE) and gates them back into the residual stream.
4. **Crash-Proof Streaming & Atomic Checkpointing:** Streams multi-billion-token datasets (`FineWeb-Edu`) directly into `uint16`/`int32` memory-mapped binary files (`np.memmap`) with near-zero RAM usage, protected by disk-space checks, `os.fsync` atomic checkpointing, and an automatic NaN/Inf gradient shield.

---

## Repository Structure

```text
KAN-VQ-LLM/
├── README.md                     # Documentation & execution guide
├── requirements.txt              # Python package dependencies
├── train.py                      # Unified trainer (SwiGLU/KAN + VQ modes: none/head/bottleneck/both)
├── generate.py                   # Standalone interactive CLI & single-prompt generator
├── evaluate_benchmarks.py        # Hypersphere isotropy, codebook perplexity & benchmark evaluator
├── make_latex_tables.py          # Automated LaTeX table generator from metrics.jsonl & checkpoints
├── run_ablations.sh              # One-command bash runner for the 6-run ablation suite + flagship configs
└── configs/                      # Generated automatically by run_ablations.sh (or created manually)
    ├── a1_baseline_swiglu.json   # Tier 1 (42M): SwiGLU Baseline (mlp_ratio=3.5, vq_mode=none)
    ├── a2_kan_only.json          # Tier 1 (42M): Gated ReLU-KAN (mlp_ratio=2.0, vq_mode=none)
    ├── a3_swiglu_vq_head.json    # Tier 1 (42M): SwiGLU + VQ Output Head
    ├── a4_kan_vq_head.json       # Tier 1 (42M): Gated ReLU-KAN + VQ Output Head
    ├── a5_kan_vq_bottleneck.json # Tier 1 (42M): Gated ReLU-KAN + Mid-Layer Residual VQ Bottleneck
    ├── a6_kan_vq_both.json       # Tier 1 (42M): Gated ReLU-KAN + Both VQ Bottleneck & VQ Head
    ├── flagship_125m_swiglu.json # Tier 2 (125M, 2.5B tokens): Chinchilla SwiGLU Baseline
    └── flagship_125m_kan_vq.json # Tier 2 (125M, 2.5B tokens): Chinchilla KAN-VQ Flagship
```

---

## 1. Installation & Setup

Requires **Python 3.10+** and a CUDA-capable GPU supporting native `bfloat16` (e.g., NVIDIA RTX 3090 / 4090 / A100).

```bash
git clone [https://github.com/mossaiby/KAN-VQ-LLM.git](https://github.com/mossaiby/KAN-VQ-LLM.git)
cd KAN-VQ-LLM

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

*(Optional)* Set your Hugging Face token to avoid rate-limiting when streaming `FineWeb-Edu`:
```bash
export HF_TOKEN="hf_your_token_here"
```

---

## 2. Commands to Run

### Option A: Run the Complete 6-Model Ablation Suite Automatically
The `run_ablations.sh` script automatically generates all 6 Tier-1 ablation JSON files (`a1` through `a6`) and the 2 Tier-2 Flagship 125M JSON files inside `./configs/`, trains all 6 Tier-1 models sequentially, and runs `evaluate_benchmarks.py` after each run.

> **Note:** On the first run (`a1_baseline_swiglu`), the script streams and tokenizes `FineWeb-Edu` into `./data_cache/`. Runs `a2` through `a6` automatically memory-map the cached `.bin` files in less than 2 seconds.

```bash
chmod +x run_ablations.sh
./run_ablations.sh
```

---

### Option B: Run Individual Experiments Manually

First, generate the configuration files in `./configs/` without starting training:
```bash
python3 -c '
import json, copy
from pathlib import Path
from train import DEFAULT_CONFIG

Path("configs").mkdir(exist_ok=True)
ablations = [
    ("a1_baseline_swiglu",  "swiglu", 3.5, "none"),
    ("a2_kan_only",         "kan",    2.0, "none"),
    ("a3_swiglu_vq_head",   "swiglu", 3.5, "head"),
    ("a4_kan_vq_head",      "kan",    2.0, "head"),
    ("a5_kan_vq_bottleneck","kan",    2.0, "bottleneck"),
    ("a6_kan_vq_both",      "kan",    2.0, "both"),
]
for name, ffn, ratio, vq_mode in ablations:
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["experiment_name"] = name
    cfg["model"]["ffn_type"] = ffn
    cfg["model"]["mlp_ratio"] = ratio
    cfg["model"]["vq_mode"] = vq_mode
    cfg["paths"]["checkpoint_dir"] = f"./checkpoints/{name}"
    with open(f"configs/{name}.json", "w") as f:
        json.dump(cfg, f, indent=2)
print("Generated configs in ./configs/")
'
```

Now run any single ablation experiment (or resume an interrupted run automatically):
```bash
# Train Architecture 1: Gated ReLU-KAN + Hyperspherical VQ Output Head
python3 train.py --config configs/a4_kan_vq_head.json

# Train Architecture 2: Gated ReLU-KAN + Mid-Network Residual VQ Bottleneck
python3 train.py --config configs/a5_kan_vq_bottleneck.json

# Train Combined Architecture: Gated ReLU-KAN + Both VQ Bottleneck & Head
python3 train.py --config configs/a6_kan_vq_both.json

# Start a clean run ignoring any existing checkpoint in the folder
python3 train.py --config configs/a4_kan_vq_head.json --fresh
```

---

### Option C: Run the 125M Flagship Chinchilla Experiment (2.5B Tokens on RTX 3090 / 4090)

Once `configs/flagship_125m_kan_vq.json` and `configs/flagship_125m_swiglu.json` are generated, launch the 125M flagship runs:

```bash
# 1. Train the 125M KAN-VQ Flagship Model (~2.5B tokens, activation checkpointing enabled)
python3 train.py --config configs/flagship_125m_kan_vq.json

# 2. Train the 125M Iso-Parameter SwiGLU Baseline (~2.5B tokens)
python3 train.py --config configs/flagship_125m_swiglu.json
```

---

## 3. Evaluation, Geometry Analysis & LaTeX Table Generation

### Evaluate Hypersphere Isotropy & Codebook Perplexity
Run `evaluate_benchmarks.py` on any saved `model_best.pt` checkpoint to compute pure Cross-Entropy loss, validation perplexity, mean pairwise cosine similarity, the **Hypersphere Isotropy Index** $\mathcal{I} \in [0, 1]$, and **Codebook Perplexity**:

```bash
python3 evaluate_benchmarks.py \
  --config configs/a6_kan_vq_both.json \
  --checkpoint checkpoints/a6_kan_vq_both/model_best.pt
```

### Generate LaTeX Rows for the Paper Automatically
After your ablation runs complete, run `make_latex_tables.py` to parse all `checkpoints/*/model_best.pt` and `metrics.jsonl` files and output formatted LaTeX table rows ready to paste into `main.tex`:

```bash
python3 make_latex_tables.py
```

---

## 4. Standalone Interactive Generation (`generate.py`)

Test any trained checkpoint interactively in a terminal REPL or pass a single prompt via CLI:

```bash
# Launch interactive REPL (auto-detects tokenizer in ./data_cache)
python3 generate.py \
  --config configs/a6_kan_vq_both.json \
  --checkpoint checkpoints/a6_kan_vq_both/model_best.pt

# Single-prompt generation with custom sampling parameters
python3 generate.py \
  --config configs/a6_kan_vq_both.json \
  --checkpoint checkpoints/a6_kan_vq_both/model_best.pt \
  --prompt "Photosynthesis is the biological process by which" \
  --max-new-tokens 120 \
  --temperature 0.65 \
  --top-k 40 \
  --top-p 0.90
```

---

## 5. Ablation Matrix Reference

All Tier-1 models use `vocab_size = 16384`, `embed_dim = 384`, `num_layers = 8`, `nhead = 6`, `max_seq_len = 512`, and `32,768 tokens/step` (`batch_size = 32`, `grad_accum_steps = 2`) for `25,000 steps` (~819M tokens):

| Config File | `ffn_type` | `mlp_ratio` | `vq_mode` | Description |
| :--- | :---: | :---: | :---: | :--- |
| `configs/a1_baseline_swiglu.json` | `swiglu` | `3.5` | `none` | Standard Transformer baseline with RoPE, FlashAttention-2, and SwiGLU FFN. |
| `configs/a2_kan_only.json` | `kan` | `2.0` | `none` | Isolates `GatedKANFeedForward` at matched parameter count without VQ commitment. |
| `configs/a3_swiglu_vq_head.json` | `swiglu` | `3.5` | `head` | Tests Hyperspherical VQ Output Regularization on a standard SwiGLU Transformer. |
| `configs/a4_kan_vq_head.json` | `kan` | `2.0` | `head` | **Architecture 1:** Gated ReLU-KAN + Fused Hyperspherical VQ Output Head. |
| `configs/a5_kan_vq_bottleneck.json` | `kan` | `2.0` | `bottleneck` | **Architecture 2:** Gated ReLU-KAN + Mid-Layer Residual VQ Bottleneck (`layer=4`). |
| `configs/a6_kan_vq_both.json` | `kan` | `2.0` | `both` | **Combined:** Gated ReLU-KAN with both Mid-Layer Residual VQ and VQ Output Head. |

---

## Citation

If you use this codebase or build upon the KAN-VQ architecture in your research, please cite:

```bibtex
@article{kanvqlm2026,
  title   = {Scaling ReLU-KAN Language Models with Fused Hyperspherical Vector Quantization: An Iso-Compute Study on FineWeb-Edu},
  author  = {Author Name},
  journal = {Preprint},
  year    = {2026}
}
```