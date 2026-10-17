#!/usr/bin/env python3
# run_ablations.py
# Cross-platform experiment manager (Windows / Linux / macOS) for KAN-VQ LM ablations

import sys
import json
import copy
import time
import argparse
import subprocess
from pathlib import Path

from train import DEFAULT_CONFIG


TIER1_ABLATIONS = [
    ("a1_baseline_swiglu",   "swiglu", 3.5, "none"),
    ("a2_kan_only",          "kan",    2.0, "none"),
    ("a3_swiglu_vq_head",    "swiglu", 3.5, "head"),
    ("a4_kan_vq_head",       "kan",    2.0, "head"),
    ("a5_kan_vq_bottleneck", "kan",    2.0, "bottleneck"),
    ("a6_kan_vq_both",       "kan",    2.0, "both"),
]

TIER2_FLAGSHIPS = [
    ("flagship_125m_swiglu", "swiglu", 3.5, "none"),
    ("flagship_125m_kan_vq", "kan",    2.0, "both"),
]


def generate_configs(config_dir: Path) -> None:
    config_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate Tier 1 Ablation Configs (~42M params, 820M tokens)
    for name, ffn, ratio, vq_mode in TIER1_ABLATIONS:
        cfg = copy.deepcopy(DEFAULT_CONFIG)
        cfg["experiment_name"] = name
        cfg["model"]["ffn_type"] = ffn
        cfg["model"]["mlp_ratio"] = ratio
        cfg["model"]["vq_mode"] = vq_mode
        cfg["paths"]["checkpoint_dir"] = str(Path("checkpoints") / name)

        out_file = config_dir / f"{name}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

    # 2. Generate Tier 2 Flagship Configs (~125M params, 2.5B tokens)
    for name, ffn, ratio, vq_mode in TIER2_FLAGSHIPS:
        cfg = copy.deepcopy(DEFAULT_CONFIG)
        cfg["experiment_name"] = name
        cfg["model"].update({
            "vocab_size": 32768,
            "embed_dim": 640,
            "num_layers": 14,
            "nhead": 10,
            "ffn_type": ffn,
            "mlp_ratio": ratio,
            "max_seq_len": 1024,
            "vq_mode": vq_mode,
            "bottleneck_layer": 7,
        })
        cfg["data"]["max_articles"] = 3000000
        cfg["training"].update({
            "batch_size": 16,
            "grad_accum_steps": 2,
            "max_steps": 76000,
            "warmup_steps": 1000,
            "use_checkpointing": True,
        })
        cfg["paths"]["checkpoint_dir"] = str(Path("checkpoints") / name)

        out_file = config_dir / f"{name}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

    print(f"[setup] Generated {len(TIER1_ABLATIONS)} ablation configs and "
          f"{len(TIER2_FLAGSHIPS)} flagship configs in '{config_dir}/'")


def run_command(cmd: list[str]) -> None:
    print(f"\n[exec] {' '.join(cmd)}\n")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {' '.join(cmd)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cross-Platform Ablation & Flagship Runner for KAN-VQ LM"
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="configs",
        help="Directory to store/read generated JSON configs (default: configs)"
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Only generate the JSON configuration files and exit without training"
    )
    parser.add_argument(
        "--tier",
        choices=["tier1", "flagship", "all"],
        default="tier1",
        help="Which experiment suite to run: 'tier1' (6 ablations), 'flagship' (125M), or 'all'"
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Run only specific experiment names (e.g., --only a1_baseline_swiglu a4_kan_vq_head)"
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Pass --fresh to train.py to ignore existing checkpoints"
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip running evaluate_benchmarks.py after each training run"
    )
    parser.add_argument(
        "--skip-config-gen",
        action="store_true",
        help="Do not overwrite existing JSON files in configs/ before running"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_dir = Path(args.config_dir)

    if not args.skip_config_gen:
        generate_configs(config_dir)

    if args.generate_only:
        print("[done] Config generation complete (--generate-only specified).")
        return

    # Select target experiments
    experiments = []
    if args.tier in ("tier1", "all"):
        experiments.extend([name for name, _, _, _ in TIER1_ABLATIONS])
    if args.tier in ("flagship", "all"):
        experiments.extend([name for name, _, _, _ in TIER2_FLAGSHIPS])

    if args.only:
        valid_set = set(experiments) | {n for n, _, _, _ in TIER1_ABLATIONS + TIER2_FLAGSHIPS}
        for req in args.only:
            if req not in valid_set:
                print(f"[warning] Requested experiment '{req}' is not in standard templates, "
                      f"will look for '{config_dir / (req + '.json')}'.")
        experiments = args.only

    python_exe = sys.executable
    suite_start = time.time()

    for idx, exp_name in enumerate(experiments, start=1):
        cfg_file = config_dir / f"{exp_name}.json"
        if not cfg_file.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_file}")

        ckpt_file = Path("checkpoints") / exp_name / "model_best.pt"

        print("\n" + "=" * 72)
        print(f" [{idx}/{len(experiments)}] Starting Experiment: {exp_name}")
        print("=" * 72)

        train_cmd = [python_exe, "train.py", "--config", str(cfg_file)]
        if args.fresh:
            train_cmd.append("--fresh")
        run_command(train_cmd)

        if not args.skip_eval:
            if ckpt_file.exists():
                eval_cmd = [
                    python_exe,
                    "evaluate_benchmarks.py",
                    "--config", str(cfg_file),
                    "--checkpoint", str(ckpt_file),
                ]
                run_command(eval_cmd)
            else:
                print(f"[warning] Best checkpoint not found at {ckpt_file}; skipping evaluation.")

    # Automatically print the LaTeX summary table if make_latex_tables.py exists
    latex_script = Path("make_latex_tables.py")
    if latex_script.exists():
        print("\n" + "=" * 72)
        print(" Generating LaTeX Summary Table Rows")
        print("=" * 72)
        run_command([python_exe, str(latex_script)])

    total_hours = (time.time() - suite_start) / 3600.0
    print("\n" + "=" * 72)
    print(f" All {len(experiments)} experiment(s) finished in {total_hours:.2f} hours.")
    print("=" * 72)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[interrupted] Execution stopped by user.")
        sys.exit(130)