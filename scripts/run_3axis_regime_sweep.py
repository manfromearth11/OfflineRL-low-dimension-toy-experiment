"""3-axis regime sweep:
modality level x Q-quality level x action dimension.

This wrapper reuses `scripts/run_ot_wins_dim_sweep.py` and aggregates winners.
"""

from __future__ import annotations

import argparse
import copy
import csv
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER = PROJECT_ROOT / "scripts" / "run_ot_wins_dim_sweep.py"


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_yaml(path: Path, obj: dict) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def read_winner_table(path: Path) -> List[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def save_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Saved: {path}")


MODALITY_PROFILES: Dict[str, dict] = {
    # Easy modality: less separated / less bad mass.
    "low": {
        "offsets": {"mediocre": 0.4, "bad": 2.5},
        "stds": {"good": 0.12, "mediocre": 0.50, "bad": 0.50},
        "weights": {
            "good": 0.45,
            "mediocre_pos": 0.225,
            "mediocre_neg": 0.225,
            "bad_pos": 0.05,
            "bad_neg": 0.05,
        },
    },
    # Reference modality (current default in ot_wins_dim_sweep.yaml).
    "mid": {
        "offsets": {"mediocre": 0.8, "bad": 5.0},
        "stds": {"good": 0.10, "mediocre": 0.40, "bad": 0.20},
        "weights": {
            "good": 0.20,
            "mediocre_pos": 0.15,
            "mediocre_neg": 0.15,
            "bad_pos": 0.25,
            "bad_neg": 0.25,
        },
    },
    # Hard modality: far bad modes with dominant bad mass.
    "high": {
        "offsets": {"mediocre": 1.0, "bad": 6.0},
        "stds": {"good": 0.08, "mediocre": 0.35, "bad": 0.15},
        "weights": {
            "good": 0.10,
            "mediocre_pos": 0.10,
            "mediocre_neg": 0.10,
            "bad_pos": 0.35,
            "bad_neg": 0.35,
        },
    },
}


Q_PROFILES: Dict[str, dict] = {
    # High-quality Q signal.
    "clean": {
        "q_noise_std": 0.0,
        "q_bias_bad": 0.0,
        "bad_spike_prob": 0.0,
        "bad_spike_value": 0.0,
    },
    # Moderate corruption.
    "mid": {
        "q_noise_std": 2.0,
        "q_bias_bad": 0.3,
        "bad_spike_prob": 0.0,
        "bad_spike_value": 0.0,
    },
    # Low-SNR / adversarial corruption.
    "noisy": {
        "q_noise_std": 5.0,
        "q_bias_bad": 1.0,
        "bad_spike_prob": 0.01,
        "bad_spike_value": 10.0,
    },
}


def build_cfg(base: dict, modality: str, q_quality: str) -> dict:
    cfg = copy.deepcopy(base)
    m = MODALITY_PROFILES[modality]
    q = Q_PROFILES[q_quality]

    cfg["env"]["offsets"] = copy.deepcopy(m["offsets"])
    cfg["env"]["stds"] = copy.deepcopy(m["stds"])
    cfg["env"]["weights"] = copy.deepcopy(m["weights"])
    cfg["corruption"]["q_noise_std"] = float(q["q_noise_std"])
    cfg["corruption"]["q_bias_bad"] = float(q["q_bias_bad"])
    cfg["corruption"]["bad_spike_prob"] = float(q["bad_spike_prob"])
    cfg["corruption"]["bad_spike_value"] = float(q["bad_spike_value"])
    return cfg


def plot_heatmaps(rows: List[dict], dims: List[int], prefix: str) -> None:
    out_dir = PROJECT_ROOT / "results"
    q_levels = ["clean", "mid", "noisy"]
    m_levels = ["low", "mid", "high"]

    for modality in m_levels:
        sub = [r for r in rows if r["modality"] == modality]

        best_minus_kl = np.full((len(q_levels), len(dims)), np.nan, dtype=np.float64)
        all_ot_win = np.full((len(q_levels), len(dims)), np.nan, dtype=np.float64)

        for r in sub:
            yi = q_levels.index(r["q_quality"])
            xi = dims.index(int(r["dim"]))
            best_minus_kl[yi, xi] = float(r["best_ot_minus_kl"])
            all_ot_win[yi, xi] = float(r["all_ot_beat_kl"])

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        im0 = axes[0].imshow(best_minus_kl, cmap="RdYlGn", aspect="auto")
        axes[0].set_title(f"{modality}: best(OT)-KL")
        axes[0].set_xticks(range(len(dims)), labels=[str(d) for d in dims])
        axes[0].set_yticks(range(len(q_levels)), labels=q_levels)
        for y in range(len(q_levels)):
            for x in range(len(dims)):
                if np.isfinite(best_minus_kl[y, x]):
                    axes[0].text(x, y, f"{best_minus_kl[y, x]:.3f}", ha="center", va="center", fontsize=9)
        fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        im1 = axes[1].imshow(all_ot_win, cmap="Blues", vmin=0.0, vmax=1.0, aspect="auto")
        axes[1].set_title(f"{modality}: all_ot_beat_kl")
        axes[1].set_xticks(range(len(dims)), labels=[str(d) for d in dims])
        axes[1].set_yticks(range(len(q_levels)), labels=q_levels)
        for y in range(len(q_levels)):
            for x in range(len(dims)):
                if np.isfinite(all_ot_win[y, x]):
                    axes[1].text(x, y, f"{int(all_ot_win[y, x])}", ha="center", va="center", fontsize=10)
        fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        fig.suptitle(f"3-axis regime map (modality={modality})", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        out_path = out_dir / f"{prefix}_heatmap_{modality}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=str, default="configs/ot_wins_dim_sweep.yaml")
    parser.add_argument("--dims", type=str, default="3,4,5,8")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--n-data", type=int, default=15000)
    parser.add_argument("--output-prefix", type=str, default="regime_3axis")
    args = parser.parse_args()

    dims = parse_int_list(args.dims)
    seeds = parse_int_list(args.seeds)

    base_cfg = load_yaml(PROJECT_ROOT / args.base_config)
    tmp_dir = PROJECT_ROOT / "results" / "_tmp_regime_cfgs"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[dict] = []
    m_levels = ["low", "mid", "high"]
    q_levels = ["clean", "mid", "noisy"]

    total = len(m_levels) * len(q_levels)
    idx = 0
    for m in m_levels:
        for q in q_levels:
            idx += 1
            cfg = build_cfg(base_cfg, modality=m, q_quality=q)
            cfg_path = tmp_dir / f"{args.output_prefix}_{m}_{q}.yaml"
            save_yaml(cfg_path, cfg)

            run_prefix = f"{args.output_prefix}_{m}_{q}"
            cmd = [
                sys.executable,
                str(RUNNER),
                "--config",
                str(cfg_path),
                "--dims",
                ",".join(map(str, dims)),
                "--seeds",
                ",".join(map(str, seeds)),
                "--epochs",
                str(args.epochs),
                "--n-data",
                str(args.n_data),
                "--output-prefix",
                run_prefix,
            ]
            print("=" * 90)
            print(f"[{idx}/{total}] modality={m}, q_quality={q}")
            print(" ".join(cmd))
            subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))

            winner_path = PROJECT_ROOT / "results" / f"{run_prefix}_winner_table.csv"
            winners = read_winner_table(winner_path)
            for r in winners:
                kl = float(r["forward_kl_mean_reward"])
                w = float(r["best_wasserstein_reward"])
                p = float(r["best_partial_reward"])
                u = float(r["best_unbalanced_reward"])
                best_ot = max(w, p, u)
                all_rows.append(
                    {
                        "modality": m,
                        "q_quality": q,
                        "dim": int(r["dim"]),
                        "forward_kl_mean_reward": kl,
                        "best_wasserstein_reward": w,
                        "best_partial_reward": p,
                        "best_unbalanced_reward": u,
                        "best_ot_minus_kl": best_ot - kl,
                        "all_ot_beat_kl": int(r["all_ot_beat_kl"]),
                    }
                )

    out_csv = PROJECT_ROOT / "results" / f"{args.output_prefix}_grid.csv"
    save_csv(
        out_csv,
        all_rows,
        fieldnames=[
            "modality",
            "q_quality",
            "dim",
            "forward_kl_mean_reward",
            "best_wasserstein_reward",
            "best_partial_reward",
            "best_unbalanced_reward",
            "best_ot_minus_kl",
            "all_ot_beat_kl",
        ],
    )
    plot_heatmaps(all_rows, dims=dims, prefix=args.output_prefix)
    print("Done.")


if __name__ == "__main__":
    main()
