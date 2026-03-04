"""4-axis regime sweep:
scenario level x modality level x Q-quality level x action dimension.

This wrapper reuses `src/runners/run_ot_wins_dim_sweep.py` and aggregates winners.
File name is kept for backward compatibility.
"""

from __future__ import annotations

import argparse
import copy
import csv
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNNER = PROJECT_ROOT / "src" / "runners" / "run_ot_wins_dim_sweep.py"
RESULTS_DIR = PROJECT_ROOT / "results"


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_str_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


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


def reset_results_dir() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for p in RESULTS_DIR.iterdir():
        if p.is_dir():
            shutil.rmtree(p)
        else:
            p.unlink()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Reset results directory: {RESULTS_DIR}")


MODALITY_PROFILES: Dict[str, dict] = {
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
    "clean": {
        "q_noise_std": 0.0,
        "q_bias_bad": 0.0,
        "bad_spike_prob": 0.0,
        "bad_spike_value": 0.0,
    },
    "mid": {
        "q_noise_std": 2.0,
        "q_bias_bad": 0.3,
        "bad_spike_prob": 0.0,
        "bad_spike_value": 0.0,
    },
    "noisy": {
        "q_noise_std": 5.0,
        "q_bias_bad": 1.0,
        "bad_spike_prob": 0.01,
        "bad_spike_value": 10.0,
    },
}


SCENARIO_PROFILES: Dict[str, Dict[str, dict]] = {
    "moderate": {
        "baseline": {
            "mean_shift": 0.0,
            "reward_center_shift": 0.0,
            "anisotropy_ratio": 1.0,
            "rotation_deg": 0.0,
            "l2_anchor_noise_std": 0.0,
        },
        "good_shift": {
            "mean_shift": 1.0,
            "reward_center_shift": 0.0,
            "anisotropy_ratio": 1.0,
            "rotation_deg": 0.0,
            "l2_anchor_noise_std": 0.0,
        },
        "rotated": {
            "mean_shift": 0.0,
            "reward_center_shift": 0.0,
            "anisotropy_ratio": 3.0,
            "rotation_deg": 35.0,
            "l2_anchor_noise_std": 0.0,
        },
        "anchor_corrupt": {
            "mean_shift": 0.0,
            "reward_center_shift": 0.0,
            "anisotropy_ratio": 1.0,
            "rotation_deg": 0.0,
            "l2_anchor_noise_std": 0.35,
        },
    }
}


def build_cfg(base: dict, modality: str, q_quality: str, scenario: str, scenario_level: str) -> dict:
    cfg = copy.deepcopy(base)
    m = MODALITY_PROFILES[modality]
    q = Q_PROFILES[q_quality]
    s = SCENARIO_PROFILES[scenario_level][scenario]

    cfg["env"]["offsets"] = copy.deepcopy(m["offsets"])
    cfg["env"]["stds"] = copy.deepcopy(m["stds"])
    cfg["env"]["weights"] = copy.deepcopy(m["weights"])

    cfg["corruption"]["q_noise_std"] = float(q["q_noise_std"])
    cfg["corruption"]["q_bias_bad"] = float(q["q_bias_bad"])
    cfg["corruption"]["bad_spike_prob"] = float(q["bad_spike_prob"])
    cfg["corruption"]["bad_spike_value"] = float(q["bad_spike_value"])

    cfg["env"]["mean_shift"] = float(s["mean_shift"])
    cfg["env"]["reward_center_shift"] = float(s["reward_center_shift"])
    cfg["env"]["anisotropy_ratio"] = float(s["anisotropy_ratio"])
    cfg["env"]["rotation_deg"] = float(s["rotation_deg"])

    cfg.setdefault("l2_constraint", {})
    cfg["l2_constraint"]["anchor_noise_std"] = float(s["l2_anchor_noise_std"])
    return cfg


def plot_single_metric_heatmaps(
    rows: List[dict],
    dims: List[int],
    scenarios: List[str],
    modalities: List[str],
    q_levels: List[str],
    prefix: str,
    metric_key: str,
    metric_title: str,
    filename_suffix: str,
) -> None:
    for scenario in scenarios:
        for modality in modalities:
            sub = [r for r in rows if r["scenario"] == scenario and r["modality"] == modality]
            mat = np.full((len(q_levels), len(dims)), np.nan, dtype=np.float64)

            for r in sub:
                yi = q_levels.index(r["q_quality"])
                xi = dims.index(int(r["dim"]))
                mat[yi, xi] = float(r[metric_key])

            fig, ax = plt.subplots(figsize=(5.5, 4.0))
            im = ax.imshow(mat, cmap="RdYlGn", aspect="auto")
            ax.set_xticks(range(len(dims)), labels=[str(d) for d in dims])
            ax.set_yticks(range(len(q_levels)), labels=q_levels)
            ax.set_xlabel("dimension")
            ax.set_ylabel("Q quality")

            for y in range(len(q_levels)):
                for x in range(len(dims)):
                    if np.isfinite(mat[y, x]):
                        ax.text(x, y, f"{mat[y, x]:.3f}", ha="center", va="center", fontsize=8)

            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            out_path = RESULTS_DIR / f"{prefix}_heatmap_{scenario}_{modality}_{filename_suffix}.png"
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=str, default="configs/ot_wins_dim_sweep.yaml")
    parser.add_argument("--dims", type=str, default="2,4,8")
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--n-data", type=int, default=15000)
    parser.add_argument("--scenarios", type=str, default="baseline,good_shift,rotated,anchor_corrupt")
    parser.add_argument("--scenario-level", type=str, default="moderate")
    parser.add_argument("--modalities", type=str, default="low,mid,high")
    parser.add_argument("--q-levels", type=str, default="clean,mid,noisy")
    parser.add_argument("--output-prefix", type=str, default="regime_4axis_scenario_balanced")
    parser.add_argument("--reset-results", action="store_true")
    args = parser.parse_args()

    if args.reset_results:
        reset_results_dir()

    dims = parse_int_list(args.dims)
    seeds = parse_int_list(args.seeds)
    scenarios = parse_str_list(args.scenarios)
    modalities = parse_str_list(args.modalities)
    q_levels = parse_str_list(args.q_levels)

    if args.scenario_level not in SCENARIO_PROFILES:
        raise ValueError(f"Unknown scenario level: {args.scenario_level}")

    for s in scenarios:
        if s not in SCENARIO_PROFILES[args.scenario_level]:
            raise ValueError(f"Unknown scenario: {s}")
    for m in modalities:
        if m not in MODALITY_PROFILES:
            raise ValueError(f"Unknown modality level: {m}")
    for q in q_levels:
        if q not in Q_PROFILES:
            raise ValueError(f"Unknown Q quality level: {q}")

    base_cfg = load_yaml(PROJECT_ROOT / args.base_config)
    tmp_dir = RESULTS_DIR / "_tmp_regime_cfgs"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[dict] = []
    total = len(scenarios) * len(modalities) * len(q_levels)
    idx = 0

    for s in scenarios:
        for m in modalities:
            for q in q_levels:
                idx += 1
                cfg = build_cfg(base_cfg, modality=m, q_quality=q, scenario=s, scenario_level=args.scenario_level)
                cfg_path = tmp_dir / f"{args.output_prefix}_{s}_{m}_{q}.yaml"
                save_yaml(cfg_path, cfg)

                run_prefix = f"{args.output_prefix}_{s}_{m}_{q}"
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
                print(f"[{idx}/{total}] scenario={s}, modality={m}, q_quality={q}")
                print(" ".join(cmd))
                subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))

                winner_path = RESULTS_DIR / f"{run_prefix}_winner_table.csv"
                winners = read_winner_table(winner_path)
                for r in winners:
                    kl = float(r["forward_kl_mean_reward"])
                    rkl = float(r.get("reverse_kl_mean_reward", "nan"))
                    w = float(r["best_wasserstein_reward"])
                    p = float(r["best_partial_reward"])
                    u = float(r["best_unbalanced_reward"])
                    l2 = float(r["best_l2_reward"])
                    best_ot = max(w, p, u)
                    all_rows.append(
                        {
                            "scenario": s,
                            "modality": m,
                            "q_quality": q,
                            "dim": int(r["dim"]),
                            "forward_kl_mean_reward": kl,
                            "reverse_kl_mean_reward": rkl,
                            "best_wasserstein_reward": w,
                            "best_partial_reward": p,
                            "best_unbalanced_reward": u,
                            "best_l2_reward": l2,
                            "best_ot_minus_kl": best_ot - kl,
                            "best_ot_minus_reverse_kl": (
                                best_ot - rkl if rkl == rkl else float("nan")
                            ),
                            "best_pot_minus_kl": float(r["best_partial_minus_kl"]),
                            "best_l2_minus_kl": float(r["best_l2_minus_kl"]),
                            "best_l2_minus_reverse_kl": float(
                                r.get("best_l2_minus_reverse_kl", "nan")
                            ),
                            "all_ot_beat_kl": int(r["all_ot_beat_kl"]),
                        }
                    )

    out_csv = RESULTS_DIR / f"{args.output_prefix}_grid.csv"
    save_csv(
        out_csv,
        all_rows,
        fieldnames=[
            "scenario",
            "modality",
            "q_quality",
            "dim",
            "forward_kl_mean_reward",
            "reverse_kl_mean_reward",
            "best_wasserstein_reward",
            "best_partial_reward",
            "best_unbalanced_reward",
            "best_l2_reward",
            "best_ot_minus_kl",
            "best_ot_minus_reverse_kl",
            "best_pot_minus_kl",
            "best_l2_minus_kl",
            "best_l2_minus_reverse_kl",
            "all_ot_beat_kl",
        ],
    )

    plot_single_metric_heatmaps(
        all_rows,
        dims=dims,
        scenarios=scenarios,
        modalities=modalities,
        q_levels=q_levels,
        prefix=args.output_prefix,
        metric_key="best_pot_minus_kl",
        metric_title="best POT - KL",
        filename_suffix="pot",
    )
    plot_single_metric_heatmaps(
        all_rows,
        dims=dims,
        scenarios=scenarios,
        modalities=modalities,
        q_levels=q_levels,
        prefix=args.output_prefix,
        metric_key="best_l2_minus_kl",
        metric_title="best L2 - KL",
        filename_suffix="l2",
    )

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    print("Done.")


if __name__ == "__main__":
    main()
