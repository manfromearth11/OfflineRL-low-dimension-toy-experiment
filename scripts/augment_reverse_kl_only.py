#!/usr/bin/env python3
"""Augment existing 4-axis results with reverse_kl only.

This script does NOT rerun OT/L2/KL sweeps.
It trains reverse_kl per (scenario, modality, q_quality, dim, seed),
then merges reverse_kl metrics into:
  - per-cell summary CSV
  - per-cell winner_table CSV
  - global grid CSV
and regenerates per-cell reward curve plots.
"""

from __future__ import annotations

import argparse
import copy
import csv
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.runners.run_3axis_regime_sweep import (
    MODALITY_PROFILES,
    Q_PROFILES,
    SCENARIO_PROFILES,
    build_cfg,
    load_yaml,
)
from src.runners.run_ot_wins_dim_sweep import (
    StaticDiagGaussian,
    aggregate,
    build_action_transform,
    build_modes_for_dim,
    corrupted_q_torch,
    evaluate_metrics,
    plot_reward_curves,
    sample_behavior,
    train_bc,
    train_reverse_kl,
)


def parse_str_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def read_csv(path: Path) -> List[dict]:
    with open(path) as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[dict], fieldnames: List[str]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def coerce_summary_rows(rows: List[dict]) -> List[dict]:
    out: List[dict] = []
    for r in rows:
        out.append(
            {
                "dim": int(float(r["dim"])),
                "method": str(r["method"]),
                "lambda": float(r["lambda"]),
                "n_seeds": int(float(r["n_seeds"])),
                "mean_reward_mean": float(r["mean_reward_mean"]),
                "mean_reward_std": float(r["mean_reward_std"]),
                "bad_mass_mean": float(r["bad_mass_mean"]),
                "bad_mass_std": float(r["bad_mass_std"]),
                "good_recall_mean": float(r["good_recall_mean"]),
                "good_recall_std": float(r["good_recall_std"]),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", type=str, default="configs/ot_wins_dim_sweep.yaml")
    parser.add_argument("--prefix", type=str, default="regime_4axis_scenario_balanced")
    parser.add_argument("--scenarios", type=str, default="baseline,good_shift,rotated,anchor_corrupt")
    parser.add_argument("--modalities", type=str, default="low,mid,high")
    parser.add_argument("--q-levels", type=str, default="clean,mid,noisy")
    parser.add_argument("--scenario-level", type=str, default="moderate")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if args.device == "cpu":
        torch.set_num_threads(max(1, torch.get_num_threads()))

    project_root = Path(__file__).resolve().parents[1]
    results_dir = project_root / "results"
    grid_path = results_dir / f"{args.prefix}_grid.csv"
    if not grid_path.exists():
        raise FileNotFoundError(f"Grid not found: {grid_path}")

    base_cfg = load_yaml(project_root / args.base_config)
    scenarios = parse_str_list(args.scenarios)
    modalities = parse_str_list(args.modalities)
    q_levels = parse_str_list(args.q_levels)
    scenario_level = args.scenario_level

    dims = [int(x) for x in base_cfg["env"]["action_dims"]]
    seeds = [int(x) for x in base_cfg["experiment"]["seeds"]]

    print("=" * 88)
    print("AUGMENT REVERSE_KL ONLY")
    print(f"prefix={args.prefix}")
    print(f"dims={dims} seeds={seeds}")
    print(f"scenarios={scenarios}")
    print(f"modalities={modalities}")
    print(f"q_levels={q_levels}")
    print("=" * 88)

    # Keep reverse_kl mean reward per cell for grid merge.
    rkl_cell_mean: Dict[Tuple[str, str, str, int], float] = {}

    total = len(scenarios) * len(modalities) * len(q_levels)
    idx = 0
    for s in scenarios:
        for m in modalities:
            for q in q_levels:
                idx += 1
                cell_name = f"{args.prefix}_{s}_{m}_{q}"
                summary_path = results_dir / f"{cell_name}_summary.csv"
                winner_path = results_dir / f"{cell_name}_winner_table.csv"
                raw_path = results_dir / f"{cell_name}_raw.csv"
                reward_png = results_dir / f"{cell_name}_reward_curves.png"
                if not summary_path.exists() or not winner_path.exists():
                    raise FileNotFoundError(f"Missing per-cell files for {cell_name}")

                cfg = build_cfg(copy.deepcopy(base_cfg), modality=m, q_quality=q, scenario=s, scenario_level=scenario_level)
                env_cfg = cfg["env"]
                corr_cfg = cfg["corruption"]
                train_cfg = cfg["training"]
                kl_cfg = cfg["kl"]
                kl_reverse_cfg = cfg.get("kl_reverse", {})
                eval_cfg = cfg["eval"]

                print(f"\n[{idx}/{total}] {s}/{m}/{q}: training reverse_kl only...")
                rkl_raw_records: List[dict] = []
                for dim in dims:
                    means, stds, weights = build_modes_for_dim(env_cfg, dim)
                    action_transform = build_action_transform(
                        dim,
                        float(env_cfg.get("anisotropy_ratio", 1.0)),
                        float(env_cfg.get("rotation_deg", 0.0)),
                    )
                    reward_center = np.zeros(dim, dtype=np.float32)
                    reward_center[0] = float(env_cfg.get("reward_center_shift", 0.0))

                    means_t = torch.tensor(means, dtype=torch.float32)
                    bad_ids_t = torch.tensor(env_cfg["bad_mode_ids"], dtype=torch.long)
                    reward_center_t = torch.tensor(reward_center, dtype=torch.float32)

                    for seed in seeds:
                        rng = np.random.default_rng(seed)
                        a_np, _ = sample_behavior(
                            int(env_cfg["n_data"]),
                            means,
                            stds,
                            weights,
                            rng,
                            action_transform=action_transform,
                        )
                        actions_t = torch.tensor(a_np, dtype=torch.float32)
                        qhat_t = corrupted_q_torch(
                            actions_t,
                            means_t,
                            bad_ids_t,
                            reward_center_t=reward_center_t,
                            q_noise_std=float(corr_cfg["q_noise_std"]),
                            q_bias_bad=float(corr_cfg["q_bias_bad"]),
                            bad_spike_prob=float(corr_cfg["bad_spike_prob"]),
                            bad_spike_value=float(corr_cfg["bad_spike_value"]),
                        ).detach()

                        bc = StaticDiagGaussian(dim)
                        _ = train_bc(bc, actions_t, train_cfg, seed=seed * 100 + 1)
                        behavior_model = StaticDiagGaussian(dim)
                        behavior_model.load_state_dict(bc.state_dict())
                        behavior_model.eval()
                        for p in behavior_model.parameters():
                            p.requires_grad_(False)

                        rkl = StaticDiagGaussian(dim)
                        rkl_loss = train_reverse_kl(
                            rkl,
                            behavior_model=behavior_model,
                            actions_t=actions_t,
                            qhat_t=qhat_t,
                            train_cfg=train_cfg,
                            alpha=float(kl_reverse_cfg.get("alpha", kl_cfg.get("alpha", 0.2))),
                            beta=float(kl_reverse_cfg.get("beta", 0.1)),
                            seed=seed * 100 + 5,
                        )

                        met = evaluate_metrics(
                            rkl,
                            means,
                            stds,
                            list(env_cfg["good_mode_ids"]),
                            list(env_cfg["bad_mode_ids"]),
                            reward_center_t=reward_center_t,
                            n_eval=int(eval_cfg["n_eval_samples"]),
                            recall_radius_mult=float(eval_cfg["recall_radius_mult"]),
                        )
                        rkl_raw_records.append(
                            {
                                "dim": int(dim),
                                "seed": int(seed),
                                "method": "reverse_kl",
                                "lambda": 1.0,
                                "mean_reward": float(met["mean_reward"]),
                                "bad_mass": float(met["bad_mass"]),
                                "good_recall": float(met["good_recall"]),
                                "final_loss": float(rkl_loss),
                            }
                        )

                # Build reverse_kl summary rows.
                rkl_summary = [r for r in aggregate(rkl_raw_records) if r["method"] == "reverse_kl"]
                rkl_by_dim = {int(r["dim"]): r for r in rkl_summary}

                # Merge into summary CSV.
                summary_rows = read_csv(summary_path)
                summary_rows = [r for r in summary_rows if r["method"] != "reverse_kl"]
                summary_rows.extend(
                    [
                        {
                            "dim": int(r["dim"]),
                            "method": "reverse_kl",
                            "lambda": float(r["lambda"]),
                            "n_seeds": int(r["n_seeds"]),
                            "mean_reward_mean": float(r["mean_reward_mean"]),
                            "mean_reward_std": float(r["mean_reward_std"]),
                            "bad_mass_mean": float(r["bad_mass_mean"]),
                            "bad_mass_std": float(r["bad_mass_std"]),
                            "good_recall_mean": float(r["good_recall_mean"]),
                            "good_recall_std": float(r["good_recall_std"]),
                        }
                        for r in rkl_summary
                    ]
                )
                # Deterministic order: dim -> method -> lambda.
                summary_rows = sorted(
                    summary_rows,
                    key=lambda r: (int(float(r["dim"])), str(r["method"]), float(r["lambda"])),
                )
                write_csv(
                    summary_path,
                    summary_rows,
                    fieldnames=[
                        "dim",
                        "method",
                        "lambda",
                        "n_seeds",
                        "mean_reward_mean",
                        "mean_reward_std",
                        "bad_mass_mean",
                        "bad_mass_std",
                        "good_recall_mean",
                        "good_recall_std",
                    ],
                )

                # Merge into winner table.
                winner_rows = read_csv(winner_path)
                for wr in winner_rows:
                    dim = int(float(wr["dim"]))
                    rkl_reward = float(rkl_by_dim[dim]["mean_reward_mean"])
                    wr["reverse_kl_mean_reward"] = rkl_reward
                    best_ot = max(
                        float(wr["best_wasserstein_reward"]),
                        float(wr["best_partial_reward"]),
                        float(wr["best_unbalanced_reward"]),
                    )
                    wr["best_ot_minus_reverse_kl"] = best_ot - rkl_reward
                    wr["best_l2_minus_reverse_kl"] = float(wr["best_l2_reward"]) - rkl_reward

                    rkl_cell_mean[(s, m, q, dim)] = rkl_reward

                write_csv(
                    winner_path,
                    winner_rows,
                    fieldnames=[
                        "dim",
                        "forward_kl_mean_reward",
                        "reverse_kl_mean_reward",
                        "best_wasserstein_lambda",
                        "best_wasserstein_reward",
                        "best_partial_lambda",
                        "best_partial_reward",
                        "best_unbalanced_lambda",
                        "best_unbalanced_reward",
                        "best_partial_minus_kl",
                        "best_l2_lambda",
                        "best_l2_reward",
                        "best_l2_minus_kl",
                        "best_ot_minus_reverse_kl",
                        "best_l2_minus_reverse_kl",
                        "all_ot_beat_kl",
                    ],
                )

                # Optionally append reverse_kl rows to raw CSV.
                if raw_path.exists():
                    raw_rows = read_csv(raw_path)
                    raw_rows = [r for r in raw_rows if r["method"] != "reverse_kl"]
                    raw_rows.extend(rkl_raw_records)
                    raw_rows = sorted(
                        raw_rows,
                        key=lambda r: (
                            int(float(r["dim"])),
                            int(float(r["seed"])),
                            str(r["method"]),
                            float(r["lambda"]),
                        ),
                    )
                    write_csv(
                        raw_path,
                        raw_rows,
                        fieldnames=[
                            "dim",
                            "seed",
                            "method",
                            "lambda",
                            "mean_reward",
                            "bad_mass",
                            "good_recall",
                            "final_loss",
                        ],
                    )

                # Regenerate reward curve png so reverse_kl line appears.
                plot_reward_curves(coerce_summary_rows(summary_rows), str(reward_png))
                print(f"  merged and re-plotted: {cell_name}")

    # Merge into grid CSV.
    grid_rows = read_csv(grid_path)
    for gr in grid_rows:
        key = (gr["scenario"], gr["modality"], gr["q_quality"], int(float(gr["dim"])))
        rkl = rkl_cell_mean.get(key, float("nan"))
        gr["reverse_kl_mean_reward"] = rkl
        best_ot = max(
            float(gr["best_wasserstein_reward"]),
            float(gr["best_partial_reward"]),
            float(gr["best_unbalanced_reward"]),
        )
        gr["best_ot_minus_reverse_kl"] = best_ot - rkl if rkl == rkl else float("nan")
        gr["best_l2_minus_reverse_kl"] = (
            float(gr["best_l2_reward"]) - rkl if rkl == rkl else float("nan")
        )

    write_csv(
        grid_path,
        grid_rows,
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

    print("\nDone. reverse_kl merged into all existing 4-axis artifacts.")


if __name__ == "__main__":
    main()
