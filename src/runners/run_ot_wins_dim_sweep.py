"""Dimension sweep toy: KL failure vs constrained methods under Q misranking.

Designed toy (single-state bandit):
  - action dims: configurable (e.g., 3, 4, 5, 8)
  - behavior modes: good + mediocre +/- + catastrophic bad +/-
  - true reward: r(a) = -||a||^2
  - corrupted Q: Qhat = r + gaussian_noise + bad_bias + rare_bad_spike

Methods:
  - BC
  - Forward KL (exp reweighting)
  - Reverse KL (AWR-style weighted MLE + KL(pi||beta) penalty)
  - Wasserstein OT
  - Partial OT (potential-based replacement)
  - Unbalanced OT
  - L2 constraint baseline
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import ot as pot
import torch
import torch.nn as nn
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


class StaticDiagGaussian(nn.Module):
    """State-free diagonal Gaussian policy over action space."""

    def __init__(self, action_dim: int, init_std: float = 0.6):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(action_dim))
        self.log_std = nn.Parameter(torch.full((action_dim,), math.log(init_std)))

    def _std(self) -> torch.Tensor:
        return self.log_std.clamp(-4.0, 2.0).exp()

    def rsample(self, n: int) -> torch.Tensor:
        std = self._std()
        eps = torch.randn(n, std.shape[0], device=std.device)
        return self.mu.unsqueeze(0) + std.unsqueeze(0) * eps

    def log_prob(self, a: torch.Tensor) -> torch.Tensor:
        std = self._std()
        log_std = std.log()
        z = (a - self.mu.unsqueeze(0)) / std.unsqueeze(0)
        return -0.5 * (z.pow(2) + 2.0 * log_std.unsqueeze(0) + math.log(2.0 * math.pi)).sum(dim=-1)


class PotentialMLP(nn.Module):
    """Potential network f_omega(s, a); s is constant in this toy."""

    def __init__(self, action_dim: int, hidden_dim: int, output_nonnegative: bool = True):
        super().__init__()
        self.output_nonnegative = output_nonnegative
        self.net = nn.Sequential(
            nn.Linear(action_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.softplus = nn.Softplus()

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        s = torch.zeros(actions.shape[0], 1, dtype=actions.dtype, device=actions.device)
        out = self.net(torch.cat([s, actions], dim=-1)).squeeze(-1)
        if self.output_nonnegative:
            out = self.softplus(out)
        return out


def build_modes_for_dim(env_cfg: dict, dim: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Construct 5-mode template for a given dimension."""
    off_med = float(env_cfg["offsets"]["mediocre"])
    off_bad = float(env_cfg["offsets"]["bad"])
    mean_shift = float(env_cfg.get("mean_shift", 0.0))
    std_good = float(env_cfg["stds"]["good"])
    std_med = float(env_cfg["stds"]["mediocre"])
    std_bad = float(env_cfg["stds"]["bad"])
    w = env_cfg["weights"]

    def vec(x: float) -> np.ndarray:
        out = np.zeros(dim, dtype=np.float32)
        out[0] = x
        return out

    means = np.stack(
        [
            vec(mean_shift),   # good
            vec(off_med),      # mediocre +
            vec(-off_med),     # mediocre -
            vec(off_bad),      # bad +
            vec(-off_bad),     # bad -
        ],
        axis=0,
    ).astype(np.float32)
    stds = np.array([std_good, std_med, std_med, std_bad, std_bad], dtype=np.float32)
    weights = np.array(
        [w["good"], w["mediocre_pos"], w["mediocre_neg"], w["bad_pos"], w["bad_neg"]],
        dtype=np.float32,
    )
    weights = weights / weights.sum()
    return means, stds, weights


def build_action_transform(dim: int, anisotropy_ratio: float, rotation_deg: float) -> np.ndarray:
    """Build linear transform used for sampling behavior noise."""
    mat = np.eye(dim, dtype=np.float32)
    if dim < 2:
        return mat

    if anisotropy_ratio != 1.0:
        ratio = max(float(anisotropy_ratio), 1e-6)
        scale = np.eye(dim, dtype=np.float32)
        scale[0, 0] = math.sqrt(ratio)
        scale[1, 1] = 1.0 / math.sqrt(ratio)
        mat = scale @ mat

    if rotation_deg != 0.0:
        th = math.radians(float(rotation_deg))
        c = math.cos(th)
        s = math.sin(th)
        rot = np.eye(dim, dtype=np.float32)
        rot[:2, :2] = np.array([[c, -s], [s, c]], dtype=np.float32)
        mat = rot @ mat

    return mat


def sample_behavior(
    n: int,
    means: np.ndarray,
    stds: np.ndarray,
    weights: np.ndarray,
    rng: np.random.Generator,
    action_transform: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    k = means.shape[0]
    idx = rng.choice(k, size=n, p=weights)
    eps = rng.normal(0.0, 1.0, size=(n, means.shape[1])).astype(np.float32)
    if action_transform is not None:
        eps = eps @ action_transform.T
    a = means[idx] + eps * stds[idx, None]
    return a.astype(np.float32), idx.astype(np.int64)


def true_reward_torch(a: torch.Tensor, reward_center: Optional[torch.Tensor] = None) -> torch.Tensor:
    if reward_center is None:
        diff = a
    else:
        diff = a - reward_center.unsqueeze(0)
    return -diff.pow(2).sum(dim=-1)


def nearest_mode_torch(a: torch.Tensor, means_t: torch.Tensor) -> torch.Tensor:
    d = torch.cdist(a, means_t, p=2)
    return d.argmin(dim=1)


def corrupted_q_torch(
    a: torch.Tensor,
    means_t: torch.Tensor,
    bad_ids_t: torch.Tensor,
    reward_center_t: Optional[torch.Tensor],
    q_noise_std: float,
    q_bias_bad: float,
    bad_spike_prob: float,
    bad_spike_value: float,
) -> torch.Tensor:
    q = true_reward_torch(a, reward_center=reward_center_t)
    nearest = nearest_mode_torch(a, means_t)
    is_bad = torch.isin(nearest, bad_ids_t)

    if q_bias_bad != 0.0:
        q = q + q_bias_bad * is_bad.float()
    if bad_spike_prob > 0.0 and bad_spike_value != 0.0:
        spikes = (torch.rand_like(q) < bad_spike_prob) & is_bad
        q = q + bad_spike_value * spikes.float()
    if q_noise_std > 0.0:
        q = q + q_noise_std * torch.randn_like(q)
    return q


def evaluate_metrics(
    policy: StaticDiagGaussian,
    means: np.ndarray,
    stds: np.ndarray,
    good_mode_ids: List[int],
    bad_mode_ids: List[int],
    reward_center_t: Optional[torch.Tensor],
    n_eval: int,
    recall_radius_mult: float,
) -> Dict[str, float]:
    with torch.no_grad():
        a = policy.rsample(n_eval)
        reward_mean = float(true_reward_torch(a, reward_center=reward_center_t).mean().item())

    means_t = torch.tensor(means, dtype=torch.float32)
    nearest = nearest_mode_torch(a, means_t).cpu().numpy()
    bad_mass = float(np.isin(nearest, np.array(bad_mode_ids)).mean())

    a_np = a.cpu().numpy()
    d_scale = math.sqrt(a_np.shape[1])
    good_hits = []
    for gid in good_mode_ids:
        center = means[gid]
        radius = recall_radius_mult * stds[gid] * d_scale
        hit = np.linalg.norm(a_np - center[None, :], axis=1) <= radius
        good_hits.append(hit.mean())
    good_recall = float(np.mean(good_hits)) if good_hits else float("nan")

    return {
        "mean_reward": reward_mean,
        "bad_mass": bad_mass,
        "good_recall": good_recall,
    }


def train_bc(
    policy: StaticDiagGaussian,
    actions_t: torch.Tensor,
    train_cfg: dict,
    seed: int,
) -> float:
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = torch.optim.Adam(policy.parameters(), lr=float(train_cfg["lr"]))
    n = actions_t.shape[0]
    bsz = int(train_cfg["batch_size"])
    epochs = int(train_cfg["epochs"])
    steps = int(train_cfg["steps_per_epoch"])
    final_loss = 0.0
    for _ in range(epochs):
        for _ in range(steps):
            idx = torch.randint(0, n, (bsz,))
            a = actions_t[idx]
            loss = -policy.log_prob(a).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            final_loss = float(loss.item())
    return final_loss


def train_forward_kl(
    policy: StaticDiagGaussian,
    actions_t: torch.Tensor,
    qhat_t: torch.Tensor,
    train_cfg: dict,
    alpha: float,
    seed: int,
) -> float:
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = torch.optim.Adam(policy.parameters(), lr=float(train_cfg["lr"]))
    n = actions_t.shape[0]
    bsz = int(train_cfg["batch_size"])
    epochs = int(train_cfg["epochs"])
    steps = int(train_cfg["steps_per_epoch"])
    final_loss = 0.0
    for _ in range(epochs):
        for _ in range(steps):
            idx = torch.randint(0, n, (bsz,))
            a = actions_t[idx]
            q = qhat_t[idx]
            lw = (q / alpha).float()
            lw = lw - lw.max()
            w = torch.softmax(lw, dim=0)
            loss = -(w.detach() * policy.log_prob(a)).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            final_loss = float(loss.item())
    return final_loss


def train_reverse_kl(
    policy: StaticDiagGaussian,
    behavior_model: StaticDiagGaussian,
    actions_t: torch.Tensor,
    qhat_t: torch.Tensor,
    train_cfg: dict,
    alpha: float,
    beta: float,
    seed: int,
) -> float:
    """AWR-style weighted MLE + reverse-KL penalty KL(pi||beta)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = torch.optim.Adam(policy.parameters(), lr=float(train_cfg["lr"]))
    n = actions_t.shape[0]
    bsz = int(train_cfg["batch_size"])
    epochs = int(train_cfg["epochs"])
    steps = int(train_cfg["steps_per_epoch"])
    final_loss = 0.0

    for _ in range(epochs):
        for _ in range(steps):
            idx = torch.randint(0, n, (bsz,))
            a = actions_t[idx]
            q = qhat_t[idx]

            adv = q - q.mean()
            lw = (adv / alpha).float()
            lw = lw - lw.max()
            w = torch.softmax(lw, dim=0)
            mle_loss = -(w.detach() * policy.log_prob(a)).sum()

            a_pi = policy.rsample(bsz)
            kl_pen = (policy.log_prob(a_pi) - behavior_model.log_prob(a_pi)).mean()
            loss = mle_loss + beta * kl_pen

            opt.zero_grad()
            loss.backward()
            opt.step()
            final_loss = float(loss.item())
    return final_loss


def ot_plan_cost(
    a_pi: torch.Tensor,
    a_b: torch.Tensor,
    method: str,
    ub_reg: float,
    ub_reg_m: float,
    sinkhorn_reg: float,
    sinkhorn_iter: int,
) -> torch.Tensor:
    n = a_pi.shape[0]
    C = torch.cdist(a_pi, a_b, p=2) ** 2
    C_np = C.detach().cpu().numpy().astype(np.float64)
    uniform = np.ones(n, dtype=np.float64) / n

    if method == "wasserstein":
        cmax = C_np.max()
        Cn = C_np / cmax if cmax > 0 else C_np
        T = pot.sinkhorn(
            uniform,
            uniform,
            Cn,
            reg=sinkhorn_reg,
            numItermax=sinkhorn_iter,
            method="sinkhorn_log",
        )
        T = np.nan_to_num(T, nan=0.0, posinf=0.0, neginf=0.0)
    elif method == "unbalanced_ot":
        cmax = C_np.max()
        Cn = C_np / cmax if cmax > 0 else C_np
        T = pot.unbalanced.sinkhorn_unbalanced(
            uniform,
            uniform,
            Cn,
            reg=ub_reg,
            reg_m=ub_reg_m,
            numItermax=200,
        )
        T = np.nan_to_num(T, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        raise ValueError(f"Unknown OT method for plan cost: {method}")

    T_t = torch.tensor(T, dtype=torch.float32, device=C.device)
    return (T_t * C).sum()


def train_ot(
    policy: StaticDiagGaussian,
    means_t: torch.Tensor,
    bad_ids_t: torch.Tensor,
    reward_center_t: Optional[torch.Tensor],
    means_np: np.ndarray,
    stds_np: np.ndarray,
    weights_np: np.ndarray,
    action_transform_np: Optional[np.ndarray],
    train_cfg: dict,
    corr_cfg: dict,
    ot_cfg: dict,
    method: str,
    lam: float,
    seed: int,
) -> float:
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed + 12345)
    opt = torch.optim.Adam(policy.parameters(), lr=float(train_cfg["lr"]))

    bsz = int(ot_cfg.get("batch_size", train_cfg["batch_size"]))
    epochs = int(train_cfg["epochs"])
    steps = int(train_cfg["steps_per_epoch"])
    lam_q = float(ot_cfg["lam_q"])
    ub_reg = float(ot_cfg["unbalanced_reg"])
    ub_reg_m = float(ot_cfg["unbalanced_reg_m"])
    sinkhorn_reg = float(ot_cfg.get("sinkhorn_reg", 0.08))
    sinkhorn_iter = int(ot_cfg.get("sinkhorn_iter", 100))

    q_noise_std = float(corr_cfg["q_noise_std"])
    q_bias_bad = float(corr_cfg["q_bias_bad"])
    bad_spike_prob = float(corr_cfg["bad_spike_prob"])
    bad_spike_value = float(corr_cfg["bad_spike_value"])

    final_loss = 0.0
    for _ in range(epochs):
        for _ in range(steps):
            a_pi = policy.rsample(bsz)
            q_hat = corrupted_q_torch(
                a_pi,
                means_t,
                bad_ids_t,
                reward_center_t=reward_center_t,
                q_noise_std=q_noise_std,
                q_bias_bad=q_bias_bad,
                bad_spike_prob=bad_spike_prob,
                bad_spike_value=bad_spike_value,
            )
            q_term = -lam_q * q_hat.mean()

            a_b_np, _ = sample_behavior(
                bsz,
                means_np,
                stds_np,
                weights_np,
                rng,
                action_transform=action_transform_np,
            )
            a_b = torch.tensor(a_b_np, dtype=torch.float32)
            ot_cost = ot_plan_cost(
                a_pi,
                a_b,
                method=method,
                ub_reg=ub_reg,
                ub_reg_m=ub_reg_m,
                sinkhorn_reg=sinkhorn_reg,
                sinkhorn_iter=sinkhorn_iter,
            )
            loss = q_term + lam * ot_cost
            opt.zero_grad()
            loss.backward()
            opt.step()
            final_loss = float(loss.item())
    return final_loss


def train_potential_partial(
    policy: StaticDiagGaussian,
    means_t: torch.Tensor,
    bad_ids_t: torch.Tensor,
    reward_center_t: Optional[torch.Tensor],
    means_np: np.ndarray,
    stds_np: np.ndarray,
    weights_np: np.ndarray,
    action_transform_np: Optional[np.ndarray],
    train_cfg: dict,
    corr_cfg: dict,
    ot_cfg: dict,
    partial_cfg: dict,
    lam: float,
    seed: int,
) -> float:
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed + 54321)

    policy_opt = torch.optim.Adam(policy.parameters(), lr=float(train_cfg["lr"]))
    pot = PotentialMLP(
        action_dim=means_np.shape[1],
        hidden_dim=int(partial_cfg.get("pot_hidden_dim", 128)),
        output_nonnegative=bool(partial_cfg.get("pot_output_nonnegative", True)),
    )
    pot_opt = torch.optim.Adam(
        pot.parameters(),
        lr=float(partial_cfg.get("pot_lr", 1e-4)),
    )

    bsz = int(ot_cfg.get("batch_size", train_cfg["batch_size"]))
    epochs = int(train_cfg["epochs"])
    steps = int(train_cfg["steps_per_epoch"])
    lam_q = float(ot_cfg["lam_q"])
    w = float(partial_cfg.get("w", partial_cfg.get("mass_fraction", 1.0)))
    pot_steps = int(partial_cfg.get("pot_steps_per_batch", 1))
    pot_weight_decay = float(partial_cfg.get("pot_weight_decay", 0.0))

    q_noise_std = float(corr_cfg["q_noise_std"])
    q_bias_bad = float(corr_cfg["q_bias_bad"])
    bad_spike_prob = float(corr_cfg["bad_spike_prob"])
    bad_spike_value = float(corr_cfg["bad_spike_value"])

    final_loss = 0.0
    for _ in range(epochs):
        for _ in range(steps):
            for _ in range(pot_steps):
                with torch.no_grad():
                    a_pi_det = policy.rsample(bsz).detach()
                a_b_np, _ = sample_behavior(
                    bsz,
                    means_np,
                    stds_np,
                    weights_np,
                    rng,
                    action_transform=action_transform_np,
                )
                a_b = torch.tensor(a_b_np, dtype=torch.float32)

                l2_reg = torch.tensor(0.0)
                for p in pot.parameters():
                    l2_reg = l2_reg + p.pow(2).sum()
                pot_loss = pot(a_pi_det).mean() - w * pot(a_b).mean() + pot_weight_decay * l2_reg

                pot_opt.zero_grad()
                pot_loss.backward()
                pot_opt.step()

            for p in pot.parameters():
                p.requires_grad_(False)

            a_pi = policy.rsample(bsz)
            q_hat = corrupted_q_torch(
                a_pi,
                means_t,
                bad_ids_t,
                reward_center_t=reward_center_t,
                q_noise_std=q_noise_std,
                q_bias_bad=q_bias_bad,
                bad_spike_prob=bad_spike_prob,
                bad_spike_value=bad_spike_value,
            )
            q_term = -lam_q * q_hat.mean()
            loss = q_term - lam * pot(a_pi).mean()

            policy_opt.zero_grad()
            loss.backward()
            policy_opt.step()
            final_loss = float(loss.item())

            for p in pot.parameters():
                p.requires_grad_(True)

    return final_loss


def train_l2_constraint(
    policy: StaticDiagGaussian,
    means_t: torch.Tensor,
    bad_ids_t: torch.Tensor,
    reward_center_t: Optional[torch.Tensor],
    means_np: np.ndarray,
    stds_np: np.ndarray,
    weights_np: np.ndarray,
    action_transform_np: Optional[np.ndarray],
    train_cfg: dict,
    corr_cfg: dict,
    ot_cfg: dict,
    l2_cfg: dict,
    lam: float,
    seed: int,
) -> float:
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed + 7777)
    opt = torch.optim.Adam(policy.parameters(), lr=float(train_cfg["lr"]))

    bsz = int(ot_cfg.get("batch_size", train_cfg["batch_size"]))
    epochs = int(train_cfg["epochs"])
    steps = int(train_cfg["steps_per_epoch"])
    lam_q = float(ot_cfg["lam_q"])
    anchor_mode = l2_cfg.get("anchor_mode", "dataset_action")
    anchor_noise_std = float(l2_cfg.get("anchor_noise_std", 0.0))
    if anchor_mode != "dataset_action":
        raise ValueError(f"Unsupported l2_constraint.anchor_mode={anchor_mode}")

    q_noise_std = float(corr_cfg["q_noise_std"])
    q_bias_bad = float(corr_cfg["q_bias_bad"])
    bad_spike_prob = float(corr_cfg["bad_spike_prob"])
    bad_spike_value = float(corr_cfg["bad_spike_value"])

    final_loss = 0.0
    for _ in range(epochs):
        for _ in range(steps):
            a_pi = policy.rsample(bsz)
            q_hat = corrupted_q_torch(
                a_pi,
                means_t,
                bad_ids_t,
                reward_center_t=reward_center_t,
                q_noise_std=q_noise_std,
                q_bias_bad=q_bias_bad,
                bad_spike_prob=bad_spike_prob,
                bad_spike_value=bad_spike_value,
            )
            a_b_np, _ = sample_behavior(
                bsz,
                means_np,
                stds_np,
                weights_np,
                rng,
                action_transform=action_transform_np,
            )
            a_b = torch.tensor(a_b_np, dtype=torch.float32)
            if anchor_noise_std > 0.0:
                a_b = a_b + anchor_noise_std * torch.randn_like(a_b)
            l2_pen = (a_pi - a_b).pow(2).sum(dim=-1).mean()
            loss = -lam_q * q_hat.mean() + lam * l2_pen
            opt.zero_grad()
            loss.backward()
            opt.step()
            final_loss = float(loss.item())
    return final_loss


def save_csv(path: str, rows: List[dict], fieldnames: List[str]):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"Saved: {path}")


def aggregate(records: List[dict]) -> List[dict]:
    groups: Dict[Tuple[int, str, float], List[dict]] = {}
    for r in records:
        key = (int(r["dim"]), str(r["method"]), float(r["lambda"]))
        groups.setdefault(key, []).append(r)

    out = []
    for (dim, method, lam), vals in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1], x[0][2])):
        arr_r = np.array([v["mean_reward"] for v in vals], dtype=np.float64)
        arr_b = np.array([v["bad_mass"] for v in vals], dtype=np.float64)
        arr_g = np.array([v["good_recall"] for v in vals], dtype=np.float64)
        out.append(
            {
                "dim": dim,
                "method": method,
                "lambda": lam,
                "n_seeds": len(vals),
                "mean_reward_mean": float(arr_r.mean()),
                "mean_reward_std": float(arr_r.std(ddof=0)),
                "bad_mass_mean": float(arr_b.mean()),
                "bad_mass_std": float(arr_b.std(ddof=0)),
                "good_recall_mean": float(arr_g.mean()),
                "good_recall_std": float(arr_g.std(ddof=0)),
            }
        )
    return out


def plot_reward_curves(summary: List[dict], out_path: str):
    dims = sorted({int(r["dim"]) for r in summary})
    fig, axes = plt.subplots(1, len(dims), figsize=(5 * len(dims), 4), sharey=True)
    if len(dims) == 1:
        axes = [axes]

    curves = [
        ("wasserstein", "tab:blue"),
        ("partial_ot", "tab:green"),
        ("unbalanced_ot", "tab:orange"),
        ("l2_constraint", "tab:red"),
    ]

    for ax, dim in zip(axes, dims):
        rows = [r for r in summary if int(r["dim"]) == dim]
        fkl = next(r for r in rows if r["method"] == "forward_kl")
        ax.axhline(fkl["mean_reward_mean"], linestyle="--", color="black", label="Forward KL")
        rkl_rows = [r for r in rows if r["method"] == "reverse_kl"]
        if rkl_rows:
            ax.axhline(
                rkl_rows[0]["mean_reward_mean"],
                linestyle=":",
                color="#555555",
                label="Reverse KL",
            )

        for method, color in curves:
            rr = sorted([r for r in rows if r["method"] == method], key=lambda x: x["lambda"])
            if not rr:
                continue
            x = [r["lambda"] for r in rr]
            y = [r["mean_reward_mean"] for r in rr]
            ax.plot(x, y, marker="o", color=color, label=method)

        ax.set_xscale("log")
        ax.set_xlabel(f"lambda (dim={dim})")
        ax.grid(True, alpha=0.3)

    axes[0].set_ylabel("Mean reward (higher better)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def build_winner_table(summary: List[dict]) -> List[dict]:
    dims = sorted({int(r["dim"]) for r in summary})
    rows = []
    for dim in dims:
        s = [r for r in summary if int(r["dim"]) == dim]
        fkl = next(r for r in s if r["method"] == "forward_kl")
        fkl_r = fkl["mean_reward_mean"]
        rkl_rows = [r for r in s if r["method"] == "reverse_kl"]
        rkl_r = rkl_rows[0]["mean_reward_mean"] if rkl_rows else float("nan")

        best = {}
        for m in ["wasserstein", "partial_ot", "unbalanced_ot", "l2_constraint"]:
            cand = [r for r in s if r["method"] == m]
            best[m] = max(cand, key=lambda x: x["mean_reward_mean"])
        best_ot_reward = max(
            best["wasserstein"]["mean_reward_mean"],
            best["partial_ot"]["mean_reward_mean"],
            best["unbalanced_ot"]["mean_reward_mean"],
        )

        rows.append(
            {
                "dim": dim,
                "forward_kl_mean_reward": fkl_r,
                "reverse_kl_mean_reward": rkl_r,
                "best_wasserstein_lambda": best["wasserstein"]["lambda"],
                "best_wasserstein_reward": best["wasserstein"]["mean_reward_mean"],
                "best_partial_lambda": best["partial_ot"]["lambda"],
                "best_partial_reward": best["partial_ot"]["mean_reward_mean"],
                "best_unbalanced_lambda": best["unbalanced_ot"]["lambda"],
                "best_unbalanced_reward": best["unbalanced_ot"]["mean_reward_mean"],
                "best_partial_minus_kl": best["partial_ot"]["mean_reward_mean"] - fkl_r,
                "best_l2_lambda": best["l2_constraint"]["lambda"],
                "best_l2_reward": best["l2_constraint"]["mean_reward_mean"],
                "best_l2_minus_kl": best["l2_constraint"]["mean_reward_mean"] - fkl_r,
                "best_ot_minus_reverse_kl": (
                    best_ot_reward - rkl_r if not math.isnan(rkl_r) else float("nan")
                ),
                "best_l2_minus_reverse_kl": (
                    best["l2_constraint"]["mean_reward_mean"] - rkl_r
                    if not math.isnan(rkl_r)
                    else float("nan")
                ),
                "all_ot_beat_kl": int(
                    (best["wasserstein"]["mean_reward_mean"] > fkl_r)
                    and (best["partial_ot"]["mean_reward_mean"] > fkl_r)
                    and (best["unbalanced_ot"]["mean_reward_mean"] > fkl_r)
                ),
            }
        )
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ot_wins_dim_sweep.yaml")
    parser.add_argument("--dims", type=str, default="")
    parser.add_argument("--seeds", type=str, default="")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--steps-per-epoch", type=int, default=0)
    parser.add_argument("--ot-batch-size", type=int, default=0)
    parser.add_argument("--n-data", type=int, default=0)
    parser.add_argument("--q-noise-std", type=float, default=-1.0)
    parser.add_argument("--output-prefix", type=str, default="ot_wins_dim_sweep")
    args = parser.parse_args()

    cfg = load_config(args.config)
    env_cfg = copy.deepcopy(cfg["env"])
    corr_cfg = copy.deepcopy(cfg["corruption"])
    train_cfg = copy.deepcopy(cfg["training"])
    kl_cfg = copy.deepcopy(cfg["kl"])
    ot_cfg = copy.deepcopy(cfg["ot"])
    partial_cfg = copy.deepcopy(cfg.get("ot_partial", {}))
    l2_cfg = copy.deepcopy(cfg.get("l2_constraint", {}))
    kl_reverse_cfg = copy.deepcopy(cfg.get("kl_reverse", {}))
    eval_cfg = copy.deepcopy(cfg["eval"])
    exp_cfg = copy.deepcopy(cfg["experiment"])

    if args.dims.strip():
        dims = parse_int_list(args.dims)
    else:
        dims = list(env_cfg["action_dims"])
    if args.seeds.strip():
        seeds = parse_int_list(args.seeds)
    else:
        seeds = list(exp_cfg["seeds"])
    if args.epochs > 0:
        train_cfg["epochs"] = args.epochs
    if args.batch_size > 0:
        train_cfg["batch_size"] = args.batch_size
    if args.steps_per_epoch > 0:
        train_cfg["steps_per_epoch"] = args.steps_per_epoch
    if args.ot_batch_size > 0:
        ot_cfg["batch_size"] = args.ot_batch_size
    if args.n_data > 0:
        env_cfg["n_data"] = args.n_data
    if args.q_noise_std >= 0.0:
        corr_cfg["q_noise_std"] = args.q_noise_std
    mean_shift = float(env_cfg.get("mean_shift", 0.0))
    reward_center_shift = float(env_cfg.get("reward_center_shift", 0.0))
    anisotropy_ratio = float(env_cfg.get("anisotropy_ratio", 1.0))
    rotation_deg = float(env_cfg.get("rotation_deg", 0.0))
    anchor_noise_std = float(l2_cfg.get("anchor_noise_std", 0.0))

    os.makedirs("results", exist_ok=True)

    print("=" * 88)
    print("OT-WINS DIMENSION SWEEP TOY (POT replacement + L2 baseline)")
    print(f"dims={dims} seeds={seeds}")
    print(
        f"N={env_cfg['n_data']} epochs={train_cfg['epochs']} batch={train_cfg['batch_size']} "
        f"steps/epoch={train_cfg['steps_per_epoch']}"
    )
    print(f"q_noise={corr_cfg['q_noise_std']} q_bias_bad={corr_cfg['q_bias_bad']} spike={corr_cfg['bad_spike_prob']}")
    print(
        f"ot_batch={ot_cfg.get('batch_size', train_cfg['batch_size'])} "
        f"sinkhorn_reg={ot_cfg.get('sinkhorn_reg', 0.08)} "
        f"sinkhorn_iter={ot_cfg.get('sinkhorn_iter', 100)}"
    )
    print(
        f"partial_ot.w={partial_cfg.get('w', partial_cfg.get('mass_fraction', 1.0))} "
        f"pot_hidden={partial_cfg.get('pot_hidden_dim', 128)} "
        f"pot_steps={partial_cfg.get('pot_steps_per_batch', 1)}"
    )
    print(
        f"scenario(mean_shift={mean_shift}, reward_center_shift={reward_center_shift}, "
        f"anisotropy_ratio={anisotropy_ratio}, rotation_deg={rotation_deg}, "
        f"l2_anchor_noise_std={anchor_noise_std})"
    )
    print("=" * 88)

    raw: List[dict] = []
    lambdas = list(ot_cfg["lambdas"])
    partial_lambdas = list(partial_cfg.get("lambdas", lambdas))
    l2_lambdas = list(l2_cfg.get("lambdas", lambdas))
    inprogress_raw_csv = f"results/{args.output_prefix}_raw_inprogress.csv"

    for dim in dims:
        print("\n" + "#" * 88)
        print(f"Dimension {dim}")
        print("#" * 88)

        means, stds, weights = build_modes_for_dim(env_cfg, dim)
        action_transform = build_action_transform(dim, anisotropy_ratio, rotation_deg)
        reward_center = np.zeros(dim, dtype=np.float32)
        reward_center[0] = reward_center_shift
        good_mode_ids = list(env_cfg["good_mode_ids"])
        bad_mode_ids = list(env_cfg["bad_mode_ids"])

        means_t = torch.tensor(means, dtype=torch.float32)
        bad_ids_t = torch.tensor(bad_mode_ids, dtype=torch.long)
        reward_center_t = torch.tensor(reward_center, dtype=torch.float32)

        for seed in seeds:
            print(f"\n[dim={dim} seed={seed}]")
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
            bc_loss = train_bc(bc, actions_t, train_cfg, seed=seed * 100 + 1)
            m = evaluate_metrics(
                bc,
                means,
                stds,
                good_mode_ids,
                bad_mode_ids,
                reward_center_t=reward_center_t,
                n_eval=int(eval_cfg["n_eval_samples"]),
                recall_radius_mult=float(eval_cfg["recall_radius_mult"]),
            )
            raw.append(
                {
                    "dim": dim,
                    "seed": seed,
                    "method": "bc",
                    "lambda": 1.0,
                    "mean_reward": m["mean_reward"],
                    "bad_mass": m["bad_mass"],
                    "good_recall": m["good_recall"],
                    "final_loss": bc_loss,
                }
            )
            print(
                f"  BC            reward={m['mean_reward']:8.4f} bad={m['bad_mass']:.3f} "
                f"recall={m['good_recall']:.3f}"
            )

            fkl = StaticDiagGaussian(dim)
            fkl_loss = train_forward_kl(
                fkl,
                actions_t,
                qhat_t,
                train_cfg=train_cfg,
                alpha=float(kl_cfg["alpha"]),
                seed=seed * 100 + 2,
            )
            m = evaluate_metrics(
                fkl,
                means,
                stds,
                good_mode_ids,
                bad_mode_ids,
                reward_center_t=reward_center_t,
                n_eval=int(eval_cfg["n_eval_samples"]),
                recall_radius_mult=float(eval_cfg["recall_radius_mult"]),
            )
            raw.append(
                {
                    "dim": dim,
                    "seed": seed,
                    "method": "forward_kl",
                    "lambda": 1.0,
                    "mean_reward": m["mean_reward"],
                    "bad_mass": m["bad_mass"],
                    "good_recall": m["good_recall"],
                    "final_loss": fkl_loss,
                }
            )
            print(
                f"  Forward KL    reward={m['mean_reward']:8.4f} bad={m['bad_mass']:.3f} "
                f"recall={m['good_recall']:.3f}"
            )

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
            m = evaluate_metrics(
                rkl,
                means,
                stds,
                good_mode_ids,
                bad_mode_ids,
                reward_center_t=reward_center_t,
                n_eval=int(eval_cfg["n_eval_samples"]),
                recall_radius_mult=float(eval_cfg["recall_radius_mult"]),
            )
            raw.append(
                {
                    "dim": dim,
                    "seed": seed,
                    "method": "reverse_kl",
                    "lambda": 1.0,
                    "mean_reward": m["mean_reward"],
                    "bad_mass": m["bad_mass"],
                    "good_recall": m["good_recall"],
                    "final_loss": rkl_loss,
                }
            )
            print(
                f"  Reverse KL    reward={m['mean_reward']:8.4f} bad={m['bad_mass']:.3f} "
                f"recall={m['good_recall']:.3f}"
            )

            for lam in lambdas:
                for method in ["wasserstein", "unbalanced_ot"]:
                    p = StaticDiagGaussian(dim)
                    loss = train_ot(
                        p,
                        means_t=means_t,
                        bad_ids_t=bad_ids_t,
                        reward_center_t=reward_center_t,
                        means_np=means,
                        stds_np=stds,
                        weights_np=weights,
                        action_transform_np=action_transform,
                        train_cfg=train_cfg,
                        corr_cfg=corr_cfg,
                        ot_cfg=ot_cfg,
                        method=method,
                        lam=float(lam),
                        seed=seed * 1000 + int(round(lam * 1000)) + (1 if method == "wasserstein" else 3),
                    )
                    m = evaluate_metrics(
                        p,
                        means,
                        stds,
                        good_mode_ids,
                        bad_mode_ids,
                        reward_center_t=reward_center_t,
                        n_eval=int(eval_cfg["n_eval_samples"]),
                        recall_radius_mult=float(eval_cfg["recall_radius_mult"]),
                    )
                    raw.append(
                        {
                            "dim": dim,
                            "seed": seed,
                            "method": method,
                            "lambda": float(lam),
                            "mean_reward": m["mean_reward"],
                            "bad_mass": m["bad_mass"],
                            "good_recall": m["good_recall"],
                            "final_loss": loss,
                        }
                    )
                    print(
                        f"  {method:<13s} lam={lam:>5g} reward={m['mean_reward']:8.4f} "
                        f"bad={m['bad_mass']:.3f} recall={m['good_recall']:.3f}"
                    )

            for lam in partial_lambdas:
                p = StaticDiagGaussian(dim)
                loss = train_potential_partial(
                    p,
                    means_t=means_t,
                    bad_ids_t=bad_ids_t,
                    reward_center_t=reward_center_t,
                    means_np=means,
                    stds_np=stds,
                    weights_np=weights,
                    action_transform_np=action_transform,
                    train_cfg=train_cfg,
                    corr_cfg=corr_cfg,
                    ot_cfg=ot_cfg,
                    partial_cfg=partial_cfg,
                    lam=float(lam),
                    seed=seed * 1000 + int(round(lam * 1000)) + 2,
                )
                m = evaluate_metrics(
                    p,
                    means,
                    stds,
                    good_mode_ids,
                    bad_mode_ids,
                    reward_center_t=reward_center_t,
                    n_eval=int(eval_cfg["n_eval_samples"]),
                    recall_radius_mult=float(eval_cfg["recall_radius_mult"]),
                )
                raw.append(
                    {
                        "dim": dim,
                        "seed": seed,
                        "method": "partial_ot",
                        "lambda": float(lam),
                        "mean_reward": m["mean_reward"],
                        "bad_mass": m["bad_mass"],
                        "good_recall": m["good_recall"],
                        "final_loss": loss,
                    }
                )
                print(
                    f"  {'partial_ot':<13s} lam={lam:>5g} reward={m['mean_reward']:8.4f} "
                    f"bad={m['bad_mass']:.3f} recall={m['good_recall']:.3f}"
                )

            for lam in l2_lambdas:
                p = StaticDiagGaussian(dim)
                loss = train_l2_constraint(
                    p,
                    means_t=means_t,
                    bad_ids_t=bad_ids_t,
                    reward_center_t=reward_center_t,
                    means_np=means,
                    stds_np=stds,
                    weights_np=weights,
                    action_transform_np=action_transform,
                    train_cfg=train_cfg,
                    corr_cfg=corr_cfg,
                    ot_cfg=ot_cfg,
                    l2_cfg=l2_cfg,
                    lam=float(lam),
                    seed=seed * 1000 + int(round(lam * 1000)) + 4,
                )
                m = evaluate_metrics(
                    p,
                    means,
                    stds,
                    good_mode_ids,
                    bad_mode_ids,
                    reward_center_t=reward_center_t,
                    n_eval=int(eval_cfg["n_eval_samples"]),
                    recall_radius_mult=float(eval_cfg["recall_radius_mult"]),
                )
                raw.append(
                    {
                        "dim": dim,
                        "seed": seed,
                        "method": "l2_constraint",
                        "lambda": float(lam),
                        "mean_reward": m["mean_reward"],
                        "bad_mass": m["bad_mass"],
                        "good_recall": m["good_recall"],
                        "final_loss": loss,
                    }
                )
                print(
                    f"  {'l2_constraint':<13s} lam={lam:>5g} reward={m['mean_reward']:8.4f} "
                    f"bad={m['bad_mass']:.3f} recall={m['good_recall']:.3f}"
                )

            save_csv(
                inprogress_raw_csv,
                raw,
                fieldnames=["dim", "seed", "method", "lambda", "mean_reward", "bad_mass", "good_recall", "final_loss"],
            )

    summary = aggregate(raw)
    winners = build_winner_table(summary)

    prefix = args.output_prefix
    raw_csv = f"results/{prefix}_raw.csv"
    summary_csv = f"results/{prefix}_summary.csv"
    winner_csv = f"results/{prefix}_winner_table.csv"
    reward_png = f"results/{prefix}_reward_curves.png"

    save_csv(
        raw_csv,
        raw,
        fieldnames=["dim", "seed", "method", "lambda", "mean_reward", "bad_mass", "good_recall", "final_loss"],
    )
    save_csv(
        summary_csv,
        summary,
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
    save_csv(
        winner_csv,
        winners,
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
    plot_reward_curves(summary, reward_png)

    print("\n" + "=" * 88)
    print("Winner Table")
    print("=" * 88)
    for r in winners:
        print(
            f"dim={r['dim']} KL={r['forward_kl_mean_reward']:.4f} | "
            f"RKL={r['reverse_kl_mean_reward']:.4f} | "
            f"W={r['best_wasserstein_reward']:.4f}@{r['best_wasserstein_lambda']} "
            f"P={r['best_partial_reward']:.4f}@{r['best_partial_lambda']} "
            f"U={r['best_unbalanced_reward']:.4f}@{r['best_unbalanced_lambda']} "
            f"L2={r['best_l2_reward']:.4f}@{r['best_l2_lambda']} "
            f"all_ot_beat_kl={r['all_ot_beat_kl']}"
        )

    print("\nDone.")
    print(f"Raw: {raw_csv}")
    print(f"Summary: {summary_csv}")
    print(f"Winners: {winner_csv}")
    print(f"Plot: {reward_png}")
    if os.path.exists(inprogress_raw_csv):
        os.remove(inprogress_raw_csv)


if __name__ == "__main__":
    main()
