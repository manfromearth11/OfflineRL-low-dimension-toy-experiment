# OfflineRL Toy Reboot: 4-Axis Results (Balanced)

This document is the clean, English-only report for the completed 4-axis run on **March 4, 2026**.

- Main runner: `scripts/run_experiment.py`
- Result prefix: `regime_4axis_scenario_balanced`
- Main aggregate file: `results/regime_4axis_scenario_balanced_grid.csv`

## 1) Experiment Axes (Explicit)

1. `scenario`: `baseline`, `good_shift`, `rotated`, `anchor_corrupt`
2. `modality`: `low`, `mid`, `high`
3. `q_quality`: `clean`, `mid`, `noisy`
4. `action_dim`: `2`, `4`, `8`

Total cells:
- `4 x 3 x 3 x 3 = 108`

## 2) Scenario Definitions (Moderate Profile)

| Scenario | Parameters | Intended stress |
|---|---|---|
| `baseline` | `mean_shift=0.0`, `reward_center_shift=0.0`, `anisotropy_ratio=1.0`, `rotation_deg=0.0`, `l2_anchor_noise_std=0.0` | Reference regime with no geometric distortion and no L2 anchor corruption. |
| `good_shift` | `mean_shift=+1.0`, others same as baseline | Shifts the good mode along action axis 1. Tests sensitivity to shifted behavior geometry. |
| `rotated` | `anisotropy_ratio=3.0`, `rotation_deg=35.0`, others baseline | Applies anisotropic + rotated geometry (first two action dimensions). |
| `anchor_corrupt` | `l2_anchor_noise_std=0.35`, others baseline | Adds noise only to L2 anchor actions (`a_beta + sigma * noise`). |

## 3) 2D Scenario Sketch (No In-Image Titles)

### 2D Overview (`modality=mid`, `scenario_level=moderate`)

![2D scenario overview](results/scenario_2d_overview_moderate.png)

Panel order:
- Top-left: `baseline`
- Top-right: `good_shift`
- Bottom-left: `rotated`
- Bottom-right: `anchor_corrupt`

## 4) Modality and Q-Quality Profiles

### Modality profiles

| Modality | Offsets `(mediocre,bad)` | Stds `(good,mediocre,bad)` | Weights `(good, med+, med-, bad+, bad-)` |
|---|---|---|---|
| `low` | `(0.4, 2.5)` | `(0.12, 0.50, 0.50)` | `(0.45, 0.225, 0.225, 0.05, 0.05)` |
| `mid` | `(0.8, 5.0)` | `(0.10, 0.40, 0.20)` | `(0.20, 0.15, 0.15, 0.25, 0.25)` |
| `high` | `(1.0, 6.0)` | `(0.08, 0.35, 0.15)` | `(0.10, 0.10, 0.10, 0.35, 0.35)` |

### Q-quality profiles

| Q quality | `q_noise_std` | `q_bias_bad` | `bad_spike_prob` | `bad_spike_value` |
|---|---:|---:|---:|---:|
| `clean` | 0.0 | 0.0 | 0.0 | 0.0 |
| `mid` | 2.0 | 0.3 | 0.0 | 0.0 |
| `noisy` | 5.0 | 1.0 | 0.01 | 10.0 |

## 5) Methods Compared

- `bc`
- `forward_kl`
- `reverse_kl`
- `wasserstein`
- `partial_ot` (potential-game replacement, PPL-style)
- `unbalanced_ot`
- `l2_constraint`

Policy family in this toy:
- State-free static diagonal Gaussian policy.

Note:
- `reverse_kl` is now wired into the 4-axis runner and reward-curve plots.
- In reward-curve plots, the dashed baseline label is per-dimension:
- `KL best(f)` if forward KL is higher.
- `KL best(r)` if reverse KL is higher.
- The numeric tables in this report are from the previously completed run (forward-KL-referenced grid).
- To refresh all numbers with reverse-KL columns included everywhere, rerun `full4`.

## 6) Budget and Outputs

Budget used:
- Seeds: `0,1,2`
- Epochs: `30`
- Dataset size: `15000`
- Dimensions: `2,4,8`

Main outputs:
- Grid CSV: `results/regime_4axis_scenario_balanced_grid.csv`
- Per-cell files: `*_raw.csv`, `*_summary.csv`, `*_winner_table.csv`, `*_reward_curves.png`
- Scenario-modality heatmaps: `*_heatmap_<scenario>_<modality>_pot.png`, `*_heatmap_<scenario>_<modality>_l2.png`

## 7) Quantitative Results (108 Cells)

Notation used in this section:
- `KL best(f)` = `forward_kl_mean_reward`
- `KL best(r)` = `reverse_kl_mean_reward`

### Overall

| Metric | Value |
|---|---:|
| Mean `best_ot_minus_kl` | `+0.1548` |
| Mean `best_l2_minus_kl` | `+0.1553` |
| Mean `best_pot_minus_kl` | `+0.0606` |
| Cells where `UOT > KL` | `68 / 108` |
| Cells where `L2 > KL` | `68 / 108` |
| Cells where `L2 > UOT` | `68 / 108` |
| Cells where `all_ot_beat_kl = 1` | `59 / 108` |
| Best constrained winner count: `l2_constraint` | `68` |
| Best constrained winner count: `unbalanced_ot` | `40` |
| Best constrained winner count: `wasserstein`, `partial_ot` | `0`, `0` |

### By scenario (27 cells each)

| Scenario | `UOT > KL` | `L2 > KL` | Mean `best_ot - KL` | Mean `best_l2 - KL` |
|---|---:|---:|---:|---:|
| `baseline` | 14 | 14 | `+0.0361` | `+0.0365` |
| `good_shift` | 27 | 27 | `+0.4951` | `+0.4958` |
| `rotated` | 13 | 13 | `+0.0405` | `+0.0410` |
| `anchor_corrupt` | 14 | 14 | `+0.0474` | `+0.0480` |

### By Q quality (36 cells each)

| Q quality | `UOT > KL` | `L2 > KL` | Mean `best_ot - KL` | Mean `best_l2 - KL` |
|---|---:|---:|---:|---:|
| `clean` | 9 | 9 | `-0.0776` | `-0.0779` |
| `mid` | 23 | 23 | `+0.1490` | `+0.1489` |
| `noisy` | 36 | 36 | `+0.3929` | `+0.3950` |

### By action dimension (36 cells each)

| Dimension | `UOT > KL` | `L2 > KL` | Mean `best_ot - KL` | Mean `best_l2 - KL` |
|---|---:|---:|---:|---:|
| `2` | 27 | 27 | `+0.1963` | `+0.1965` |
| `4` | 23 | 23 | `+0.1694` | `+0.1703` |
| `8` | 18 | 18 | `+0.0985` | `+0.0991` |

### UOT vs L2 gap

| Metric | Value |
|---|---:|
| Mean `(best_l2_reward - best_uot_reward)` | `+0.00058` |
| Cells with `L2 > UOT` | `68` |
| Cells with `UOT > L2` | `40` |
| Cells with `|L2-UOT| < 0.005` | `108 / 108` |

### UOT vs L2 head-to-head by Q quality

| Q quality | `L2 > UOT` | `UOT > L2` | Mean `(L2 - UOT)` |
|---|---:|---:|---:|
| `clean` | `19` | `17` | `-0.00032` |
| `mid` | `15` | `21` | `-0.00009` |
| `noisy` | `34` | `2` | `+0.00216` |

### UOT vs L2 head-to-head by scenario

| Scenario | `L2 > UOT` | `UOT > L2` | Mean `(L2 - UOT)` |
|---|---:|---:|---:|
| `baseline` | `16` | `11` | `+0.00044` |
| `good_shift` | `19` | `8` | `+0.00085` |
| `rotated` | `16` | `11` | `+0.00047` |
| `anchor_corrupt` | `17` | `10` | `+0.00056` |

## 8) Analysis

1. KL is generally stronger in `clean` Q regimes.
- Average constrained gain is negative in `clean` (`best_ot_minus_kl < 0`).
- This is consistent with KL benefiting from reliable Q ranking.

2. Constrained methods dominate as Q corruption increases.
- In `noisy`, `UOT > KL` in `36/36` cells.
- The same pattern appears for L2 versus KL.

3. `baseline` and `rotated` are close under this moderate scenario profile.
- Both produce small positive average gains (`~+0.04`) versus KL.
- Rotation/anisotropy here does not overturn the Q-quality effect.

4. `good_shift` is the strongest constrained win regime.
- `UOT > KL` and `L2 > KL` are both `27/27`.
- Mean gains are around `+0.50`.

5. L2 and UOT are practically tied in this setup.
- Differences are very small across all cells.
- Strong claims about strict L2 or strict UOT superiority are not supported here.

## 9) Conclusions

- This run supports a **regime-dependent** view, not absolute method ranking.
- KL is favored when Q is accurate (`clean`).
- UOT/L2-style constraints are favored when Q is unreliable (`mid`, `noisy`).
- In this toy, UOT and L2 are near-equivalent at the tuned-best level.

## 10) Trust Scope and Limits

Reliable conclusions:
- The direction of the Q-quality trend (`clean -> mid -> noisy`) is stable.
- KL vulnerability under strong Q corruption is consistently observed.

Moderate-confidence conclusions:
- Fine-grained UOT vs L2 ranking (very small margins).
- Absolute magnitude of dimension effects (`2`, `4`, `8`).

Hard limits:
1. Single-state toy bandit, not full MDP credit assignment.
2. Static diagonal Gaussian policy class (not state-dependent policy learning).
3. Oracle-style lambda tuning per cell (best-of-sweep comparison).
4. Only 3 seeds.

Uncertainty proxy:
- Mean seed-std at best points: `KL≈0.0268`, `UOT≈0.0024`, `L2≈0.0021`
- Improvement above RSS std threshold:
  - `UOT > KL`: `67/108`
  - `L2 > KL`: `67/108`
- By Q quality (`UOT > KL`, same counts for L2):
  - `clean`: `9/36`
  - `mid`: `22/36`
  - `noisy`: `36/36`

## 11) Full Heatmap Gallery (Markdown Titles, No In-Image Titles)

### Scenario: `baseline`

#### `baseline / low / POT - KL`
![baseline low pot](results/regime_4axis_scenario_balanced_heatmap_baseline_low_pot.png)

#### `baseline / low / L2 - KL`
![baseline low l2](results/regime_4axis_scenario_balanced_heatmap_baseline_low_l2.png)

#### `baseline / mid / POT - KL`
![baseline mid pot](results/regime_4axis_scenario_balanced_heatmap_baseline_mid_pot.png)

#### `baseline / mid / L2 - KL`
![baseline mid l2](results/regime_4axis_scenario_balanced_heatmap_baseline_mid_l2.png)

#### `baseline / high / POT - KL`
![baseline high pot](results/regime_4axis_scenario_balanced_heatmap_baseline_high_pot.png)

#### `baseline / high / L2 - KL`
![baseline high l2](results/regime_4axis_scenario_balanced_heatmap_baseline_high_l2.png)

### Scenario: `good_shift`

#### `good_shift / low / POT - KL`
![good_shift low pot](results/regime_4axis_scenario_balanced_heatmap_good_shift_low_pot.png)

#### `good_shift / low / L2 - KL`
![good_shift low l2](results/regime_4axis_scenario_balanced_heatmap_good_shift_low_l2.png)

#### `good_shift / mid / POT - KL`
![good_shift mid pot](results/regime_4axis_scenario_balanced_heatmap_good_shift_mid_pot.png)

#### `good_shift / mid / L2 - KL`
![good_shift mid l2](results/regime_4axis_scenario_balanced_heatmap_good_shift_mid_l2.png)

#### `good_shift / high / POT - KL`
![good_shift high pot](results/regime_4axis_scenario_balanced_heatmap_good_shift_high_pot.png)

#### `good_shift / high / L2 - KL`
![good_shift high l2](results/regime_4axis_scenario_balanced_heatmap_good_shift_high_l2.png)

### Scenario: `rotated`

#### `rotated / low / POT - KL`
![rotated low pot](results/regime_4axis_scenario_balanced_heatmap_rotated_low_pot.png)

#### `rotated / low / L2 - KL`
![rotated low l2](results/regime_4axis_scenario_balanced_heatmap_rotated_low_l2.png)

#### `rotated / mid / POT - KL`
![rotated mid pot](results/regime_4axis_scenario_balanced_heatmap_rotated_mid_pot.png)

#### `rotated / mid / L2 - KL`
![rotated mid l2](results/regime_4axis_scenario_balanced_heatmap_rotated_mid_l2.png)

#### `rotated / high / POT - KL`
![rotated high pot](results/regime_4axis_scenario_balanced_heatmap_rotated_high_pot.png)

#### `rotated / high / L2 - KL`
![rotated high l2](results/regime_4axis_scenario_balanced_heatmap_rotated_high_l2.png)

### Scenario: `anchor_corrupt`

#### `anchor_corrupt / low / POT - KL`
![anchor_corrupt low pot](results/regime_4axis_scenario_balanced_heatmap_anchor_corrupt_low_pot.png)

#### `anchor_corrupt / low / L2 - KL`
![anchor_corrupt low l2](results/regime_4axis_scenario_balanced_heatmap_anchor_corrupt_low_l2.png)

#### `anchor_corrupt / mid / POT - KL`
![anchor_corrupt mid pot](results/regime_4axis_scenario_balanced_heatmap_anchor_corrupt_mid_pot.png)

#### `anchor_corrupt / mid / L2 - KL`
![anchor_corrupt mid l2](results/regime_4axis_scenario_balanced_heatmap_anchor_corrupt_mid_l2.png)

#### `anchor_corrupt / high / POT - KL`
![anchor_corrupt high pot](results/regime_4axis_scenario_balanced_heatmap_anchor_corrupt_high_pot.png)

#### `anchor_corrupt / high / L2 - KL`
![anchor_corrupt high l2](results/regime_4axis_scenario_balanced_heatmap_anchor_corrupt_high_l2.png)

## 12) Reward Curve Index (All 36 Cells)

All reward-curve plots also have no in-image titles.

### `baseline`
- `low / clean`: [plot](results/regime_4axis_scenario_balanced_baseline_low_clean_reward_curves.png)
- `low / mid`: [plot](results/regime_4axis_scenario_balanced_baseline_low_mid_reward_curves.png)
- `low / noisy`: [plot](results/regime_4axis_scenario_balanced_baseline_low_noisy_reward_curves.png)
- `mid / clean`: [plot](results/regime_4axis_scenario_balanced_baseline_mid_clean_reward_curves.png)
- `mid / mid`: [plot](results/regime_4axis_scenario_balanced_baseline_mid_mid_reward_curves.png)
- `mid / noisy`: [plot](results/regime_4axis_scenario_balanced_baseline_mid_noisy_reward_curves.png)
- `high / clean`: [plot](results/regime_4axis_scenario_balanced_baseline_high_clean_reward_curves.png)
- `high / mid`: [plot](results/regime_4axis_scenario_balanced_baseline_high_mid_reward_curves.png)
- `high / noisy`: [plot](results/regime_4axis_scenario_balanced_baseline_high_noisy_reward_curves.png)

### `good_shift`
- `low / clean`: [plot](results/regime_4axis_scenario_balanced_good_shift_low_clean_reward_curves.png)
- `low / mid`: [plot](results/regime_4axis_scenario_balanced_good_shift_low_mid_reward_curves.png)
- `low / noisy`: [plot](results/regime_4axis_scenario_balanced_good_shift_low_noisy_reward_curves.png)
- `mid / clean`: [plot](results/regime_4axis_scenario_balanced_good_shift_mid_clean_reward_curves.png)
- `mid / mid`: [plot](results/regime_4axis_scenario_balanced_good_shift_mid_mid_reward_curves.png)
- `mid / noisy`: [plot](results/regime_4axis_scenario_balanced_good_shift_mid_noisy_reward_curves.png)
- `high / clean`: [plot](results/regime_4axis_scenario_balanced_good_shift_high_clean_reward_curves.png)
- `high / mid`: [plot](results/regime_4axis_scenario_balanced_good_shift_high_mid_reward_curves.png)
- `high / noisy`: [plot](results/regime_4axis_scenario_balanced_good_shift_high_noisy_reward_curves.png)

### `rotated`
- `low / clean`: [plot](results/regime_4axis_scenario_balanced_rotated_low_clean_reward_curves.png)
- `low / mid`: [plot](results/regime_4axis_scenario_balanced_rotated_low_mid_reward_curves.png)
- `low / noisy`: [plot](results/regime_4axis_scenario_balanced_rotated_low_noisy_reward_curves.png)
- `mid / clean`: [plot](results/regime_4axis_scenario_balanced_rotated_mid_clean_reward_curves.png)
- `mid / mid`: [plot](results/regime_4axis_scenario_balanced_rotated_mid_mid_reward_curves.png)
- `mid / noisy`: [plot](results/regime_4axis_scenario_balanced_rotated_mid_noisy_reward_curves.png)
- `high / clean`: [plot](results/regime_4axis_scenario_balanced_rotated_high_clean_reward_curves.png)
- `high / mid`: [plot](results/regime_4axis_scenario_balanced_rotated_high_mid_reward_curves.png)
- `high / noisy`: [plot](results/regime_4axis_scenario_balanced_rotated_high_noisy_reward_curves.png)

### `anchor_corrupt`
- `low / clean`: [plot](results/regime_4axis_scenario_balanced_anchor_corrupt_low_clean_reward_curves.png)
- `low / mid`: [plot](results/regime_4axis_scenario_balanced_anchor_corrupt_low_mid_reward_curves.png)
- `low / noisy`: [plot](results/regime_4axis_scenario_balanced_anchor_corrupt_low_noisy_reward_curves.png)
- `mid / clean`: [plot](results/regime_4axis_scenario_balanced_anchor_corrupt_mid_clean_reward_curves.png)
- `mid / mid`: [plot](results/regime_4axis_scenario_balanced_anchor_corrupt_mid_mid_reward_curves.png)
- `mid / noisy`: [plot](results/regime_4axis_scenario_balanced_anchor_corrupt_mid_noisy_reward_curves.png)
- `high / clean`: [plot](results/regime_4axis_scenario_balanced_anchor_corrupt_high_clean_reward_curves.png)
- `high / mid`: [plot](results/regime_4axis_scenario_balanced_anchor_corrupt_high_mid_reward_curves.png)
- `high / noisy`: [plot](results/regime_4axis_scenario_balanced_anchor_corrupt_high_noisy_reward_curves.png)

## 13) Reproduction Command

```bash
python3 scripts/run_experiment.py --mode full4 \
  --base-config configs/ot_wins_dim_sweep.yaml \
  --dims 2,4,8 \
  --seeds 0,1,2 \
  --epochs 30 \
  --n-data 15000 \
  --scenarios baseline,good_shift,rotated,anchor_corrupt \
  --output-prefix regime_4axis_scenario_balanced
```
