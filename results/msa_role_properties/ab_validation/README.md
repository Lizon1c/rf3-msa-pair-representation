# ab_validation — archive manifest

A/B validation experiments addressing the collaborator's two methodological
concerns: (1) best-epoch selection on a single val set may overfit that set;
(2) standardizing labels on all samples *before* splitting is preprocessing
leakage. Plus the two mechanistic follow-ups (ESM-noise ladder, f_enc LR pace)
on the x_msa_faesm fusion cell.

All numbers are cross-position split-paired Spearman ρ. RBD full set N=3998
(60/20/20 train/val1/val2 for V1; 70/30 for V2), 5 splits (seeds 100–104) ×
3 inits (7/107/207) = 15 runs/cell unless noted. Raw per-run records (incl.
full per-epoch trajectories for V1) are in the JSONs; the `summary` block of
each holds the per-cell aggregates cited below.

Provenance: copied 2026-07-31 from `/mnt/k/output_heads/{rbd,casp3}/` (the
live results drive). Original K: paths listed per file. Nothing here is
regenerated; the scripts below read these JSONs.

## Files

| repo file | K: source | runs | what | analysis script |
|---|---|---|---|---|
| `v1_rbd_full.json` | `rbd/ab_bestepoch/results_full.json` | 45 | **V1 canonical** — dual-val + held-out test, cells sp_no/sp_msa/x_msa_faesm | `ab_bestepoch_testset.py` (summary), `plot_ab_v1.py` |
| `v1_rbd_full_esm.json` | `rbd/ab_bestepoch/results_full_esm.json` | 30 | **V1 canonical** — cells sp_faesm/x_no_faesm (added later; together with the above = the 5-cell V1) | same |
| `v1_rbd_main_pilot.json` | `rbd/ab_bestepoch/results_main.json` | 45 | **SUPERSEDED early pilot** (sp_no/sp_msa/x_msa_faesm). Different protocol/numbers from the canonical V1 (e.g. sp_msa best_v1 0.642 vs 0.661; x_msa_faesm 0.566 vs 0.654). Kept for provenance only — do NOT cite. | — |
| `v1_casp3_main.json` | `casp3/ab_bestepoch/results_main.json` | 75 | **V1 CASP3 replication**, 5 cells × 15 | `ab_bestepoch_casp3.py`, `analyze_casp3_v1.py`, `plot_ab_v1.py` |
| `v2_std_subset.json` | `rbd/ab_std_split/results_main.json` | 240 | **V2 subset** (N=500): 4 std arms {none, overall, strict, tglob} × 4 cells | `ab_std_split.py` (summary) |
| `v2_std_full_sp.json` | `rbd/ab_std_split/results_full.json` | 60 | **V2 full set**, sp_no/sp_msa × {overall, strict} | `ab_std_split.py` (summary) |
| `v2_std_full_fusion.json` | `rbd/ab_std_split/results_full_fusion.json` | 90 | **V2 full set**, 6 fusion cells × strict (paired vs archived v3 overall) | `analyze_strict_fusion.py` |
| `v3_esm_noise.json` | `rbd/ab_esm_noise/results.json` | 75 | **V3** ESM-noise ladder on x_msa_faesm, nl {0,0.25,0.5,0.75,1.0} × 15; per-run late z/f encoder grad norms | `analyze_ab_noise_pace.py` |
| `v4_pace_fenc.json` | `rbd/ab_pace_fenc/results.json` | 60 | **V4** f_enc LR pace on x_msa_faesm, λ {1.0,0.25,0.125,0.06} × 15; per-run late z/f grad norms | `analyze_ab_noise_pace.py` |

## Conclusions (see docs/findings.md §1/§4/§7b and docs/overnight_report.md)

**V1 — best-epoch overfit is real and quantified (RBD).** Single-val
best-epoch selection overstates ρ by +0.04–0.09 (sp_msa worst: sel_optim
+0.094, 5/5 splits). Two independent de-biasing routes converge: held-out
val2@same-epoch MSA effect +0.141 / best_val2 +0.143, and strict preprocessing
+0.146. **Honest RBD MSA effect ≈ +0.14–0.15; honest SP(Z_msa) ≈ 0.59**, not
0.644 (best-val overall = optimistic bound). sp_msa shows a val1>val2 late gap
of −0.061 (5/5 splits, locked by ep5 — feature-geometric).

**V1 — CASP3 replication falsifies the pinning-driven gap hypothesis.** CASP3
sp_msa late gap is only −0.023 (2/5 splits, one outlier drives it), ep5 gap
+0.004, despite CASP3's *stronger* pinning (−42% vs −26%). Stability inverts:
on CASP3 Z_msa is the stable modality (sp_msa traj_r 0.66 > sp_faesm 0.32),
the reverse of RBD. Fusion inherits whichever single stream is dominant/stable
per protein — sp_msa on CASP3 (fusion↔sp_msa r 0.90 > ↔sp_faesm 0.82), sp_faesm
on RBD. Honest CASP3 MSA effect +0.14–0.18 (same class as RBD).

**V2 — preprocessing leakage shifts absolutes, not the triad ordering.**
Strict (train-only std fit) lowers single-modality absolutes, loss scaling with
reliance on Z_msa: sp_no 0.465→0.443 (−0.022, 10/15), sp_msa 0.644→0.589
(−0.055, 15/15). Fusion cells drop 0.007–0.039 (x_msa_zeros most, x_no_zeros
least). Dead-stream marginals are convention-robust: FAESM-over-Z_msa
−0.005→+0.016 (≈0 either way), FAESM-over-Z_no +0.075→+0.070, Z_no-over-
(Z_msa+FAESM) +0.002→+0.003. Only Z_msa-over-Z_no moves >±0.02 (+0.197→+0.165).
**Triad ordering (H1/H2) is convention-robust; report absolutes under strict.**

**V3 — ESM-noise gives an inverted-U, mechanism = gradient rebalancing.**
x_msa_faesm val ρ by nl: 0.0→0.649, 0.25→0.655, 0.5→0.659, **0.75→0.662
(peak, +0.013)**, 1.0→0.648 (back to baseline, paired Δ −0.002). Late f/z
gradient ratio collapses monotonically 2.91→0.04 (z-encoder grad ↑8×, f-encoder
↓9×): noising FAESM removes its gradient dominance and unleashes Z_msa. At
nl=1.0 (pure noise) the result ≈ the x_msa_zeros dead-stream cell (0.648 ≈
0.655) — full noise merely kills the stream; the benefit is at the partial-
noise sweet spot, not "noise beats information". Peak at nl=0.75 matches the
fusion_as_validator CASP3 inverted-U.

**V4 — f_enc LR pace gives a plateau, same direction.** val ρ by λ:
1.0→0.649, **0.25→0.663 (argmax, +0.014, 5/5 splits — cleanest arm)**,
0.125→0.661, 0.06→0.662. Slowing the F encoder raises the z-encoder gradient
~4× (0.066→0.287) and recovers ~+0.013, the content-cost-free analogue of V3.
Both V3/V4 support the gradient-interference account: on RBD FAESM is
information-redundant (marginal ≈0) but optimization-harmful — suppressing its
gradient dominance recovers a small, robust part of Z_msa's contribution.
Effect size ~+0.013–0.014 is modest; means exceed the 0.644 sp_msa ceiling but
the paired gain is a marginal (not strongly significant) improvement.
