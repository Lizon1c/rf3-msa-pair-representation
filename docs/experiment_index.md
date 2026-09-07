# Experiment Index — msa_role_properties

Status: ✓ done · ✓ done · queued. Numbers are split-paired Spearman ρ unless noted.

## Extraction / data prep
| ID | Script | What | Status |
|---|---|---|---|
| MSA-build | `extraction/build_msas.sh` | **E2-target MSAs only** (UniRef30, cap 256). RBD = pre-existing 15-row GOLD MSA (18 rows length-filtered to 15); CASP3 `casp3_human.a3m` (69 Swiss-Prot homologs) and CI `ci_lambda.a3m` (512 UniRef30 rows) were built from local mmseqs2 DBs — **build scripts for these two not archived** (provenance gap) | ✓ |
| E1-extract | `launchers/launch_e1_msa_arms.py` | RBD Z_II, 8 MSA arms (depth01/04/08/full15, colshuffle, consensus, rowshuffle, random), 500-subset + full15 full-set | ✓ |
| E7-extract | `launchers/launch_e7_recycle_gate.py` | RBD Z_II, recycle-gated MSA (k1/k2/k3) via `RF3_MSA_CYCLES` env hook | ✓ |
| CI-extract | `launchers/launch_ci_msa.py` | CI Z_II clean noMSA + MSA (true-site row+col; fixes slice@0) | ✓ |
| E2-extract | `launchers/launch_e2_targets.py` + `analysis/e2_06_ga_gb.py` | 13 two-state targets × {noMSA, MSA} distograms + GA88/GB88 | ✓ |

## Geometry / representation
| ID | Script | Finding | Status |
|---|---|---|---|
| E0 | `analysis/e0_msa_geometry.py` | **retracted-table replication only** — reads the slice@0 artifact (`/tmp/e0/zii_msa` = staged `zii_wh1_msa`) and uses alphabetically-sorted mutant_id indexing; its archived JSON is INVALID (warning key inside), kept for provenance. Corrected RBD full-set geometry — eff90 231→171, CKA 0.527, supp 1.21, dCos 0.474 — comes from `analysis/e1_analyze.py` (`results/e1_arms/results_fullset.json`) | ✓ |
| E1-arms | `analysis/e1_analyze.py` + `e1_supervised.py` | dose×content decomposition; random collapses rank but no functional gain; conservation ~2/3, covariation ~1/3. Supervised arms use the rbd_repro V2 head + scalar std + 500-subset (NOT the canonical protocol — absolute values not comparable); the zero-content baseline is depth01, not noMSA (+0.076 featurization-path gap) | ✓ |
| E7 | (E1 scripts on k-arms) | one recycle = ~75% geometric / ~85% functional effect; near-irreversible | ✓ |
| E4-CASP3 | `analysis/e4_casp3_msa_analysis.py` | eff90 333→193 (−42%), Δ+0.162 | ✓ |
| E4-CI | `analysis/e4_ci_msa_analysis.py` | eff90 61→51 (−16%), Δ+0.075 | ✓ |

## Fusion matrix (validator protocol, dead-stream controlled)
| ID | Script | Finding | Status |
|---|---|---|---|
| E2-4cond-RBD | `analysis/e2_four_condition.py` | A/B run: sp_no 0.442 / sp_msa 0.646 / x_no_msa 0.657 / x_msa_faesm 0.655, paired Δ+0.204 (5/5); archived v3 run: 0.465/0.644/0.646/0.651, paired Δ+0.179 (5/5); FAESM & Z_no marginals over Z_msa ≈ 0 in both | ✓ |
| E2-matrix-RBD | `analysis/e2_matrix_complete.py` | + sp_faesm 0.484 / x_no_faesm 0.534 / x_faesm_zeros 0.491 / tri 0.650 / tri_z 0.648 (archived v3); tri−tri_z = +0.002 ≈ 0 | ✓ |
| E2-bin | `analysis/e2_bin_analysis.py` | per-bin ρ + adjacent-bin AUC; MSA gain in deleterious-bin ranking + high-label separation. Canonical aggregate = `results/e2_bin/bin_agg_joint.json` (the K-drive `results.json` was overwritten by a later partial run and holds only x_no_msa/x_msa_faesm; all per-run preds npz retained on K) | ✓ |
| E5-CASP3 | `analysis/e5_matrix_protein.py casp3` | full matrix on CASP3 | ✓ |
| E5-CI | `analysis/e5_matrix_protein.py ci` | full matrix on CI (per-channel clean rerun) | ✓ |

## Distillation (D1)
| ID | Script | Finding | Status |
|---|---|---|---|
| D1 | `analysis/d1_distill.py` | (Z_no+ESM)→Z_msa (archived per-channel run, best-epoch val): lin R²0.25/CKA0.74 → mlp512 0.28/0.83 → tf2 0.36/0.80 (CKA plateaus ~0.8 from mlp512); zonly vs fonly at mlp512 R² 0.267 vs 0.222 (≈ tied, not 3×) | ✓ |
| D1-func | `analysis/d1_functional.py` | reconstructed Z_msa predicts bind_avg 0.575 (Z_no 0.446 → real 0.663), ~60% of gap | ✓ |
| D1-deep | `analysis/d1_deep.py` | depth ladder TF{2,3,4,6,8} × {both,zonly,fonly}; train-CKA vs val-CKA memorization gap (TF8: both 0.903 ≈ fonly 0.895 > zonly 0.783; gap +0.09) | ✓ |

## Conformational-state selection (E2)
| ID | Script | Finding | Status |
|---|---|---|---|
| E2-conf | `analysis/e2_analyze.py` | 13 targets: fold-switchers split (KaiB/RfaH away, Mad2/XCL1 toward), controls null. **Range-matched rerun DONE 2026-08-02 (`analysis/e2_rangematched_analyze.py` → `/mnt/k/output_heads/e2/e2_rangematched_results.json`)**: shared residue universes per target + RfaH stateA compression-bug fix (2OUG lacks UniProt 101–114; old rows ≥99 misaligned by 14). Flagships survive (XCL1 +4.81→+2.87; RfaH −1.42→−2.99 doubles; KaiB −2.29; Mad2 −0.91); LacY/EGFR collapse to null (the "GPCR/kinase/transporter toward" claim shrinks to β2AR +0.15, inside noise); GA88/GB88 now scored in-flow (+2.56 sharpening); XCL1 steer_tpl anomaly resolved (−1.28→+0.63). Caveats: single forward per target; pre-matching magnitudes superseded | ✓ |
| E2-06 | `analysis/e2_06_ga_gb.py` | GA88/GB88: RF3 predicts each seq → own fold; MSA sharpens not redirects | ✓ |

## Methods audit
| ID | Script | Finding | Status |
|---|---|---|---|
| AB-std | `analysis/ab_std_check.py` | scalar-per-position std degrades fusion cells (+0.037 on x_no_faesm); per-channel canonical. Evidence strength: 2 splits × 2 inits quick check, originally print-only (fig8 values hard-coded, no archived JSON); the script now writes `rbd/ab_std_check/results.json` on reruns | ✓ |
| slice-fix | `analysis/fix_e1_wt_replicates.py` | WT-replicate full-block → row+col conversion | ✓ |

## A/B validation (collaborator concerns: best-epoch overfit + preprocessing leakage)

Archived raw JSONs + manifest: `results/ab_validation/` (copied from
`/mnt/k/output_heads/{rbd,casp3}/`; see its README.md for per-file provenance,
schema, and the superseded-pilot note).

| ID | Script | Finding | Status |
|---|---|---|---|
| AB-V1-RBD | `analysis/ab_bestepoch_testset.py` | dual-val (60/20/20 cross-position) + held-out test, 5 cells × 15. Single-val best-epoch overfits val1 by +0.04–0.09 (sp_msa worst, sel_optim +0.094 5/5). De-biased MSA effect: val2@same +0.141 / best_val2 +0.143 / strict +0.146 → honest ≈ +0.14–0.15, honest SP(Z_msa) ≈ 0.59. sp_msa val1>val2 late gap −0.061 (5/5, ep5-locked) | ✓ |
| AB-V1-CASP3 | `analysis/ab_bestepoch_casp3.py` | CASP3 replication, 5 cells × 15. Prediction (pinning-driven gap) FALSIFIED: sp_msa late gap −0.023 (2/5, one outlier), ep5 +0.004, despite stronger pinning. Stability inverts (sp_msa traj_r 0.66 > sp_faesm 0.32, reverse of RBD). Fusion tracks the dominant/stable single stream per protein (CASP3 ↔sp_msa 0.90 > ↔sp_faesm 0.82; RBD the reverse). Honest CASP3 MSA effect +0.14–0.18 | ✓ |
| AB-V2-strict | `analysis/ab_std_split.py` | strict (train-only std fit) vs overall, paired. sp_no 0.465→0.443, sp_msa 0.644→0.589 (−0.055, 15/15); MSA effect +0.179→+0.146. Full fusion triad (90/90, `analyze_strict_fusion.py`): all cells −0.007…−0.039; marginals convention-robust except Z_msa-over-Z_no +0.197→+0.165 | ✓ |
| AB-V3-noise | `analysis/ab_esm_noise.py` | ESM-noise ladder on x_msa_faesm (RBD), nl {0,0.25,0.5,0.75,1.0} × 15 (`analyze_ab_noise_pace.py`). **Inverted-U**: val ρ 0.649→0.655→0.659→**0.662 (peak nl=0.75, +0.013)**→0.648 (nl=1.0 back to baseline, paired Δ −0.002). Late f/z gradient ratio collapses 2.91→0.04 (z-enc grad ↑8×, f-enc ↓9×): noising removes FAESM's gradient dominance, unleashes Z_msa. nl=1.0 ≈ x_msa_zeros (0.648≈0.655) — full noise just kills the stream; benefit is the partial-noise sweet spot, not "noise > information". Peak nl=0.75 matches fusion_as_validator CASP3. Archive `results/ab_validation/v3_esm_noise.json` | ✓ |
| AB-V4-pace | `analysis/ab_pace_fenc.py` | f_enc LR pace on x_msa_faesm (RBD), λ {1.0,0.25,0.125,0.06} × 15 (`analyze_ab_noise_pace.py`). **Plateau**: val ρ 0.649→**0.663 (λ=0.25, +0.014, 5/5 splits — cleanest arm)**→0.661→0.662. z-enc grad ↑~4× (0.066→0.287); content-cost-free analogue of AB-V3, same gradient-interference direction. Archive `results/ab_validation/v4_pace_fenc.json` | ✓ |
| D1-deep-traj | `analysis/d1_deep_traj.py` | epoch-resolved grokking probe, TF{2,4,8} × {both,fonly,zonly}; is the TF2→TF8 train-val gap collapse temporal grokking (late val-CKA jump) or a depth effect? Script ready and compile-checked; **not run** — deferred by user 2026-07-31 (no further experiments this round) | deferred |

## Key caveats (carry into manuscript)
1. **Retraction**: original MSA-collapse numbers were slice@0 artifacts (§0 of findings.md).
2. **Std convention**: per-channel canonical; scalar-per-position specifically hurts fusion cells.
3. **E2 range-matched (RESOLVED 2026-08-02)**: `analysis/e2_rangematched_analyze.py` scores both states on shared residue universes (and fixes the RfaH-2OUG 14-residue compression misalignment); pre-matching XCL1/RfaH magnitudes superseded (findings.md §6). Remaining E2 caveat: KaiB/RfaH alt-states are NMR refs; single forward per target with run-to-run pref noise ±0.14–0.27 (post-matching, only the fold-switcher shifts sit outside it).
4. **D1 std**: the archived D1 scripts use the canonical per-channel std (imported `std_per_residue`). An earlier scalar-std run produced the shallow-ladder numbers now retracted in findings.md §5 (lin 0.33/0.71 → tf2 0.60/0.90, "Z_no 3× > ESM"); the archived `d1_distill/results.json` holds the per-channel values cited there.
5. **Preprocessing leakage + best-epoch overfit**: the archived matrices standardize Y on all samples before splitting and select best epoch on a single val set — both optimistically biased. Strict (train-only) standardization and held-out-test selection independently converge on honest RBD MSA effect ≈ +0.14–0.15 and SP(Z_msa) ≈ 0.59 (not 0.644). Fusion-triad *ordering* is convention-robust; single-modality absolutes and the Z_msa-over-Z_no magnitude are convention-dependent. Report strict/test-set as primary, best-val overall as the optimistic bound (findings.md §1/§7b, overnight_report.md §2–3).
