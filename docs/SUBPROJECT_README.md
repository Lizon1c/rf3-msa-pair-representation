# MSA Role Properties — What does the MSA actually do inside a structure predictor?

**Companion audit to `fusion_as_validator/`.** A systematic, multi-split,
content-decomposed investigation of the MSA input's role in RF3 (an open
AlphaFold3-style trunk) for protein mutation-effect prediction — testing the
hypothesis that the MSA acts as a *crystallographic template / family-consensus
write* rather than a clean coevolution-information channel.

This repository accompanies the manuscript *"Multimodal Joint Embedding of Pair
Features and the Properties of Fusion Architectures"* (in preparation); it is the
"MSA vs no-MSA" thread reserved for expansion there.

---

## Headline results

All supervised numbers are **split-paired** (≥3 position splits × 2–3 inits) on
frozen embeddings with small heads (Spearman ρ, cross-position validation).

### 0. A retraction first — the original "MSA collapse" claim was artifacted

The June 2026 with-MSA extraction sliced every mutant's Z_II at **position 0**
(`ci_bench`-style `z[0,:L,:]` bug, same family as the WH1 extraction), not the
mutation site. The reported "effective-rank collapse 231→30, supervised −40%,
MSA-free should be default" is **invalid**. Clean re-extraction (true-site
row+col) gives the corrected picture below. (See `docs/findings.md` §0.)

### 1. MSA pins the pair representation — mildly — but *improves* prediction

| protein | MSA depth | eff90 collapse | CKA(msa,no) | dCos | supervised Δ (msa−no) |
|---|---|---|---|---|---|
| RBD | 15 (shallow) | 231→171 (−26%) | 0.527 | 0.474 | **+0.18–0.20** (5/5 × 2 runs) |
| CASP3 | 69 | 333→193 (−42%) | 0.553 | 0.362 | **+0.162** (5/5) |
| CI | 512 (deep) | 61→51 (−16%) | 0.917 | 0.844 | **+0.075** (4/5) |

RBD Δ comes from two independent per-channel full-set runs, both 5/5 splits
positive: paired +0.179 (`e2_matrix_rbd/four_condition_v3.json`: sp_no
0.465/sp_msa 0.644) and +0.204 (`e2_matrix_rbd/four_condition_A.json`: sp_no
0.442/sp_msa 0.646). The 0.025 spread between runs sits at the project's
measured cross-GPU/process noise floor (E14/E16).

**Honest (de-biased) numbers.** Those Δ are best-val under overall
standardization and are optimistically biased. Two independent corrections
converge: held-out-test (val2@same-epoch +0.141 / best_val2 +0.143) and
strict train-only standardization (+0.146; sp_msa 0.644→0.589, sp_no
0.465→0.443). **Honest RBD MSA effect ≈ +0.14–0.15, honest SP(Z_msa) ≈ 0.59**;
the best-val overall figures are the optimistic upper bound. See
`docs/findings.md` §1/§7b and `docs/overnight_report.md` §2–3.

The MSA compresses the pair representation toward a family consensus (mild
rank collapse, partial direction preservation), yet this *helps* cross-position
mutation-effect prediction on all three proteins. The gain **tracks the
rewriting magnitude** cross-protein (lower CKA ⇒ larger Δ).

### 2. Content decomposition: shared-context tax + conservation + covariation

E1 dose×content arms (RBD, eff90 / supervised): depth01 > consensus > colshuffle
> full15. Even **random sequences** collapse rank (a generic "shared-context
tax"), but the **functional gain comes only from real biological content** —
mostly per-column conservation (~2/3), covariation secondary (~1/3). Saturates
at ~8 sequences; family *coherence*, not raw depth, is the operative variable
(CI's deep-but-diverse MSA gives the weakest effect).

**Baseline note**: the honest zero-content baseline is **depth01** (a 1-row
a3m), not noMSA. noMSA (no a3m at all) → depth01 is +0.076 in supervised ρ
(0.414→0.490) — a featurization-path effect of presenting any a3m, not
biology. Against depth01, random/rowshuffle add ~0 (0.479/0.477 vs 0.490) and
the biological gain is depth04 +0.046, colshuffle +0.106, depth08 +0.136,
full15 +0.148.

### 3. Write-once dynamics (E7, recycle gating)

Gating MSA to the first K recycles: **one MSA recycle installs ~75% of the
geometric effect and ~85% of the functional gain**; 9 subsequent MSA-free
recycles wash out only ~25%. The family-consensus write is near-irreversible —
the trunk cannot self-correct it.

### 4. Fusion-matrix triad: on RBD, MSA-Z_II ⊇ {no-MSA Z_II, FAESM}

Complete RBD per-channel matrix (11 cells × 15 runs, dead-stream-controlled).
All values from the archived v3 run (`e2_matrix_rbd/four_condition_v3.json` +
`e2_matrix_rbd/matrix_complete_v3.json`); the independent A/B run's dead-stream
controls are x_no_zeros 0.434 / x_msa_zeros 0.657 (`four_condition_B.json`):

| cell | ρ | | cell | ρ |
|---|---|---|---|---|
| sp_no | 0.465 | | x_no_zeros | 0.459 |
| sp_msa | 0.644 | | x_msa_zeros | 0.655 |
| sp_faesm | 0.484 | | x_faesm_zeros | 0.491 |
| x_no_msa | 0.646 | | tri (Z_no+Z_msa+FAESM) | 0.650 |
| x_msa_faesm | 0.651 | | tri_z (zeros+Z_msa+FAESM) | 0.648 |
| x_no_faesm | 0.534 | | | |

Dead-stream-corrected marginals: FAESM over Z_msa = x_msa_faesm − x_msa_zeros =
**−0.005 ≈ 0** (H2: MSA duplicates ESM); Z_no over Z_msa = x_no_msa − x_msa_zeros =
**−0.009 ≈ 0**; Z_no over (Z_msa+FAESM) = tri − tri_z = **+0.002 ≈ 0** (H1: MSA
dominates the structural signal); FAESM over Z_no = x_no_faesm − x_no_zeros =
**+0.075** (FAESM carries signal Z_no lacks, but Z_msa already has it). Architecture
effects small (±0.01). **Z_msa alone (0.644) captures essentially everything;
neither Z_no nor FAESM adds beyond it.** No complementarity (H3 excluded on RBD).

**Cross-protein triad (dead-stream-corrected):**

| protein | sp_no | sp_msa | sp_faesm | FAESM over Z_msa | Z_no over (Z_msa+FAESM) |
|---|---|---|---|---|---|
| RBD | 0.465 | 0.644 | 0.484 | −0.005 ≈ 0 | +0.002 ≈ 0 |
| CASP3 | 0.394 | 0.556 | 0.584 | **+0.051** (5/5 splits) | −0.001 ≈ 0 |
| CI | 0.374 | 0.449 | 0.452 | −0.039 (noisy, 2/5) | +0.008 ≈ 0 |

**Z_no's marginal over (Z_msa+FAESM) ≈ 0 on all three proteins (H1 holds
cross-protein).** FAESM's marginal over Z_msa is protein-dependent: ≈0 on RBD,
+0.051 on CASP3 (5/5 splits positive — MSA does not fully subsume ESM there),
and −0.039 on CI, which is **not a clean ≈0**: the mean is driven by one
outlier split (per-split [−0.015, +0.077, −0.235, +0.017, −0.038], 2/5
positive; CI split σ~0.09 at N=351) — consistent with zero or a small
negative. H2 (MSA≈ESM) holds on RBD, is partial on CASP3, and is not cleanly
resolved on CI.

**Strict-standardization robustness (full set, 90/90).** Re-running all six
fusion cells with train-only (strict) standardization lowers every cell by
0.007–0.039 (x_msa_zeros most, x_no_zeros least) but leaves the dead-stream
marginals essentially unchanged: FAESM-over-Z_msa −0.005→+0.016 (≈0 either
way), FAESM-over-Z_no +0.075→+0.070, Z_no-over-(Z_msa+FAESM) +0.002→+0.003.
The only marginal moving more than ±0.02 is Z_msa-over-Z_no (+0.197→+0.165).
**The triad ordering (H1/H2) is convention-robust**; only single-modality
absolutes and the precise Z_msa-over-Z_no magnitude are convention-dependent
(`docs/findings.md` §7b, `docs/overnight_report.md` §2.3).

Gradient economy runs *opposite* to the information accounting: FAESM's encoder
dominates late-training gradients on all three proteins (RBD tri f/z≈6.5,
f/m≈2.7; CASP3/CI f/m≈3.5–3.8), and Z_msa's encoder exceeds Z_no's only on RBD
(m/z≈2.4; CASP3/CI m/z≈0.6–0.8) — and only transiently (early in RBD training
m/z peaks at ~5.8 before Z_msa's signal is absorbed; f/m crosses 1 near epoch
10). Gradient dominance ≠ information (the audit's central lesson, reproduced
in the MSA context).

### 5. Distillation (D1): MSA content is partially reconstructible from (Z_no, ESM)

A (Z_no + FAESM) → Z_msa mapping (archived `d1_distill/results.json`, best-epoch
val, 3 splits × 2 inits): linear R²=0.25 / CKA 0.74 → mlp512 0.28 / 0.83 → TF-2L
0.36 / 0.80 — CKA plateaus at ~0.8 from mlp512 (tf1/tf2 do not exceed mlp512).
At mlp512, Z_no-alone is only slightly ahead of ESM-alone (R² 0.267 vs 0.222;
CKA 0.748 vs 0.752); the earlier "~3×" claim came from a superseded
scalar-standardization run and is not reproducible from the archived data.
Functionally, reconstructed Z_msa (TF-2L) recovers ~60% of the real Z_msa's
prediction advantage over Z_no (0.575 vs Z_no 0.446, real 0.663). D1-deep extends
the depth ladder (last-epoch metrics; val-CKA rises to 0.90 at TF8, where fonly
matches both) and D1-XAttn-CKA (separate per-modality encoders) shows
both 0.81 > fonly 0.75–0.76 > zonly 0.72 stably across depth — Z_no's
contribution is real but was masked by the shared-projection architecture at
shallow depth.

### 6. Conformational-state selection (E2): MSA is a context-dependent state selector

13 two-state targets (fold-switchers, GPCR, kinases, transporters, X-ray-vs-NMR).
Controls clean (shallow-MSA selecase and X-ray-vs-NMR pairs ≈ 0 shift).
GPCR/kinase/transporter shift **toward the crystallized state**. Fold-switchers
**split**: KaiB/RfaH flip *away* from the crystal ground state toward the
alternative fold; Mad2/XCL1 shift toward crystal; GA88/GB88 — RF3 predicts each
sequence into its *own* fold and MSA *sharpens* (not redirects) the preference.
The MSA's conformational influence is context-dependent: sequence dominates for
near-identical fold-switch pairs; family consensus can redirect divergent cases.

**Caveats (E2).** Single forward per target, no seed replicates; repeat runs of
identical inputs (archived raw distograms) show a preference noise band of
±0.14–0.27. **Range-matched rerun DONE (2026-08-02,
`scripts/analysis/e2_rangematched_analyze.py` →
`/mnt/k/output_heads/e2/e2_rangematched_results.json`)**: both states are now
scored on shared residue universes (also fixing a newly-found RfaH-2OUG
compression bug — the missing 101–114 loop had rows ≥ 99 misaligned by 14).
Post-matching secure signals (outside the noise band): XCL1 +2.87 (was +4.81),
RfaH −2.99 (was −1.42, *strengthened*), KaiB −2.29, Mad2 −0.91, GA88/GB88
+2.56 (now scored in the main flow). LacY (+0.94 → −0.08) and EGFR collapse
to null — the old "GPCR/kinase/transporter toward crystal" group shrinks to
β2AR +0.15 (~1σ). XCL1's E49 anti-template anomaly also resolves
(steer_tpl −1.28 → +0.63). Pre-matching magnitudes are superseded; see
findings.md §6 for the full table.
"Toward crystal" signs are per-target (Mad2's crystallized state is state B;
fig7 sign-corrects for this). Tally (pre-matching): 7 toward crystal (2 within
the noise band), 2 away (KaiB/RfaH), 3 null controls (|shift| ≤ 0.03); GA88/GB88 is
scored by `e2_06_ga_gb.py`, not present in the main JSON (whose E2-06 row is
stale, msa=null).

### 7. Methods bug found & fixed: three standardization conventions

The project mixed three per-residue standardization conventions. A/B test proved
**scalar-per-position specifically degrades fusion cells** (x_no_faesm
0.483→0.521 with per-channel, +0.037) while single-modality cells are robust.
All matrices re-run with the canonical per-position **per-channel** convention.

---

## Repository layout

```
msa_role_properties/
├── README.md                 this file
├── docs/
│   ├── findings.md           detailed findings + retraction + caveats
│   ├── experiment_index.md   every experiment, status, key numbers
│   └── lit_review.md         MSA-in-structure-prediction literature (verified quotes)
├── figures/                  (generated by make_figures.py)
├── make_figures.py
├── results/                  per-experiment results.json + logs
│   ├── e0_geometry/          RETRACTED slice@0 replication run (INVALID note inside)
│   ├── e1_arms/  e1_supervised/  e2_matrix_rbd/  e2_bin/
│   ├── e4_casp3/  e4_ci/  e5_matrix_casp3/  e5_matrix_ci/
│   ├── d1_distill/           incl. deep_{both,f,z}/onehot/xattn_cka files
│   ├── e2_conformation/
│   (ab_std_check has no archived JSON — see fig8 provenance note)
└── scripts/
    ├── analysis/             e0–e5, d1, ab_std, build/fix helpers
    ├── launchers/            RF3 extraction launchers (e1 arms, e7, ci, e2 targets)
    └── extraction/           build_msas.sh (E2-target MSA construction only)
```

## Reproduction

Embeddings: RF3 pair representation Z_II extracted at the distogram head
(`RF3.py`, env-gated `RF3_ZII_PATH` / `RF3_DISTOGRAM_PATH` hooks), with and
without MSA, per mutant, sliced at the true mutation site (row+col). MSAs: RBD
uses the pre-existing 15-row GOLD MSA; CASP3 `casp3_human.a3m` (69 Swiss-Prot
homologs) and CI `ci_lambda.a3m` (512 UniRef30 rows) were built from local
mmseqs2 DBs (build scripts for these two not archived — provenance gap);
`scripts/extraction/build_msas.sh` builds the **E2-target** MSAs only
(UniRef30, cap 256), not RBD/CASP3/CI.

Two supervised protocols are in use — numbers are **not interchangeable**
across them:

- **Canonical (E2 matrices, E4, E5, D1, ab_std)**: `fusion_v3.SinglePredictorV2`
  / `CrossAttnOrthoConcatFusion`, per-position **per-channel** standardization
  (`X[:, pos].std(0)` must yield a vector, not a scalar), cross-position 70/30
  splits (seeds 100–104), 3 inits (7/107/207), 300 ep, AdamW 1e-4/0.05,
  best-epoch val Spearman, raw labels.
- **E1/E7 dose curves** (`e1_supervised.py`): rbd_repro V2 `ZII_Model` head (an
  Encoder variant without fusion_v3's GELU/dropout), per-position **scalar**
  standardization, 500-mutant stride-8 subset. The scalar convention moves
  single-modality cells by ≤0.02 (ab_std_check), so the arm ranking and
  decompositions survive, but absolute values are not comparable to the
  canonical protocol.

See `docs/experiment_index.md` for per-experiment commands.
