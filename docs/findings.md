# Findings — the MSA's role inside an AF3-style trunk

Detailed narrative companion to `README.md`. Numbers are split-paired unless noted.

---

## §0. Retraction: the original "MSA collapse" claim was a slicing artifact

The June 2026 with-MSA Z_II extraction (and the WH1 RBD extraction) saved
`z[0, :L, :]` — the pair-representation **row at position 0** — for *every*
mutant, instead of the row at the mutation site. Because position 0 of a
(weakly) MSA-pinned pair matrix barely varies across mutants, the ensemble
looked artificially low-rank (eff90 30) and the supervised signal looked
destroyed (−40%). The no-MSA extraction was sliced correctly, so the *contrast*
was bogus.

**Corrected (true-site row+col, full N=3998 RBD):** eff90 231→171 (−26%, not
−87%); CKA(msa,no)=0.527; per-mutant delta direction preserved at 0.474 (not
erased); delta norms *amplified* (supp 1.21, not suppressed 0.23). The
supervised effect **reverses sign**: MSA *improves* prediction (+0.18–0.20
across two independent per-channel runs, 5/5 splits each).

Lesson carried into the manuscript: per-mutant structural features must be
sliced at the mutation site; a position-0 slice is a silent, severe artifact.

## §1. The corrected geometry: mild pinning, helpful function

On all three proteins the MSA compresses the pair representation toward a family
consensus — a *mild* rank collapse with *partial* direction preservation — and
this compression *helps* cross-position mutation-effect prediction:

- RBD (15-row shallow MSA): eff90 231→171, CKA 0.527, dCos 0.474, Δ+0.18–0.20
  across two independent per-channel full-set runs (5/5 splits each: +0.179
  archived v3 run, +0.204 A/B run; run-to-run spread 0.025 ≈ the measured
  cross-GPU noise floor).
- CASP3 (69-row): eff90 333→193 (−42%), CKA 0.553, dCos 0.362, Δ+0.162 (5/5).
- CI (512-row deep): eff90 61→51 (−16%), CKA 0.917, dCos 0.844, Δ+0.075 (4/5).

The supervised gain **tracks the rewriting magnitude** cross-protein: lower
CKA(msa,no) ⇒ larger Δ. CI's deep-but-diverse MSA produces the *softest*
consensus (highest CKA, smallest gain); RBD's shallow, coherent MSA the
*hardest*. **Family coherence, not raw MSA depth, is the operative variable** —
a direct refinement of the naive "deeper MSA = more information" assumption
(and of AF2's ~30-sequence threshold, which is about family *identification*).

**Honest (de-biased) effect size.** The Δ above are best-val under overall
standardization and are optimistically biased by two independent routes:
(i) single-val best-epoch selection overfits val1 (+0.04–0.09; sp_msa worst),
and (ii) overall standardization leaks val/test statistics (§7b). Two
de-biasing corrections converge on the same number on RBD: held-out-test
(val2@same-epoch +0.141 / best_val2 +0.143) and strict preprocessing (+0.146).
**Honest RBD MSA effect ≈ +0.14–0.15; honest SP(Z_msa) ≈ 0.59** (strict 0.589 /
best_val2 0.617), not 0.644. The best-val overall figure (+0.179/0.204, sp_msa
0.644) is retained only as the optimistic upper bound.

**CASP3 dual-val replication (`ab_bestepoch_casp3`, 75/75).** The RBD
val1>val2 late offset of sp_msa (−0.061, 5/5 splits, ep5-locked) does **not**
replicate on CASP3: there the sp_msa late gap is −0.023 (2/5 splits, one outlier
drives the mean), ep5 gap +0.004 — despite CASP3's *stronger* pinning (−42% vs
−26%). The pinning-driven hypothesis for RBD's offset is falsified; the offset
is RBD-specific. Stability also inverts: on RBD FAESM is the stable modality
(sp_faesm traj_r 0.69, sp_msa 0.21); on CASP3 Z_msa is (sp_msa 0.66, sp_faesm
0.32). The fusion readout inherits **whichever single stream is dominant/stable
per protein** — FAESM on RBD (fusion↔sp_faesm r 0.90/0.76, ↔sp_msa 0.37), Z_msa
on CASP3 (fusion↔sp_msa 0.90, ↔sp_faesm 0.82) — not "FAESM specifically". The
honest CASP3 MSA effect (sp_msa−sp_no) is +0.14–0.18, same magnitude class as
RBD: the headline contribution is protein-robust even though the gap/stability
mechanism is protein-specific. See `docs/overnight_report.md` §3.

## §2. Content decomposition (E1 dose×content arms)

Eight RBD MSA arms separate three channels. (Supervised numbers here are from
`e1_supervised.py`: rbd_repro V2 `ZII_Model` head + per-position **scalar**
std + 500-subset — a different protocol from the canonical per-channel
full-set runs; arm comparisons are within-protocol, and the ≤0.02 convention
effect does not change the ranking.)

- **Shared-context tax**: even *random* sequences (rowshuffle/random) collapse
  rank to ~149–155 (vs noMSA 181) — adding *any* shared per-residue context
  diverts representational capacity. But this buys **no functional gain**:
  the honest zero-content baseline is depth01 (a 1-row a3m), not noMSA —
  noMSA→depth01 (+0.076, 0.414→0.490) is a featurization-path effect of
  presenting any a3m at all, and random/rowshuffle sit at/below depth01
  (0.479/0.477 vs 0.490).
- **Per-column conservation**: consensus (identical consensus rows, ρ 0.501)
  and colshuffle (per-column frequencies preserved, covariation destroyed,
  ρ 0.596). In eff90 the two are not cleanly ordered (colshuffle 150 ≈ random
  149); the clean separation is in dCos (consensus 0.815 vs colshuffle 0.548)
  and supervised ρ. Conservation carries ~2/3 of the biological gain
  (consensus→colshuffle +0.095).
- **Pairwise covariation**: full15 vs colshuffle (+0.042) — the residual ~1/3
  of the gain comes from genuine covariation.

Biological gain over the depth01 baseline: depth04 +0.046, colshuffle +0.106,
depth08 +0.136, full15 +0.148. Depth saturates at ~8 sequences
(depth04→depth08→full15 plateau in both eff90 and supervised ρ). The
supervised gain is monotone in the *conservation/covariation content*, not in
sequence count past saturation.

## §3. Write-once dynamics (E7, recycle gating)

RF3 injects MSA → pair via `OuterProductMean` in the first 4 MSAModule blocks of
*each* recycle. Gating MSA to the first K recycles (`RF3_MSA_CYCLES`):

- k1 (MSA only in recycle 0): eff90 150, supervised 0.605 — installs ~75% of the
  geometric effect and ~85% of the functional gain.
- k2/k3/full15: saturate the remainder.
- 9 subsequent MSA-free recycles wash out only ~25%.

The family-consensus write is **near-irreversible**: the trunk cannot
self-correct it. This is the causal, representation-level confirmation of AF3's
"the MSA representation is not retained; all information passes through the pair
representation" — and shows the write happens early and sticks.

## §4. Fusion-matrix triad (validator protocol)

Dead-stream-controlled marginals on RBD bind_avg:

- FAESM marginal over Z_msa = x_msa_faesm − x_msa_zeros ≈ 0 (−0.005 on the
  archived v3 run) → **H2: MSA duplicates the ESM signal** (and sp_msa 0.644 ≫
  sp_faesm 0.484: MSA carries *more* than ESM here).
- Z_no marginal over (Z_msa+FAESM) = tri − tri_z ≈ 0 → **H1: MSA dominates the
  no-MSA structural signal**.
- No complementarity (H3 excluded on RBD bind_avg).

Replicates in direction on CASP3 and CI (weakest on CI, where FAESM ≈ Z_msa).
Gradient economy runs *opposite* to the information accounting. Late-training
encoder grad norms (tri cell, mean of 15 runs): RBD z/m/f = 0.022/0.053/0.143
(f/z≈6.5, f/m≈2.7, m/z≈2.4); CASP3 0.102/0.056/0.214 (f/m≈3.8, m/z≈0.6); CI
1.06/0.89/3.13 (f/m≈3.5, m/z≈0.8). FAESM's encoder dominates everywhere;
Z_msa's encoder exceeds Z_no's only on RBD — and only transiently: the RBD
trajectory shows m/z peak at ~5.8 around epoch 5 (Z_msa's signal is absorbed
first), with f/m crossing 1 near epoch 10 and widening thereafter. Absolute
norms are not comparable across proteins (CI labels are raw ProteinGym
DMS_score, ~50× RBD's gradient scale); compare ratios. (An earlier version of
this paragraph claimed "Z_msa's encoder dominates (m/z≈2, m/f≈5–8)" — the
ratios were mislabeled: m/f never exceeds 1.3 at any epoch; the 5–8 figure is
f/z.) Gradient dominance ≠ information (the audit's central lesson, reproduced
in the MSA context).

**Functional test of the gradient-interference account (AB-V3/V4, RBD full
set, `results/ab_validation/v3_esm_noise.json`, `v4_pace_fenc.json`).** If
FAESM's gradient dominance actively *suppresses* Z_msa's learning (rather than
merely being redundant), then rebalancing the gradients should recover part of
Z_msa's contribution. Two interventions on x_msa_faesm confirm this:

- *ESM-noise ladder* (nl ∈ {0,0.25,0.5,0.75,1.0}, fusion_as_validator paradigm,
  train+val noised alike): an **inverted-U** — val ρ 0.649→0.655→0.659→**0.662
  (peak nl=0.75, paired +0.013)**→0.648 (nl=1.0 back to baseline, paired
  −0.002). The late f/z gradient ratio collapses monotonically 2.91→0.04
  (z-encoder grad ↑8×, f-encoder ↓9×). At nl=1.0 (pure noise) the cell equals
  the x_msa_zeros dead-stream cell (0.648 ≈ 0.655): full noise merely removes
  the stream, so the benefit lives at the partial-noise sweet spot — it is
  gradient rebalancing, not "noise beats information". The nl=0.75 peak matches
  the inverted-U fusion_as_validator found on CASP3 (Z_no+FAESM).
- *f_enc LR pace* (λ ∈ {1.0,0.25,0.125,0.06}, content-cost-free analogue): a
  **plateau** — val ρ 0.649→**0.663 (λ=0.25, paired +0.014, 5/5 splits — the
  cleanest single arm)**→0.661→0.662; z-encoder grad ↑~4× (0.066→0.287).

Both recover a small, robust +0.013–0.014 over the fusion baseline. The effect
is modest (means exceed the 0.644 sp_msa ceiling but the paired gain is
marginal, not strongly significant), and there is no "more suppression is
better" monotonicity. Reading: on RBD FAESM is information-redundant (marginal
≈0, §4 triad) yet optimization-harmful — its encoder's gradient dominance
partially crowds out Z_msa's, and rebalancing recovers a slice of Z_msa's
contribution. This is the causal counterpart to the correlational
gradient-economy observation above.

**Per-channel cross-protein triad (dead-stream-corrected marginals):**

| protein | sp_no | sp_msa | FAESM over Z_msa (x_msa_faesm−x_msa_zeros) | Z_no over (Z_msa+FAESM) (tri−tri_z) |
|---|---|---|---|---|
| RBD | 0.465 | 0.644 | −0.005 ≈ 0 | +0.002 ≈ 0 |
| CASP3 | 0.394* | 0.556* | **+0.051** (5/5 splits) | **−0.001 ≈ 0** |
| CI | 0.374* | 0.449* | −0.039 (noisy, 2/5) | +0.008 ≈ 0 |

(RBD row from the archived v3 run; an independent A/B run gives sp_no
0.442/sp_msa 0.646, paired Δ+0.204, and dead controls 0.434/0.657.
*sp_no/sp_msa for CASP3/CI from e4 single-modality runs.) **Z_no's marginal over
(Z_msa+FAESM) ≈ 0 on all three proteins (H1 holds cross-protein).** FAESM's
marginal over Z_msa is protein-dependent: ≈0 on RBD, +0.051 on CASP3 (5/5
splits — MSA does not fully subsume ESM there), and −0.039 on CI, which is not
a clean ≈0: the mean is driven by one outlier split (per-split [−0.015, +0.077,
−0.235, +0.017, −0.038], 2/5 positive; CI split σ~0.09 at N=351) — consistent
with zero or a small negative. H2 (MSA≈ESM) holds on RBD, is partial on CASP3,
and is not cleanly resolved on CI.

## §5. Distillation (D1): partial reconstructibility

A (Z_no + FAESM) → Z_msa regression.

**Shallow ladder** (archived `d1_distill/results.json`; best-epoch val MSE;
3 splits × 2 inits; per-channel standardization):

| arch | val R² | val CKA |
|---|---|---|
| lin | 0.246 | 0.742 |
| mlp256 | 0.258 | 0.763 |
| mlp512 | 0.275 | 0.826 |
| tf1 | 0.343 | 0.812 |
| tf2 | 0.360 | 0.804 |

CKA plateaus at ~0.80–0.83 from mlp512 (tf1/tf2 do not exceed mlp512 under
best-epoch selection); R² keeps rising slowly. Single-modality ablation at
mlp512: zonly R² 0.267 / CKA 0.748 vs fonly R² 0.222 / CKA 0.752 — Z_no only
slightly ahead in R², essentially tied in CKA. **Retraction**: an earlier
version of this section reported lin 0.33/0.71 → MLP512 0.41/0.77 → TF-1L
0.53/0.84 → TF-2L 0.60/0.90 and "Z_no-alone ~3× > ESM-alone (R² 0.31 vs 0.11)".
Those numbers came from a superseded scalar-standardization run; none of them
appears in the archived data.

- Functional: reconstructed Z_msa predicts bind_avg 0.575 (Z_no 0.446 → real
  0.663) — recovers ~60% of the functional gap, matching the geometric fidelity.

**D1-deep** (depth ladder TF{2,3,4,6,8} × {both, fonly, zonly}; train-CKA vs val-CKA
memorization gap). Results (val_CKA / train-val gap):

| depth | both | fonly (ESM-only) | both gap |
|---|---|---|---|
| TF2 | 0.757 | 0.817 | +0.233 |
| TF4 | 0.803 | 0.874 | +0.190 |
| TF6 | 0.891 | 0.889 | +0.104 |
| TF8 | 0.903 | 0.895 | +0.093 |

Two findings: (1) **memorization concern partly allayed** — at TF8 the train-val
gap is only +0.09 (train 0.996 vs val 0.90), so the high CKA is mostly *genuine
generalization*, not pure memorization. (2) **But ESM alone suffices at depth** —
fonly *exceeds* both at shallow depth (0.817 vs 0.757 at TF2; adding Z_no hurts
via harder optimization/overfitting) and *matches* both at TF6+ (~0.90). So
**Z_no is redundant for reconstructing Z_msa at depth; ESM alone carries the
reconstructible content.** This corrects the shallow-depth D1 impression
twice over: the archived shallow data shows Z_no ≈ ESM at mlp512 (the "Z_no 3×
> ESM" figure was a superseded-run artifact, retracted above), and at depth
fonly ≥ both under naive concat. val_R² tells a slightly different story (both
0.414 > fonly 0.350 at TF8): Z_no adds variance-explained but no new subspace
direction at depth.

**D1-onehot (pure orthogonal one-hot control)**: each AA → a fixed mutually-orthogonal
128-dim vector (QR codebook, off-diag deviation 1e-4), so the input carries ONLY
per-residue amino-acid identity. val_CKA / gap by depth:

| depth | both | fonly | zonly | onehot |
|---|---|---|---|---|
| TF2 | 0.757 | 0.817 | 0.784 | 0.453 |
| TF4 | 0.803 | 0.874 | 0.777 | 0.373 |
| TF8 | 0.903 | 0.895 | 0.783 | 0.531 |
| gap@TF8 | +0.093 | +0.099 | +0.207 | +0.237 |

**one-hot is the worst at every depth (~0.5), non-monotonic, and its train-val gap
GROWS with depth (+0.15→+0.24) — pure memorization without generalization.** This
is the decisive negative control: pure sequence identity is genuinely insufficient
to reconstruct Z_msa, so the high fonly CKA (0.90) is *real evolutionary
information*, not trivial sequence→structure memorization. Information hierarchy
for reconstructing Z_msa: **ESM (evolution) 0.90 > Z_no (structure) 0.78 ≫ one-hot
(sequence identity) 0.5.**

**D1-XAttn-CKA (separate-encoder control)**: the naive-concat both used a shared
projection where the weak Z_no (128-d) is swamped by F (1280-d) — both≈F-concat-noise,
so both≤fonly there was partly an architecture artifact. With SEPARATE per-modality
encoders (XAttn-style) + CKA optimized directly as the loss (feature-space formula,
memory-safe), the hierarchy is stable across depth (nl2/4/6/8):

| depth | both | fonly | zonly | gaps (both/fonly/zonly) |
|---|---|---|---|---|
| 2 | 0.811 | 0.748 | 0.712 | +0.14/+0.17/+0.18 |
| 4 | 0.811 | 0.751 | 0.716 | +0.15/+0.16/+0.18 |
| 6 | 0.812 | 0.754 | 0.716 | +0.15/+0.16/+0.18 |
| 8 | 0.811 | 0.760 | 0.717 | +0.15/+0.15/+0.18 |

**both (0.811) > fonly (0.75–0.76) > zonly (0.716), stable across depth** — with a
proper architecture Z_no DOES contribute (+0.05–0.06 over fonly); ESM still leads
(+0.035 over zonly). All three plateau early (depth doesn't help under CKA loss,
unlike the MSE ladder) and the train-val gaps stay flat (no depth-driven
memorization, unlike naive-concat zonly). So the naive-concat both≤fonly
understated Z_no's contribution. (Absolute CKA differs from the MSE-trained ladder
— different objective/depth — so the clean read is the within-experiment hierarchy.) Refined conclusion: Z_msa is
reconstructible from (Z_no + ESM); ESM is the stronger source but Z_no carries a
real, architecture-revealed contribution; pure sequence identity (one-hot) is
insufficient.

## §6. Conformational-state selection (E2)

13 two-state targets, distogram NLL-preference (state B − state A; positive ⇒
state A preferred), shift = pref_msa − pref_nomsa. The crystallized state is
not always state A (Mad2: crystal = state B), so toward-crystal signs are
per-target:

- **Controls clean**: selecase (1-homolog MSA) and the three X-ray-vs-NMR pairs
  give |shift| ≤ 0.13 (selecase −0.006, ubiquitin −0.028, GB1 −0.008, TrxA
  +0.127) — no spurious crystal bias at this sensitivity; TrxA +0.127 is the
  largest control shift and sits inside the run-to-run noise band (below).
- **GPCR / kinases / transporter**: shift *toward* the crystallized state
  (+0.03 to +0.94; the lower end — Abl1 +0.03, β2AR/EGFR ~0.31 — is within
  ~1–1.5σ of the noise band) — MSA pulls toward the PDB-dominant functional
  state.
- **Fold-switchers split**:
  - KaiB, RfaH: MSA flips *away* from the crystal ground state toward the
    alternative (NMR) fold (shifts −2.6, −1.4).
  - Mad2, XCL1: MSA shifts strongly *toward* the crystal state (+2.3, +4.8).
  - GA88/GB88 (88%-identical designed pair): RF3 predicts each sequence into its
    *own* fold; MSA *sharpens* the sequence's intrinsic preference (GA88→GA88
    +0.38→+2.74; GB88→GB88 −0.23→−2.66), not redirects. (Scored by
    `e2_06_ga_gb.py`; the four values reproduce exactly from the archived
    distograms. The main `e2_results.json` E2-06 row is stale — msa=null.)

**Range-matched rerun (2026-08-02, `scripts/analysis/e2_rangematched_analyze.py`,
output `/mnt/k/output_heads/e2/e2_rangematched_results.json`).** Both states are
now scored on a *shared residue universe* per target (difflib alignment of each
state's npy sequence to the a3m row0; pair set = pairs finite in BOTH
experimental maps). This also fixed a second, unreported bug: RfaH stateA
(2OUG) lacks the disordered loop UniProt 101–114, so its npy rows are
compressed and the old contiguous-slice scoring misaligned rows ≥ 99 by 14
residues. Verified shared universes: XCL1 = UniProt 22–60 (39 res), RfaH =
115–156 (42 res, the only overlap). Outcomes:

- **Flagship fold-switcher verdicts survive**: XCL1 shift +4.81 → **+2.87**
  (TOWARD crystal, shrinks but stays strong); RfaH −1.42 → **−2.99**
  (away-from-crystal, *doubles* once the compression misalignment is fixed);
  KaiB −2.62 → −2.29; Mad2 −2.26 → −0.91 (TOWARD, crystal=B side). GA88/GB88
  is now scored in the main flow: shift +2.56 (old row was stale-null) —
  consistent with e2_06's "MSA sharpens, not redirects".
- **Three flips, all into the null zone**: EGFR +0.31 → −0.02, LacY +0.94 →
  **−0.08**, GB1 −0.01 → +0.09. The LacY collapse is the material one — the
  "transporter toward crystal" claim was mostly a residue-universe artifact.
  Category consequence: "GPCR/kinase/transporter toward crystal" weakens to
  β2AR (+0.15, still within ~1σ of the noise band) with Abl1/EGFR/LacY all
  ≈ 0. The paper's E2 weight should shift further onto the fold-switchers.
- **E49 template arms (5 targets, range-matched)**: template steering
  magnitudes shrink (e.g. KaiB steer_tpl −6.65 → −6.59, RfaH −2.09 → −1.19,
  Mad2 −2.65 → −1.70, LacY −2.59 → −0.47); XCL1 steer_tpl flips −1.28 →
  **+0.63**, resolving the old anti-template anomaly in the
  template-following direction. Conflict ≈ 0 throughout (msa_tplA ≡ tplA) —
  the E49 "template dominates MSA in conflict" conclusion is unchanged.

**Caveats (updated).** (1) Single forward per target, no seed replicates; repeat runs of
identical inputs (archived) give a preference noise band of ±0.14–0.27, so
shifts ≤ ~0.3 are not individually secure (post-matching: Abl1 +0.01, EGFR
−0.02, LacY −0.08, GB1 +0.09, TrxA +0.13, β2AR +0.15 are ALL inside or near
the band — the secure set is the fold-switchers + selecase/ubiquitin nulls).
(2) RESOLVED by the rerun above: XCL1/RfaH (and the milder KaiB/Mad2/LacY/Abl1
span mismatches) now rest on shared residue universes; the pre-matching
magnitudes are superseded. (3) Tally post-matching: 4 toward crystal
(XCL1/Mad2/β2AR/TrxA — only XCL1/Mad2 outside the noise band), 2 away
(KaiB/RfaH), GA88/GB88 sharpening (+2.56), the rest null.

**Refined claim**: the MSA is a *context-dependent state selector*. For
near-identical fold-switch pairs the sequence signal dominates and the MSA only
amplifies; for divergent cases the family consensus can redirect the predicted
state. This is richer than the original "MSA = crystal template" framing — the
selected state is the *family-consensus* state, which need not be the
crystallized one (KaiB/RfaH).

## §7. Methods audit: the standardization-convention bug

Three conventions coexisted: (1) fusion_v3/ci_bench per-position **per-channel**
(`std(axis=0)` vector, canonical); (2) run_fourway per-channel **global**
(`dim=(0,1)`); (3) rbd_repro/original e2 per-position **scalar**. A/B test:
scalar-per-position specifically degrades *fusion* cells (x_no_faesm
0.483→0.521, +0.037) while single-modality SP cells are robust (±0.02). The
"x_no_faesm < sp_faesm" anomaly was this artifact. All matrices re-run with the
canonical per-channel convention.

### §7b. Preprocessing-leakage boundary (strict vs overall standardization)

Distinct from the convention bug above: the archived matrices standardize
Y using per-position/per-channel mean-std fit on **all** samples, *then* split
— a preprocessing leakage (val/test statistics inform the transform). The
correct boundary is split-first: fit the statistics on training rows only and
freeze them onto val/test (`ab_std_split.py`). Two arms, paired by split×init:

- **overall** = the archived convention (fit on all). **strict** = train-only fit.
- Single-modality absolutes drop under strict, the loss scaling with reliance
  on Z_msa as the sole effective source: sp_no 0.465→0.443 (−0.022, 10/15),
  **sp_msa 0.644→0.589 (−0.055, 15/15, σ0.017)**. The RBD MSA effect shrinks
  from +0.179 (overall) to **+0.146 (strict)**, both 5/5.
- Full fusion triad under strict (90/90, `analyze_strict_fusion.py`): every
  cell drops 0.007–0.039 (x_msa_zeros most, x_no_zeros least). Dead-stream
  marginals: Z_msa-over-Z_no +0.197→**+0.165** (the only marginal moving more
  than ±0.02); FAESM-over-Z_no +0.075→+0.070; FAESM-over-Z_msa −0.005→+0.016
  (both in the ±0.02 noise layer, ≈0 either way); Z_no-over-(Z_msa+FAESM)
  +0.002→+0.003. **The triad ordering (H1/H2) is convention-robust; only the
  precise Z_msa-over-Z_no magnitude is convention-dependent.** Report
  single-modality absolutes under strict; the overall values are the optimistic
  bound. See `docs/overnight_report.md` §2.

---

## Synthesis for the manuscript

The MSA in an AF3-style trunk is best described as a **one-shot family-consensus
write** into the pair representation:

- *Mechanism*: `OuterProductMean` injects the family-mean covariation early; the
  write is near-irreversible (E7).
- *Content*: a shared-context component (capacity tax, no function) + per-column
  conservation (~2/3 of function) + pairwise covariation (~1/3) (E1).
- *Function*: it *helps* mutation-effect prediction by re-encoding mutation
  signals toward family evolutionary statistics — the signal cross-position
  generalization needs (E0/E4/E5). It subsumes both the no-MSA structural signal
  and the ESM signal on RBD (E2 triad).
- *Reconstructibility*: ~60% of its content/function is reconstructible from
  (Z_no + ESM); a nonlinear residual is MSA-specific (D1).
- *Conformation*: it selects a family-consensus conformational state,
  context-dependently (E2).

This reframes the original "MSA = crystal template that hurts prediction" into a
nuanced, evidence-backed account: the MSA is a family-consensus prior that is
*functionally beneficial* for DMS prediction, *partially redundant* with ESM, and
a *context-dependent* conformational selector — with the "crystal bias" surviving
only as the special case where the family consensus coincides with the
crystallized state.
