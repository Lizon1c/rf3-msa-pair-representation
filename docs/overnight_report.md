# Overnight report — MSA-thread validation extensions (2026-07-30/31)

Status: **in progress**. Sections 1–3 are final; 4–7 fill in as runs complete.
Goal: validate two collaborator concerns (best-epoch val overfitting;
preprocessing-leakage boundary) and extend the MSA thread (ESM noise/pace on
Z_msa+FAESM fusion; D1-deep grokking dynamics), cross-protein.

---

## 1. Archive audit fixes (done, pre-overnight)

Issue-list items 2–8 from the msa_role_properties audit are fixed and
cross-checked (full-document reread + stale-number grep clean + 26 JSONs parse
+ all 9 figures regenerated):

- **D1 docs**: README/findings/index shallow-ladder numbers replaced with
  archived values (lin R²0.246/CKA0.742 → mlp512 0.275/0.826 → tf2
  0.360/0.804; zonly vs fonly R² 0.267 vs 0.222 ≈ tied). Old numbers
  (0.33→0.60, "3×") explicitly retracted as a superseded scalar-std run.
- **fig7 Mad2**: crystal-side determination added (`_crystal_side`); Mad2
  (crystal = state B) now sign-corrected (+2.26 toward crystal, red).
- **Double-run bookkeeping**: A/B run JSONs archived
  (`four_condition_A/B.json`, verified Δ+0.204, 5/5); README/findings cite
  both runs (+0.179 v3 / +0.204 A/B) with run tags.
- **e0 artifact JSON**: `_RETRACTED_ARTIFACT` warning key; index attribution
  corrected (corrected geometry comes from e1_analyze fullset).
- **E2 caveats**: noise band ±0.14–0.27, XCL1/RfaH non-comparable residue
  universes, tally 7 toward/2 away/3 null, GA88/GB88 provenance.
- **Two supervised protocols disclosed** (canonical per-channel full-set vs
  e1_supervised scalar-std subset); depth01 baseline correction documented.
- **Misc**: ab_std_check now writes JSON; origMSA_ref rows annotated invalid;
  gradient sentence corrected (FAESM encoder dominates, f/z≈5–8, not m/f);
  CI marginal −0.039 flagged as outlier-split-driven (2/5); build_msas.sh
  attribution; CASP3 count 1567; three more stale index values caught during
  verification (tri 0.650/0.648 etc.).

Item 1 (paper draft §2.1.1/§4.1 still carries retracted claims) intentionally
deferred — manuscript rewrite, separate task.

## 2. Preprocessing-leakage boundary (collaborator concern 2)

**Verdict: the collaborator is correct; effect is real, cell-specific, and
does not overturn conclusions.**

Setup: `ab_std_split.py` — arms `none/overall/strict/tglob` (labels raw;
strict = per-token per-channel stats fit on train only, frozen to val).
Note: per-token stats pool across mutants, and splits are by mutation site, so
overall stats DO pool val mutants into train-side preprocessing (the boundary
violation is real; it is X-statistics leakage, not label leakage).

### 2.1 Subset (500, 4 arms × 4 cells × 15 paired runs)

| cell | overall | strict | strict−overall |
|---|---|---|---|
| sp_no | 0.437 | 0.422 | −0.014 (7/15 neg, noise-level) |
| sp_msa | 0.625 | 0.569 | **−0.057 (13/15 neg)** |
| x_no_faesm | 0.438 | 0.438 | −0.0001 (±0.015) |
| x_msa_faesm | 0.580 | 0.581 | +0.0005 (±0.006) |

Fusion cells essentially unaffected; sp_msa inflated ~0.06 by the old
protocol. MSA effect (sp_msa−sp_no): overall +0.189 (5/5) → strict +0.146
(5/5).

### 2.2 Full set (N=3998, sp_no/sp_msa × overall/strict × 15)

Harness self-validated: overall arm reproduces archived v3 exactly (sp_no
0.4647 = v3; sp_msa 0.6434 ≈ v3 0.6437).

| cell | overall | strict | strict−overall |
|---|---|---|---|
| sp_no | 0.4647 | 0.4427 | −0.022 (10/15 neg, borderline) |
| sp_msa | 0.6434 | 0.5888 | **−0.055 (15/15 neg, σ=0.017)** |

MSA effect: overall +0.179 (5/5) → **strict +0.146 (5/5)**, per-split
[+0.134, +0.146, +0.169, +0.115, +0.166]. Subset predicted full set almost
exactly (−0.057 vs −0.055). Mechanism: under overall, val mutants (30%) help
define per-token μ/σ, contaminating the TRAIN inputs themselves
(0.7μ_tr+0.3μ_va centering); sp_msa affected 2.5× more than sp_no, consistent
with Z_msa's low-rank common structure making per-token stats more
cohort-sensitive.

### 2.3 Fusion cells under strict (full set) — DONE (90/90)

`bash-3o33p5p9` (GPU1): strict × {x_msa_faesm, x_msa_zeros, x_no_faesm,
x_no_zeros, tri, tri_z} × 15, paired against archived v3 overall (identical
split/init grid). `analyze_strict_fusion.py`.

Per-cell strict vs overall (paired, 15 runs):

| cell | overall | strict | Δ | neg splits |
|---|---|---|---|---|
| x_msa_zeros | 0.655 | 0.617 | **−0.039** | 14/15 |
| x_msa_faesm | 0.651 | 0.633 | −0.018 | 13/15 |
| tri_z | 0.648 | 0.631 | −0.017 | 13/15 |
| tri | 0.650 | 0.634 | −0.016 | 13/15 |
| x_no_faesm | 0.534 | 0.522 | −0.012 | 13/15 |
| x_no_zeros | 0.459 | 0.452 | −0.007 | 8/15 |

The strict drop scales with dependence on Z_msa as the single effective
source: x_msa_zeros (only Z_msa alive) drops most (−0.039); x_no_zeros (no MSA
stream at all) drops least (−0.007, noise-level). Confirms the subset read and
extends it to tri/tri_z.

Dead-stream-corrected marginals under each convention:

| marginal | overall | strict | Δ | verdict |
|---|---|---|---|---|
| Z_msa over Z_no | +0.197 | **+0.165** | −0.032 (13/15) | only marginal > ±0.02; direction/magnitude-class intact |
| FAESM over Z_no | +0.075 | +0.070 | −0.005 | large, robust |
| FAESM over Z_msa | −0.005 | +0.016 | +0.021 | sign flips but both in ±0.02 noise layer → ≈0 either way |
| Z_no over (Z_msa+FAESM) | +0.002 | +0.003 | +0.001 | flat, robust |

**Verdict**: the three structural conclusions are convention-robust — (H1)
Z_msa is the dominant information source (+0.165 strict over Z_no); FAESM adds
≈0 over a good Z_msa (+0.016, noise layer) but ~+0.07 over a useless Z_no;
Z_no adds nothing on top of Z_msa+FAESM (+0.003). The script's own gate ("all
marginal Δ within ±0.02 ⇒ archived overall stands") is breached by exactly one
marginal (Z_msa-over-Z_no, −0.032), so the honest statement is: triad ordering
is convention-robust; the only convention-dependent number is the precise
Z_msa-over-Z_no magnitude (strict +0.165 / overall +0.197). Single-modality
absolutes should be reported under strict.

## 3. Best-epoch selection / dual-val trajectories (collaborator concern 1)

**Verdict: confirmed and quantified on RBD; two independent de-biasing routes
converge on the same honest numbers. CASP3 replication running.**

### 3.1 RBD full set (`ab_bestepoch`, 60/20/20 cross-position, 15 runs/cell)

| cell | best_v1 | val2@same ep | best_val2 | final v1/v2 | sel_optim | traj_r |
|---|---|---|---|---|---|---|
| sp_no | 0.466 | 0.426 | 0.474 | 0.373/0.394 | +0.040 (4/5) | 0.12 |
| sp_msa | 0.661 | 0.567 | 0.617 | 0.613/0.552 | **+0.094 (5/5)** | 0.21 |
| sp_faesm | 0.495 | 0.446 | 0.484 | 0.468/0.458 | +0.049 | **0.69** |
| x_no_faesm | 0.546 | 0.494 | 0.543 | 0.503/0.497 | +0.052 (4/5) | 0.34 |
| x_msa_faesm | 0.654 | 0.603 | 0.635 | 0.599/0.593 | +0.052 (4/5) | 0.29 |

- Single-val best-epoch selection overestimates by +0.04–0.09 (sp_msa worst).
  Decomposition for sp_msa: +0.033 pure selection (difference-in-differences)
  + +0.061 val-set difficulty asymmetry (present even at final epoch).
- Trajectory agreement is low even at full scale (r 0.12–0.29; argmax epochs
  20–60 apart) — per-epoch ρ on 600 mutants (SE ~0.04) makes curve agreement
  a poor diagnostic. Exception: sp_faesm r=0.69 (ESM-based models track;
  consistent with E16/E23 "ESM halves split variance").
- Train ρ saturates at 0.95–0.99 early; val peaks ep ~20–38 then decays —
  best-epoch selection does real work (final worse), but the winning epoch is
  val-set-specific.

**Convergence (headline)**: MSA effect under four protocols, all 5/5 positive:
best_v1 +0.195 / val2@same-epoch **+0.141** / best_val2 **+0.143** / final
+0.240. The two de-biased test-set estimates agree at +0.14 — and the
strict-preprocessing estimate is +0.146. Two independent corrections converge:
**honest RBD MSA effect ≈ +0.14–0.15**; honest SP(Z_msa) ≈ 0.59 (strict
0.589 / val2@same 0.567 / best_val2 0.617), not 0.644. Fusion ≥ sp_msa under
honest eval (+0.018–0.036) — H1 picture intact.

### 3.2 val1/val2 late-gap mechanism (RBD, 5 cells)

Late gap (val2−val1, ep250–299): sp_no +0.021; **sp_msa −0.061 (5/5 splits,
locked by ep5 — feature-geometric, not training dynamics)**; sp_faesm −0.010;
x_no_faesm −0.006; x_msa_faesm −0.006. Per-split late-gap correlation matrix:
fusion cells track **sp_faesm's difficulty ordering (r=0.90/0.76), NOT
sp_msa's (r=0.37, lowest in matrix)**. Mechanism: fusion late readout is
FAESM-dominated (f/m≈2.7) and inherits FAESM's lack of systematic position-set
asymmetry; Z_msa's −0.06 offset appears only in the early gap (ep5–20) and is
diluted out as FAESM takes over. "Balance is inherited, not an attractor
property" — hypothesis (b) rejected (fusion would have its own ordering
otherwise).

### 3.3 CASP3 replication — DONE (75/75)

`bash-qft24h94` (GPU0), 5 cells × 15 runs, identical protocol.

**Pre-registered prediction (1) FALSIFIED**: CASP3 sp_msa late gap = −0.023
(2/5 splits negative; per-split [+0.002, +0.032, −0.161, +0.021, −0.009] — one
outlier split drives the mean), ep5 gap +0.004. There is NO RBD-style
ep5-locked, 5/5-negative universal offset, despite CASP3's STRONGER pinning
(−42% vs −26%). The pinning-driven hypothesis is falsified; RBD's −0.061
offset is RBD-specific (candidate: the 15-row shallow HARD-consensus MSA vs
CASP3's 69-row softer Swiss-Prot consensus — to test: RBD depth01/consensus
arms' val1/val2 gaps, or CASP3 with a 15-row-subsampled a3m).

**Stability ordering INVERTS between proteins**:

| cell | RBD late gap / traj_r | CASP3 late gap / traj_r |
|---|---|---|
| sp_no | +0.021 / 0.12 | +0.027 / 0.33 |
| sp_msa | −0.061 (5/5) / 0.21 | −0.023 (2/5) / **0.66** |
| sp_faesm | −0.010 / **0.69** | +0.040 / 0.32 |

On RBD, FAESM is the stable modality and Z_msa carries the offset; on CASP3,
Z_msa is the stable one and FAESM is noisy. Prediction (2) holds on the gap
(sp_faesm ≈0) but RBD's high FAESM trajectory agreement (0.69) is RBD-specific.

**Prediction (3) CONFIRMED in its generalized form.** CASP3 per-split late-gap
correlation matrix (final, 5 cells):

```
                 sp_faesm  sp_msa  sp_no  x_msa_faesm  x_no_faesm
sp_faesm            +1.00   +0.74  -0.19        +0.82       +0.96
sp_msa              +0.74   +1.00  +0.38        +0.90       +0.83
sp_no               -0.19   +0.38  +1.00        -0.02       -0.14
x_msa_faesm         +0.82   +0.90  -0.02        +1.00       +0.94
x_no_faesm          +0.96   +0.83  -0.14        +0.94       +1.00
```

On CASP3, x_msa_faesm tracks **sp_msa (+0.90) more than sp_faesm (+0.82)** —
the reverse of RBD (fusion vs sp_faesm 0.90/0.76, vs sp_msa 0.37). The fusion
readout does not inherit "FAESM specifically"; it inherits **whichever single
stream is dominant/stable on that protein** — FAESM on RBD (where Z_msa carries
the offset and FAESM is the stable modality), Z_msa on CASP3 (where Z_msa is
both stronger, 0.569, and the stable one, traj_r 0.66). sp_no is noise on
CASP3 (best_v1 0.371, negatively correlated with every other cell) — CASP3,
unlike RBD, cannot be carried by a bare sequence head and needs an embedding
stream.

Honest CASP3 MSA effect (sp_msa − sp_no): val2@same-epoch +0.183, best_val2
+0.161, final +0.142 — same magnitude class as RBD (+0.14–0.24), so the
headline MSA contribution is protein-robust even though the gap/stability
mechanism is protein-specific.

Figure: `figures/ab_v1_casp3_train_val12.png` (train/val1/val2 ρ trajectories,
5 cells).

## 4. ESM noise ladder on x_msa_faesm — DONE (75/75)

`ab_esm_noise.py` (fusion_as_validator paradigm: x'=(1−nl)x+nl·N(0,1)·σ_token,
fresh per batch, train+val noised at same nl; levels {0,0.25,0.5,0.75,1.0};
5 splits × 3 inits = 75 runs, 300 ep; late z/f encoder grad norms).
Archived `results/ab_validation/v3_esm_noise.json`; `analyze_ab_noise_pace.py`.

Pre-registration: fusion_as_validator found the inverted-U on CASP3 (Z_no+FAESM,
peak nl=0.75) but NOT on RBD. Discriminating question: does noised fusion beat
the fusion baseline / the sp_msa ceiling (0.644 overall / 0.589 strict)? If
yes, FAESM was actively interfering via gradient suppression, not redundant.

**Result — inverted-U, mechanism confirmed:**

| nl | n | mean | paired Δ vs nl=0 | per-split | z-grad | f-grad | f/z |
|---|---|---|---|---|---|---|---|
| 0.0 | 15 | 0.649 | — | — | 0.067 | 0.194 | 2.91 |
| 0.25 | 15 | 0.655 | +0.005 | 3/5 | 0.080 | 0.183 | 2.30 |
| 0.5 | 15 | 0.659 | +0.009 | 4/5 | 0.123 | 0.171 | 1.39 |
| **0.75** | 15 | **0.662** | **+0.013** | 3/5 | 0.278 | 0.137 | 0.49 |
| 1.0 | 15 | 0.648 | −0.002 | 3/5 | 0.527 | 0.022 | 0.04 |

- (A) beats fusion baseline: YES, peak nl=0.75 (+0.013). (C) inverted-U: YES —
  argmax interior (nl=0.75); nl=1.0 returns to baseline.
- Gradient rebalancing is textbook: f/z collapses 2.91→0.04 (z-enc ↑8×, f-enc
  ↓9×) — noising removes FAESM's gradient dominance and unleashes Z_msa.
- **"noise > information" excluded**: nl=1.0 (pure noise) = 0.648 ≈ x_msa_zeros
  dead-stream cell (0.655); full noise merely kills the stream (paired Δ≈0).
  The benefit is the partial-noise sweet spot, not noise per se.
- (B) the peak mean 0.662 exceeds the 0.644 ceiling, but σ~0.029 and the paired
  +0.013 is a marginal (3/5 splits) gain — a modest effect, not a clean break.
- Peak nl=0.75 matches fusion_as_validator's CASP3 inverted-U. NOTE: an early
  n=5 read of the nl=1.0 arm looked monotone (0.675); the full n=15 corrected it
  to the inverted-U above — the endpoint was an under-sampling artifact.

## 5. f_enc LR pace suppression on x_msa_faesm — DONE (60/60)

`ab_pace_fenc.py` (λ ∈ {1.0,0.25,0.125,0.06}; content-cost-free version of §4;
e30b/e32 showed pace reproduces most of the noise benefit). 5 splits × 3 inits
= 60 runs, 300 ep; late z/f grad norms. Archived
`results/ab_validation/v4_pace_fenc.json`; `analyze_ab_noise_pace.py`.

**Result — plateau, same direction:**

| λ | n | mean | paired Δ vs λ=1 | per-split | z-grad | f-grad |
|---|---|---|---|---|---|---|
| 1.0 | 15 | 0.649 | — | — | 0.066 | 0.192 |
| **0.25** | 15 | **0.663** | **+0.014** | **5/5** | 0.157 | 0.559 |
| 0.125 | 15 | 0.661 | +0.012 | 4/5 | 0.226 | 0.757 |
| 0.06 | 15 | 0.662 | +0.013 | 4/5 | 0.287 | 0.863 |

- argmax λ=0.25 (+0.014, **5/5 splits — the cleanest single arm across §4/§5**);
  λ≤0.25 is a plateau ~0.661–0.663. No "more suppression is better" monotonicity.
- z-encoder grad ↑~4× (0.066→0.287). (The f-encoder grad *norm* rises because a
  tiny-LR F encoder stays under-fit with large residual gradients; what changes
  is the optimization budget, not f's instantaneous norm — the f/z ratio here is
  not the rebalancing readout it is in §4.)
- Same reading as §4: rebalancing recovers a small, robust +0.013–0.014 of
  Z_msa's contribution that FAESM's gradient dominance was crowding out.

**§4/§5 joint verdict**: the correlational gradient-economy observation (FAESM
encoder dominates late gradients while its information marginal over Z_msa is
≈0) is now *causally* validated — two independent rebalancing interventions
(noise, pace) each recover ~+0.013, with the predicted gradient signature. On
RBD, FAESM is information-redundant but optimization-harmful. Effect is modest
and the noise interior is statistically soft (3/5); pace λ=0.25 (5/5) is the
result to carry forward.

## 6. D1-deep grokking probe — DEFERRED (not run)

`d1_deep_traj.py` ready and compile-checked (TF{2,4,8} × {both, fonly, zonly}
× 6 runs; per-epoch val R², train/val CKA every 10 ep; grokking diagnostic =
val_cka increments ep50→150→299 vs train saturation). Question: is the
TF2→TF8 train-val gap collapse (fig6: +0.233→+0.093) temporal grokking (val
CKA jumps late after train saturates) or a depth effect? **Not run** — deferred
by the user on 2026-07-31 (no further experiments this round). Output would be
`/mnt/k/output_heads/rbd/d1_distill/deep_traj.json`.

## 7. Autonomous follow-ups — done + proposed

**Done this round** (designed from earlier results):
- *CASP3 V1 replication* (§3.3) — built to test whether RBD's ep5-locked sp_msa
  val1>val2 offset is pinning-driven. Outcome: falsified; offset is RBD-specific
  and stability inverts.
- *Full-set strict fusion triad* (§2.3) — built to settle whether the subset
  convention-robustness held at scale. Outcome: triad ordering robust; only
  Z_msa-over-Z_no magnitude is convention-dependent.
- *ESM-noise (§4) and f_enc pace (§5) on x_msa_faesm* — built to causally test
  the gradient-interference account behind the fusion gradient economy.
  Outcome: both recover ~+0.013 with the predicted gradient signature.

**Proposed next** (from the §4/§5 results, not yet run):
- *CASP3 noise/pace replication.* On RBD FAESM's information marginal over
  Z_msa is ≈0, so it is purely optimization-harmful there. On CASP3 that
  marginal is +0.051 (5/5) — FAESM carries real signal. Prediction: the
  noise/pace gain should be weaker or absent on CASP3 (you cannot noise away a
  stream that is contributing information). A clean discriminator for the
  "redundant-but-harmful" reading.
- *nl=1.0 vs x_msa_zeros controlled comparison.* Already near-equal on RBD
  (0.648 ≈ 0.655); a paired same-split version would nail down that full noise
  ≙ removing the stream.
- *Finer λ grid around 0.25* (the cleanest arm, 5/5) if the pace effect is
  carried forward.

---

## Open questions for the user

1. sp_msa's −0.06 val1>val2 offset is RBD-specific (CASP3 falsified the
   pinning-driven hypothesis, §3.3). Candidate cause: RBD's 15-row shallow HARD
   consensus vs CASP3's 69-row soft Swiss-Prot consensus. Worth a dedicated
   geometric experiment (per-position-set variance/eff-rank of Z_msa vs Z_no)
   and/or RBD depth01/consensus arms' val1/val2 gaps?
2. Honest-number convention for the manuscript: report strict + test-set
   estimates as primary (SP(Z_msa) ~0.59, MSA effect +0.14–0.15), with
   best-val overall as the optimistic bound? (Recommended.)
3. Paper draft §2.1.1/§4.1 rewrite — schedule as separate task?
4. Gradient-interference effect (§4/§5) is a modest +0.013–0.014 and the noise
   interior is statistically soft (3/5); pace λ=0.25 (5/5) is the result to
   carry. Does it warrant the CASP3 replication in §7 before being written up,
   or is the RBD causal evidence sufficient for a methods note?
5. Run the deferred D1-deep grokking probe (§6) in a later round?
