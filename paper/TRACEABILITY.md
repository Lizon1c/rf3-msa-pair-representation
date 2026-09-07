# TRACEABILITY — m1_msa_write

Every quantitative claim in `paper.tex` mapped to its source. Paths relative to
`/mnt/j/conda_envs/foundry/DMS_Project/` (abbreviated `$P`). Primary sources:
`$P/msa_role_properties/docs/findings.md` (= **F**, section §n, line numbers of the
file as of 2026-09-07), `$P/msa_role_properties/README.md` (= **R**),
`$P/msa_role_properties/docs/overnight_report.md` (= **O**),
`$P/msa_role_properties/docs/experiment_index.md` (= **X**),
`$P/msa_role_properties/docs/lit_review.md` (= **L**),
`$P/AGENTS.md` ledger (= **A**). Convention per task rule: where sources disagree,
findings.md's corrected/honest values win; conflicts are logged at the bottom.

## Abstract / headline numbers
- eff-rank −16%/−42% (CI/CASP3), −26% RBD; CKA 0.527/0.553/0.917 → F §1 (lines 32–37), R §1 table (lines 30–34).
- Honest effect +0.14–0.15 (RBD), +0.14–0.18 (CASP3) → F §1 (lines 46–54, 66–68); O §2.2 (lines 69–80), O §3.1 (lines 149–155), O §3.3 (lines 217–219).
- Conservation ~2/3 / covariation ~1/3 → F §2 (lines 85–92); decomposition +0.095/+0.042.
- Saturation ~8 rows → F §2 (lines 94–98); A "E1 supervised dose curve DONE".
- 1 recycle ≈ 75% geometric / 85% functional; 9 MSA-free recycles wash out ~25% → F §3 (lines 100–113); A "E7 DONE".
- ~60% functional reconstructibility → F §5 (lines 212–213); X row D1-func (line 36).
- Gain tracks rewriting magnitude; family coherence not depth → F §1 (lines 39–44); A "Three-protein replication DONE".

## §2.1 three-protein geometry + function (Table 1)
- RBD eff90 231→171 (−26%), CKA 0.527, dCos 0.474, supp 1.21, N=3998 → F §1 (lines 17–21, 32–35); full-set geometry A "FULL-SET geometry DONE" (PC1 0.145, eff95 269).
- RBD Δ +0.179 (v3) / +0.204 (A/B), both 5/5, spread 0.025 ≈ cross-GPU noise floor → F §1 (lines 32–35), R §1 (lines 36–40), X E2-4cond-RBD (line 26).
- CASP3 69-row Swiss-Prot, eff90 333→193 (−42%), CKA 0.553, dCos 0.362, supp 1.393, Δ +0.162 (5/5, per-split +0.12~+0.21) → F §1 (line 36); A "CASP3 cross-protein replication DONE".
- CI 512-row UniRef30, eff90 61→51 (−16%), CKA 0.917, dCos 0.844, Δ +0.075 (4/5), N=351 → F §1 (line 37); A "Three-protein replication DONE"; CI N=351/59pos/237aa → A §2 (line 28).
- CASP3 1469 single mutants / 244 pos / 488 tokens → A §2 (line 26).
- RBD 201-aa construct, bind_avg, Starr et al. 2020 → A §2 (line 27); citation verified via PubMed 32841599 (Cell 182:1295–1310.e20).
- CASP3 fluorescence, Roychowdhury & Romero → A §2 (line 26, says "Roychowdhury 2020"); published version verified: Cell Death Discovery 8:7 (2022), doi from journal page (see Conflicts #2).
- CI labels = ProteinGym DMS_score → F §4 (lines 134–135), A §2 (line 28); Notin et al. 2023 NeurIPS D&B → L ref [21] (line 192).

## §2.2 honest effect size
- Selection optimism +0.04–0.09, sp_msa worst +0.094 (5/5); decomposition +0.033 selection + +0.061 val-set difficulty → O §3.1 (lines 128–140).
- Strict std: sp_msa 0.644→0.589 (−0.055, 15/15, σ0.017), sp_no 0.465→0.443 (−0.022) → F §7b (lines 372–375), O §2.2 (lines 69–72).
- De-biased estimates: val2@same-epoch +0.141, best_val2 +0.143, strict +0.146 → F §1 (lines 50–52), O §3.1 (lines 149–152).
- CASP3 honest +0.14–0.18 (val2@same +0.183, best_val2 +0.161, final +0.142) → O §3.3 (lines 217–219), F §1 (line 66).
- CASP3 dual-val replication falsifies pinning-driven offset (sp_msa late gap −0.023, 2/5, vs RBD −0.061 5/5); stability inverts → F §1 (lines 56–65), O §3.3 (lines 174–193).
- ±0.02 noise floor convention → R §0 header (line 18 "split-paired"), X caveats (line 57 area), O §2.3 gate (line 118); cross-process noise 0.025 → F §1 (lines 34–35), R §1 (lines 39–40).

## §2.3 E1 arms (Table 2, Fig. 2)
- eff90 arm values noMSA 181 / depth01 165 / depth04 157 / consensus 156 / rowshuffle ~155 / random 149 / colshuffle 150 / depth08 138 / full15 139 → F §2 (lines 78–88), A "E1 FULL 8-ARM TABLE" + "E7 DONE" (k-arms same analyzer); fig2 values match.
- Supervised ρ (500-subset protocol): 0.414/0.490/0.479/0.477/0.501/0.596/0.536/0.626/0.638 → A "E1 supervised dose curve DONE" (exact string), F §2 (lines 79–92).
- Gains over depth01: +0.046/+0.106/+0.136/+0.148; consensus +0.011 → F §2 (lines 94–96), R §2 (lines 64–69).
- dCos consensus 0.815 vs colshuffle 0.548 → F §2 (lines 86–89).
- noMSA→depth01 +0.076 = featurization-path effect → F §2 (lines 80–84), A "Baseline correction (2026-07-25)".
- Two protocols not interchangeable (E1: rbd_repro V2 ZII_Model head, scalar std, 500-subset; canonical: SinglePredictorV2, per-channel, full set) → R "Reproduction" (lines 228–242), F §2 (lines 72–76).

## §2.4 E7 write-once (Table 3)
- k1 eff90 150 / ρ 0.605; k2 145 / 0.628; k3 144 / 0.609; full15 139 / 0.638; CKA_no k1 0.610→full15 0.568 → F §3 (lines 100–113), A "E7 DONE (geometry)" + "E7 supervised DONE".
- Shares: geometric (181−150)/(181−139)=74%, (181−145)/42=86%, (181−144)/42=88%; functional (0.605−0.414)/(0.638−0.414)=85%, k2 96%, k3 87% — arithmetic computed from the table values above.
- k1 ≈ colshuffle (both eff90 150) triangulation → A "E7 DONE" ("k1 ≈ colshuffle (both 150) triangulates profile+one-shot-covariation").
- RF3 injects MSA→pair via OuterProductMean in first 4 MSAModule blocks per recycle; gating via RF3_MSA_CYCLES → F §3 (lines 102–103), X E7-extract (line 10).

## §2.5 fusion triad (Table 4, Fig. 4) + gradient rebalancing
- RBD v3 cells sp_no 0.465 / sp_msa 0.644 / sp_faesm 0.484 / x_msa_faesm 0.651 / tri 0.650 / tri_z 0.648 / x_msa_zeros 0.655 / x_no_zeros 0.459 → R §4 table (lines 86–92), F §4 (lines 116–124).
- Marginals: FAESM|Z_msa −0.005; Z_no|(Z_msa+FAESM) +0.002; FAESM|Z_no +0.075; Z_msa|Z_no +0.197 (overall) / +0.165 (strict) → R §4 (lines 94–100), O §2.3 (lines 104–121).
- CASP3 row (0.394/0.556/0.584, +0.051 5/5, −0.001) and CI row (0.374/0.449/0.452, −0.039 noisy 2/5 one outlier split, +0.008) → F §4 cross-protein table (lines 171–186), R §4 (lines 104–117). sp_no/sp_msa for CASP3/CI from e4 single-modality runs (flagged in F §4 footnote, line 179).
- Strict-standardization fusion drops 0.007–0.039; marginals convention-robust → F §7b (lines 376–384), O §2.3 (lines 88–121).
- Gradient economy ratios: RBD z/m/f = 0.022/0.053/0.143 (f/z≈6.5, f/m≈2.7, m/z≈2.4); CASP3 0.102/0.056/0.214; CI 1.06/0.89/3.13; m/z peak ~5.8 at ~epoch 5 → F §4 (lines 127–139). (Uses the *corrected* ratios; see Conflicts #4.)
- AB-V3 noise ladder: 0.649→0.655→0.659→0.662 (nl=0.75, paired +0.013, 3/5)→0.648 (nl=1.0, −0.002); f/z 2.91→0.04 (z-enc ↑8×, f-enc ↓9×); nl=1.0 ≈ x_msa_zeros (0.648 vs 0.655) → F §4 (lines 141–155), O §4 (lines 239–258), X AB-V3 (line 62).
- AB-V4 pace: 0.649→0.663 (λ=0.25, +0.014, 5/5)→0.661→0.662; z-grad 0.066→0.287 → F §4 (lines 156–167), O §5 (lines 267–283), X AB-V4 (line 63).
- Fusion architecture = CrossAttnOrthoConcatFusion with 0.05|cos| orthogonality penalty; dead-stream zeros controls → A §3 (lines 36–40), X design notes.

## §2.6 D1 distillation (Fig. 5, Fig. 6)
- Shallow ladder lin 0.246/0.742, mlp256 0.258/0.763, mlp512 0.275/0.826, tf1 0.343/0.812, tf2 0.360/0.804; CKA plateaus ~0.80–0.83 from mlp512 → F §5 (lines 192–204), X D1 (line 35). Retraction of old scalar-std numbers (0.33→0.60, "3×") honored — those values NOT used → F §5 (lines 205–210), X caveat 4 (line 70).
- Functional: reconstructed 0.575 vs Z_no 0.446, real 0.663 → ~60% ((0.575−0.446)/(0.663−0.446)=0.594) → F §5 (lines 212–213), A "D1 functional equivalence DONE".
- D1-deep: TF8 both 0.903 / fonly 0.895 / zonly 0.783; train-val gap +0.093 at TF8 → F §5 (lines 215–236), X D1-deep (line 37).
- D1-onehot: ~0.5 at every depth (0.453/0.373/0.531), non-monotonic, gap grows +0.15→+0.24 → F §5 (lines 238–255).
- D1-XAttn-CKA: both 0.811 > fonly 0.75–0.76 > zonly 0.716–0.717, stable across depth 2/4/6/8 → F §5 (lines 257–279).
- 3 splits × 2 inits, per-channel std, best-epoch val → F §5 (lines 192–193).

## §2.7 cross-protein law
- Gain tracks rewriting magnitude (CKA↓ ⇒ Δ↑); family coherence not depth; refinement of AF2 ~30-threshold (family identification) → F §1 (lines 39–44), A "Three-protein replication DONE", R §1 (lines 50–53).

## Discussion / Intro literature claims (all from L citation map §6/§7 unless noted)
- AF2 Evoformer, masked-MSA loss, ~30/~100 depth threshold, "coarsely find the correct structure", PDB-most-likely objective → L §1 + map rows 1–5 (Jumper 2021 [1]).
- AF3 demotion, 4 blocks, pair-weighted averaging, "MSA representation is not retained…", AF3 retains MSA-depth dependence → L §2 + map rows 6–10 (Abramson 2024 [2]).
- ESMFold MSA removal, 60×, "language models are learning information similar to the contents of MSAs" → L §2 + rows 11–14 (Lin 2022 [3]); ESM-2 model id esm2_t33_650M_UR50D → A §2 (line 31).
- ESM3 hard-target MSA advantage (Table S9) → L §2 + rows 15–17 (Hayes 2024 [4]).
- RF3 implements AF3; only ablation is HHBlits-vs-MMseqs2; no MSA ablation → L §2 + rows 18–19 (Corley 2025 [5]).
- RFdiffusion never uses MSA → L §2 + row 20 (Watson 2023 [6]).
- MSA-perturbation family → L §3.1 rows 22–26 (del Alamo 2022, Stein & Mchaourab 2022, Wayment-Steele 2024, Kalakoti & Wallner 2025).
- GPCR/state bias; consensus attribution → L §3.2 rows 29–31 (Saldaño 2022, Sala 2023, Heo & Feig 2022).
- Fold-switcher one-of-two / memorization → L §3.3 rows 27–28 (Chakravarty & Porter 2022, Chakravarty 2024).
- ProteinGym depth stratification; MSA Pairformer scaling trade-off; Tranception robustness framing → L §4 rows 33–34 + note 9 (Notin 2023, Akiyama 2025, Notin 2022).
- ColabFold "paired alignment often not needed" → L §3.1 row 32 + note 12 (Mirdita 2022).
- OpenFold MSA–structure lineage claim (intro, coevolution without supervision) → L §1 (line 21, Ahdritz 2024 [24]).
- AF1 = Senior et al. 2020 → L "Mislabeled PDF" note (line 15) + ref [8].
- CKA method → Kornblith et al. 2019, ICML PMLR 97:3519–3529; verified via proceedings.mlr.press/v97/kornblith19a (web, 2026-09-07). NOT from L (added per task rule with verification).
- Starr et al. 2020 Cell 182:1295–1310.e20 → verified PubMed 32841599 (2026-09-07).
- Roychowdhury & Romero, Cell Death Discovery 8:7 (2022) → verified nature.com/articles/s41420-021-00799-0 (2026-09-07).

## Methods
- Extraction hook (RF3_ZII_PATH after distogram head; hooks/monkey-patching don't work on compiled submodules) → A §3 (lines 44–45), R "Reproduction" (lines 219–221).
- RBD slice = true-site row+col [2,L,128]; CASP3 full pair [488,488,128] sliced downstream; index offset RBD site−1 → A §2 (lines 29–30).
- MSAs: RBD GOLD 15 full-length rows (18 length-filtered); CASP3 casp3_human.a3m 69 Swiss-Prot rows; CI ci_lambda.a3m 512 UniRef30 rows; CASP3/CI built from local mmseqs2 DBs (build scripts not archived — provenance gap, disclosed) → X MSA-build (line 8), R "Reproduction" (lines 221–226).
- Canonical protocol details (70/30 cross-position, seeds 100–104, inits 7/107/207, 300 ep, AdamW 1e-4/0.05, cosine, MSE, best-epoch val Spearman) → R "Reproduction" (lines 231–235), A §4 (lines 52–53).
- E1 protocol (rbd_repro V2 ZII_Model, scalar std, 500-mutant stride-8; convention moves single-modality cells ≤0.02) → R (lines 236–241), F §2 (lines 72–76).
- Strict vs overall standardization; dual-val 60/20/20 → F §7b, O §2/§3.
- Noise paradigm (x'=(1−nl)x+nl·N(0,1)·σ_token, train+val noised alike) → O §4 (lines 226–229).

## Conflicts / internal inconsistencies found (and resolution used)
1. **RBD geometry, subset vs full set**: L §5.2 reports eff90 181→139 (−23%), CKA 0.568, dCos ~0.48 (500-subset); F §1/R §1 report full-set 231→171 (−26%), CKA 0.527, dCos 0.474. Not a contradiction (subset vs N=3998) — paper uses **full-set** values for the three-protein table and subset values only inside the E1/E7 arm tables (500-subset protocol), with protocols labeled.
2. **CASP3 dataset year**: A §2 says "Roychowdhury 2020"; the published article is Roychowdhury & Romero, *Cell Death Discovery* 8:7 (**2022**) (2020 ≈ preprint year). Paper cites the 2022 published version (web-verified).
3. **RBD sp_msa absolute level**: 0.638 (E1 subset protocol, full15) vs 0.644 (canonical best-val v3) vs 0.646 (A/B run) vs 0.589 (strict) vs 0.663 (D1 functional run's "real Z_msa"). These are four different protocols/runs; the paper keeps them segregated (arm tables within-protocol; Table 4 best-val labeled optimistic; honest 0.589 reported in §2.2; D1 numbers quoted only within the D1 paragraph).
4. **Gradient-ratio mislabeling (corrected upstream)**: an earlier F §4 version claimed "Z_msa encoder dominates (m/z≈2, m/f≈5–8)"; the corrected text (f/z≈6.5, f/m≈2.7, m/z≈2.4, m/f never >1.3) is what the paper reports → F §4 (lines 127–139, correction noted inline at lines 136–138).
5. **D1 shallow ladder**: superseded scalar-std numbers (lin 0.33/0.71 → tf2 0.60/0.90, "Z_no 3× > ESM") retracted in F §5; paper uses archived per-channel values only.
6. **CI FAESM marginal −0.039**: driven by one outlier split (2/5 positive; per-split list in F §4 lines 183–184) — reported as "unresolved", not as ≈0 and not as a real negative.
7. **O §3.1 table sp_msa best_v1 0.661** vs archived v3 0.644: the dual-val run is a 60/20/20 split (smaller train), so absolutes differ from the 70/30 matrices; only the paired Δ (+0.141/+0.143) is carried into the paper.
8. **E2 conformational-state results (F §6)**: deliberately NOT included in this manuscript (different story; per task scope, fig6/7/8 excluded).
