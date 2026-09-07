# What the MSA writes into an AF3-style trunk's pair representation

Companion code, data, and manuscript for the paper:

> **What the MSA writes into an AF3-style trunk's pair representation, and why it helps mutation-effect prediction**

## Headline findings

- In RF3 (an AF3-style trunk), the MSA performs a **one-shot family-consensus write** into the pair
  representation: a single MSA-bearing recycle installs ~75% of the geometric and ~85% of the
  functional effect, and later MSA-free recycles cannot undo it.
- The write mildly compresses the pair representation (effective rank −16% to −42% across three
  proteins) yet **improves** cross-position mutation-effect (DMS) prediction
  (de-biased Δ Spearman ρ ≈ +0.14–0.15).
- Content decomposition over eight MSA arms: ~2/3 of the functional gain comes from per-column
  conservation/profile content, ~1/3 from genuine pairwise covariation; a shared-context/capacity
  tax (even random sequences collapse rank) buys nothing.
- Cross-protein, the gain tracks **how much the MSA rewrites** the representation (CKA to the
  no-MSA baseline), not raw MSA depth — family coherence, not depth, is the operative variable.
- ~60% of the functional advantage is reconstructible from (no-MSA pair + ESM-2) by distillation.

## Layout

- `paper/` — manuscript (`paper.tex`, compiled `paper.pdf`, figures) and `TRACEABILITY.md`
  (every quantitative claim mapped to its source file/section).
- `docs/` — detailed experiment narrative (`findings.md`), verified literature review
  (`lit_review.md`), and the experiment index of the underlying study.
- `scripts/` — extraction launchers, geometry/analysis scripts, supervised heads, distillation.
- `results/` — small JSON aggregates behind every figure and table.
- `figures/` — key figures in standalone form.

## Reproduction notes

Scripts are archived as-run: absolute paths point to the original environment
(`/mnt/j/...`, `/mnt/k/...`) and must be adapted. All experiments used RF3 pair representations
extracted per mutant; supervised evaluations are split-paired with a ±0.02 Spearman-ρ noise-floor
convention (see Methods of the paper and `docs/findings.md` §7 for the measurement audits).

## License

MIT (see LICENSE).
