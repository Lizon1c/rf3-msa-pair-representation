#!/usr/bin/env python3
"""CI (lambda repressor) clean Z_II extraction, no-MSA and with-MSA conditions.

Replaces the slice@0-artifacted ci/zii/ (ci_bench.py saved z[0,:L,:] = row at
position 0 for every mutant). Saves row+col at the TRUE mutation site:
[2, 237, 128] fp16, named ci_{i:03d}_zii.pt (CSV row index, matching convention).

Modes:
  nomsa: chains carry seq only (true seq-only) -> ci/zii_clean/
  msa:   chains carry ci lambda-repressor a3m with row0 patched -> ci/zii_msa/
Run: CUDA_VISIBLE_DEVICES=0 python -B launch_ci_msa.py nomsa|msa
"""
import os, sys, json, re, csv, time, shutil, logging
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
for p in ["/mnt/j/foundry-production/models/rf3/src", "/mnt/j/foundry-production/src"]:
    sys.path.insert(0, p)

import torch

PROJ = Path("/mnt/j/conda_envs/foundry/DMS_Project")
A3M = PROJ / "inputs" / "GOLD_MSA" / "ci_lambda.a3m"
CSV_IN = PROJ / "data" / "DMS_ProteinGym_substitutions" / "RPC1_LAMBD_Li_2019_high-expression.csv"
BATCH = 4

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler("/tmp/ci_extract.log", mode="a")])
log = logging.getLogger("ci")


def init_engine():
    from omegaconf import OmegaConf
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from hydra.utils import instantiate
    config_dir = "/mnt/j/foundry-production/models/rf3/configs"
    GlobalHydra.instance().clear()
    overrides = ["num_steps=50", "diffusion_batch_size=2", "n_recycles=10",
                 "ckpt_path=/mnt/j/foundry_checkpoints/rf3_foundry_01_24_latest_remapped.ckpt"]
    with initialize_config_dir(config_dir=config_dir, version_base="1.3"):
        cfg = compose(config_name="inference", overrides=overrides)
    run_keys = {"inputs", "out_dir", "dump_predictions", "dump_trajectories",
                "one_model_per_file", "annotate_b_factor_with_plddt",
                "sharding_pattern", "skip_existing", "template_selection",
                "ground_truth_conformer_selection", "cyclic_chains", "add_missing_atoms"}
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    init_cfg = OmegaConf.create({k: v for k, v in cfg_dict.items() if k not in run_keys})
    engine = instantiate(init_cfg, _convert_="partial", _recursive_=False)
    engine.initialize()
    log.info("RF3 engine ready.")
    return engine


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "nomsa"
    assert mode in ("nomsa", "msa")
    out_dir = Path(f"/mnt/k/output_heads/ci/zii_{'clean' if mode == 'nomsa' else 'msa'}")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open(CSV_IN)))
    WT = None
    for r in csv.DictReader(open(PROJ / "DMS_substitutions.csv")):
        if 'RPC1_LAMBD' in r['DMS_id']:
            WT = r['target_seq']
            break
    assert WT and len(WT) == 237, f'WT len {len(WT) if WT else None}'

    def mk(mut):
        m = re.match(r'([A-Z])(\d+)([A-Z])', mut)
        p = int(m.group(2))
        s = list(WT)
        s[p - 1] = m.group(3)
        return ''.join(s), p - 1

    gold_lines = open(A3M).readlines() if mode == "msa" else None
    engine = init_engine()

    todo = []
    for i, r in enumerate(rows):
        if (out_dir / f'ci_{i:03d}_zii.pt').exists():
            continue
        seq, p = mk(r['mutant'])
        todo.append({'idx': i, 'mutant': r['mutant'], 'seq': seq, 'pos': p})
    log.info(f"mode={mode}, {len(todo)} mutants to extract -> {out_dir}")

    n_done, t0 = 0, time.time()
    for b in range(0, len(todo), BATCH):
        batch = todo[b:b + BATCH]
        work = Path(f"/tmp/ci_{mode}_{b:04d}")
        work.mkdir(parents=True, exist_ok=True)
        try:
            comps = []
            for cid, m in zip([chr(65 + i) for i in range(len(batch))], batch):
                comp = {"chain_id": cid, "seq": m['seq']}
                if mode == "msa":
                    lines = gold_lines.copy()
                    lines[0] = f">ci_{m['idx']:03d}\n"
                    lines[1] = f"{m['seq']}\n"
                    d = work / cid
                    d.mkdir(exist_ok=True)
                    mf = d / "t000_.msa0.a3m"
                    with open(mf, "w") as f:
                        f.writelines(lines)
                    comp["msa_path"] = str(mf)
                comps.append(comp)
            jp = work / "batch.json"
            with open(jp, "w") as f:
                json.dump([{"name": f"batch_{b}", "components": comps}], f, indent=2)
            os.environ["RF3_ZII_COUNTER"] = "0"
            os.environ["RF3_ZII_PATH"] = str(work / "zii.pt")
            os.environ["RF3_ZII_FINAL_PATH"] = str(work / "zii_final.pt")
            engine.run(inputs=str(jp), out_dir=str(work / "out"),
                       dump_predictions=False, dump_trajectories=False,
                       one_model_per_file=False, skip_existing=False,
                       template_selection=[])
            zf = work / "zii_final.pt"
            if not zf.exists():
                raise RuntimeError("no zii_final.pt")
            z = torch.load(zf, map_location="cpu", weights_only=True)
            lens = [len(m['seq']) for m in batch]
            assert z.shape[-2] == sum(lens), f"token mismatch {z.shape[-2]} vs {sum(lens)}"
            bounds = [0]
            for sl in lens:
                bounds.append(bounds[-1] + sl)
            for i, m in enumerate(batch):
                blk = z[bounds[i]:bounds[i + 1], bounds[i]:bounds[i + 1], :]
                p = m['pos']
                torch.save(torch.stack([blk[p, :, :], blk[:, p, :]]).half(),
                           out_dir / f"ci_{m['idx']:03d}_zii.pt")
            n_done += len(batch)
            if n_done % 40 < BATCH:
                rate = n_done / max(time.time() - t0, 1e-6)
                log.info(f"{n_done}/{len(todo)} {rate:.2f} mut/s "
                         f"ETA {(len(todo) - n_done) / max(rate, 1e-6) / 60:.0f}min")
        except Exception as e:
            log.error(f"batch {b} FAILED: {e}", exc_info=True)
        finally:
            shutil.rmtree(work, ignore_errors=True)
            torch.cuda.empty_cache()
    log.info(f"DONE {n_done}/{len(todo)} in {(time.time() - t0) / 60:.0f}min")


if __name__ == "__main__":
    main()
