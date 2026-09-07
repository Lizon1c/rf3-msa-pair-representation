#!/usr/bin/env python3
"""E1: MSA dose x content matrix — Z_II extraction for RBD mutants under 8 MSA arms.

Self-contained (does NOT use attn_extract_unified.run_batch_zii_chain, which reads
the pre-2026-07-12 hook path). Reads RF3_ZII_FINAL_PATH (current RF3.py hook writes
both counter files and the final path; counter reset per batch bounds disk).

Arms (inputs/e1_msa_arms/*.a3m, row0=WT query, patched per chain per batch):
  full15 (sanity: must reproduce zii_wh1_msa geometry), colshuffle, consensus,
  depth01 (pipeline-level noMSA control), depth04, depth08, rowshuffle, random.
Subset: every 8th mutant of manifest_SARS_CoV_2_WH1.csv (~500, all sites) + WT.
Output: /mnt/k/output_heads/rbd/zii_e1_<arm>/mutant_XXXX_zii.pt  ([2,201,128] fp16,
  row+col at mutation site — same format as zii/ and zii_wh1_msa/rf3/).

Run: CUDA_VISIBLE_DEVICES=1 python -B launch_e1_msa_arms.py [arm1 arm2 ...]
"""
import os, sys, json, time, shutil, logging
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")  # 6000D (idle); must precede torch import
for p in ["/mnt/j/foundry-production/models/rf3/src", "/mnt/j/foundry-production/src"]:
    sys.path.insert(0, p)

import numpy as np
import torch

PROJ = Path("/mnt/j/conda_envs/foundry/DMS_Project")
ARMS_DIR = PROJ / "inputs" / "e1_msa_arms"
MANIFEST = PROJ / "data" / "sarbecovirus" / "manifest_SARS_CoV_2_WH1.csv"
OUT_BASE = Path("/mnt/k/output_heads/rbd")
STRIDE = int(os.environ.get("E1_STRIDE", "8"))  # 8 = ~500 subset; 1 = full 3998
BATCH = 4
ARM_ORDER = ["full15", "colshuffle", "consensus", "depth01",
             "depth04", "depth08", "rowshuffle", "random"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler("/tmp/e1_extract.log", mode="a")])
log = logging.getLogger("e1")


def load_manifest_subset():
    import csv
    rows = list(csv.DictReader(open(MANIFEST)))
    sub = rows[::STRIDE]
    wt_seq = None
    # WT sequence = row0 of any arm a3m (all share the same query)
    with open(ARMS_DIR / "full15.a3m") as f:
        f.readline()
        wt_seq = f.readline().strip()
    sub = [dict(mutant_id="WT", protein=wt_seq)] + sub
    return sub, wt_seq


def mut_pos(seq, wt):
    return next((j for j in range(min(len(seq), len(wt))) if seq[j] != wt[j]), -1)


def build_chain_batch_json(batch, gold_lines, work_dir):
    chain_ids = [chr(65 + i) for i in range(len(batch))]
    components = []
    for cid, m in zip(chain_ids, batch):
        lines = gold_lines.copy()
        lines[0] = f">{m['mutant_id']}\n"
        lines[1] = f"{m['protein']}\n"
        d = work_dir / cid
        d.mkdir(parents=True, exist_ok=True)
        msa_file = d / "t000_.msa0.a3m"
        with open(msa_file, "w") as f:
            f.writelines(lines)
        components.append({"chain_id": cid, "seq": m["protein"], "msa_path": str(msa_file)})
    spec = {"name": f"batch_{'_'.join(m['mutant_id'] for m in batch[:3])}",
            "components": components}
    jp = work_dir / "chain_batch.json"
    with open(jp, "w") as f:
        json.dump([spec], f, indent=2)
    return jp


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


def run_arm(engine, arm, subset, wt_seq):
    out_dir = OUT_BASE / f"zii_e1_{arm}"
    out_dir.mkdir(parents=True, exist_ok=True)
    gold_lines = open(ARMS_DIR / f"{arm}.a3m").readlines()
    todo = [m for m in subset if not (out_dir / f"{m['mutant_id']}_zii.pt").exists()]
    log.info(f"[{arm}] {len(subset) - len(todo)} done, {len(todo)} to go -> {out_dir}")
    max_batches = int(os.environ.get("E1_MAX_BATCHES", "0"))
    if max_batches:
        todo = todo[:max_batches * BATCH]
        log.info(f"[{arm}] PILOT MODE: capped at {max_batches} batches")
    n_done, t0 = 0, time.time()
    for b in range(0, len(todo), BATCH):
        batch = todo[b:b + BATCH]
        work = Path(f"/tmp/e1_{arm}_{b:04d}")
        work.mkdir(parents=True, exist_ok=True)
        try:
            jp = build_chain_batch_json(batch, gold_lines, work)
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
            lens = [len(m["protein"]) for m in batch]
            total = z.shape[-2]
            assert total == sum(lens), f"token mismatch {total} vs {sum(lens)}"
            bounds = [0]
            for sl in lens:
                bounds.append(bounds[-1] + sl)
            for i, m in enumerate(batch):
                blk = z[bounds[i]:bounds[i + 1], bounds[i]:bounds[i + 1], :]
                p = mut_pos(m["protein"], wt_seq)
                if m['mutant_id'] == 'WT':  # WT reference: save full block
                    torch.save(blk.half(), out_dir / f"{m['mutant_id']}_zii.pt")
                elif p < 0:  # WT-identical replicate row: slice at position 0 (old-pipeline convention)
                    rowcol = torch.stack([blk[0, :, :], blk[:, 0, :]]).half()
                    torch.save(rowcol, out_dir / f"{m['mutant_id']}_zii.pt")
                else:
                    rowcol = torch.stack([blk[p, :, :], blk[:, p, :]]).half()
                    torch.save(rowcol, out_dir / f"{m['mutant_id']}_zii.pt")
            n_done += len(batch)
            if n_done % 40 < BATCH:
                rate = n_done / max(time.time() - t0, 1e-6)
                log.info(f"[{arm}] {n_done}/{len(todo)} {rate:.2f} mut/s "
                         f"ETA {(len(todo) - n_done) / max(rate, 1e-6) / 60:.0f}min")
        except Exception as e:
            log.error(f"[{arm}] batch {b} FAILED: {e}", exc_info=True)
        finally:
            shutil.rmtree(work, ignore_errors=True)
            torch.cuda.empty_cache()
    log.info(f"[{arm}] DONE {n_done}/{len(todo)} in {(time.time() - t0) / 60:.0f}min")


def main():
    arms = sys.argv[1:] or ARM_ORDER
    subset, wt_seq = load_manifest_subset()
    log.info(f"subset: {len(subset)} entries (incl WT), arms: {arms}")
    engine = init_engine()
    for arm in arms:
        run_arm(engine, arm, subset, wt_seq)
    log.info("ALL ARMS DONE")


if __name__ == "__main__":
    main()
