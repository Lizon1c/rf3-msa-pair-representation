#!/usr/bin/env python3
"""E7: recycle-restricted MSA — can the trunk iterate away the MSA pinning?

Arms (all with the real 15-row GOLD MSA = full15.a3m):
  k1/k2/k3: RF3_MSA_CYCLES=1/2/3 — MSA fed only in the first K recycles,
            recycles K..9 run MSA-free (query-row replacement at RF3.py:268).
            Full 500-mutant subset; readout = final Z_II ensemble geometry
            (eff-rank/PC1/CKA vs noMSA & vs full15 via e7_analyze.py).
  traj_k0 : reference trajectory, depth01.a3m, no gating, batch 0 only.
  traj_k10: reference trajectory, full15.a3m, no gating, batch 0 only.
For batch 0 of every arm, per-recycle Z_II dumps are captured
(RF3_RECYCLE_DUMP_DIR) and the WT chain trajectory is saved:
  WT_recycle_traj.pt [n_recycles, 201, 201, 128] fp16 + trajectory.json
  (per-recycle cos-to-final + participation ratio, per chain).

Known caveat: the static profile feature (full-MSA column frequencies in
S_inputs/Z_init) is NOT gated; E1 colshuffle arm bounds its contribution.

Run: CUDA_VISIBLE_DEVICES=0 python -B launch_e7_recycle_gate.py [k1 k2 k3 ...]
"""
import os, sys, json, time, shutil, logging
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # 5090D (user directive); precede torch
for p in ["/mnt/j/foundry-production/models/rf3/src", "/mnt/j/foundry-production/src"]:
    sys.path.insert(0, p)

import torch

PROJ = Path("/mnt/j/conda_envs/foundry/DMS_Project")
ARMS_DIR = PROJ / "inputs" / "e1_msa_arms"
MANIFEST = PROJ / "data" / "sarbecovirus" / "manifest_SARS_CoV_2_WH1.csv"
OUT_BASE = Path("/mnt/k/output_heads/rbd")
STRIDE = 8
BATCH = 4
# arm -> (msa_a3m, RF3_MSA_CYCLES or None, max_batches or None)
ARM_CFG = {
    "k1":       ("full15.a3m", 1,    None),
    "k2":       ("full15.a3m", 2,    None),
    "k3":       ("full15.a3m", 3,    None),
    "traj_k0":  ("depth01.a3m", None, 1),
    "traj_k10": ("full15.a3m", None, 1),
}
ARM_ORDER = ["k1", "k2", "k3", "traj_k0", "traj_k10"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler("/tmp/e7_extract.log", mode="a")])
log = logging.getLogger("e7")


def load_manifest_subset():
    import csv
    rows = list(csv.DictReader(open(MANIFEST)))
    sub = rows[::STRIDE]
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


def part_ratio(z):
    """Participation ratio of singular values of flattened pair rep (per-structure)."""
    X = z.reshape(z.shape[0], -1).float()
    sv = torch.linalg.svdvals(X)
    s2 = (sv ** 2)
    return float((s2.sum() ** 2) / (sv ** 4).sum().clamp_min(1e-20))


def capture_trajectory(dump_dir, lens, out_dir):
    """Slice per-chain diag blocks from per-recycle dumps; save WT traj + metrics."""
    files = sorted(Path(dump_dir).glob("zii_recycle_*.pt"))
    if not files:
        log.warning("no recycle dumps found")
        return
    bounds = [0]
    for sl in lens:
        bounds.append(bounds[-1] + sl)
    n_chains = len(lens)
    per_chain = [[] for _ in range(n_chains)]
    for f in files:
        z = torch.load(f, map_location="cpu", weights_only=True)
        for c in range(n_chains):
            blk = z[bounds[c]:bounds[c + 1], bounds[c]:bounds[c + 1], :].float()
            per_chain[c].append(blk)
    cos = torch.nn.functional.cosine_similarity
    metrics = {}
    for c in range(n_chains):
        traj = torch.stack(per_chain[c])  # [R, L, L, 128]
        final = traj[-1].reshape(-1)
        metrics[f"chain{c}"] = {
            "cos_to_final": [float(cos(traj[r].reshape(-1), final, dim=0)) for r in range(len(traj))],
            "part_ratio": [part_ratio(traj[r]) for r in range(len(traj))],
        }
    # save WT (chain 0) trajectory tensor
    torch.save(torch.stack(per_chain[0]).half(), out_dir / "WT_recycle_traj.pt")
    with open(out_dir / "trajectory.json", "w") as f:
        json.dump(metrics, f, indent=1)
    log.info(f"trajectory captured: {len(files)} recycles, {n_chains} chains")


def run_arm(engine, arm, subset, wt_seq):
    a3m_name, cycles, max_batches = ARM_CFG[arm]
    out_dir = OUT_BASE / f"zii_e7_{arm}"
    out_dir.mkdir(parents=True, exist_ok=True)
    gold_lines = open(ARMS_DIR / a3m_name).readlines()
    todo = [m for m in subset if not (out_dir / f"{m['mutant_id']}_zii.pt").exists()]
    if max_batches:
        todo = todo[:max_batches * BATCH]
    cap = int(os.environ.get("E7_MAX_BATCHES", "0"))
    if cap:
        todo = todo[:cap * BATCH]
        log.info(f"[{arm}] PILOT MODE: capped at {cap} batches")
    if cycles is None:
        os.environ.pop("RF3_MSA_CYCLES", None)
    else:
        os.environ["RF3_MSA_CYCLES"] = str(cycles)
    log.info(f"[{arm}] a3m={a3m_name} RF3_MSA_CYCLES={cycles} todo={len(todo)}")
    n_done, t0 = 0, time.time()
    for b in range(0, len(todo), BATCH):
        batch = todo[b:b + BATCH]
        work = Path(f"/tmp/e7_{arm}_{b:04d}")
        work.mkdir(parents=True, exist_ok=True)
        dump_dir = work / "recycle_dump"
        is_first_batch = (n_done == 0)
        try:
            jp = build_chain_batch_json(batch, gold_lines, work)
            os.environ["RF3_ZII_COUNTER"] = "0"
            os.environ["RF3_ZII_PATH"] = str(work / "zii.pt")
            os.environ["RF3_ZII_FINAL_PATH"] = str(work / "zii_final.pt")
            if is_first_batch:
                dump_dir.mkdir(exist_ok=True)
                os.environ["RF3_RECYCLE_DUMP_DIR"] = str(dump_dir)
            else:
                os.environ.pop("RF3_RECYCLE_DUMP_DIR", None)
            engine.run(inputs=str(jp), out_dir=str(work / "out"),
                       dump_predictions=False, dump_trajectories=False,
                       one_model_per_file=False, skip_existing=False,
                       template_selection=[])
            zf = work / "zii_final.pt"
            if not zf.exists():
                raise RuntimeError("no zii_final.pt")
            z = torch.load(zf, map_location="cpu", weights_only=True)
            lens = [len(m["protein"]) for m in batch]
            assert z.shape[-2] == sum(lens), f"token mismatch {z.shape[-2]} vs {sum(lens)}"
            bounds = [0]
            for sl in lens:
                bounds.append(bounds[-1] + sl)
            for i, m in enumerate(batch):
                blk = z[bounds[i]:bounds[i + 1], bounds[i]:bounds[i + 1], :]
                p = mut_pos(m["protein"], wt_seq)
                if m['mutant_id'] == 'WT':
                    torch.save(blk.half(), out_dir / f"{m['mutant_id']}_zii.pt")
                elif p < 0:
                    torch.save(torch.stack([blk[0, :, :], blk[:, 0, :]]).half(),
                               out_dir / f"{m['mutant_id']}_zii.pt")
                else:
                    torch.save(torch.stack([blk[p, :, :], blk[:, p, :]]).half(),
                               out_dir / f"{m['mutant_id']}_zii.pt")
            if is_first_batch:
                capture_trajectory(dump_dir, lens, out_dir)
            n_done += len(batch)
            if n_done % 40 < BATCH:
                rate = n_done / max(time.time() - t0, 1e-6)
                log.info(f"[{arm}] {n_done}/{len(todo)} {rate:.2f} mut/s "
                         f"ETA {(len(todo) - n_done) / max(rate, 1e-6) / 60:.0f}min")
        except Exception as e:
            log.error(f"[{arm}] batch {b} FAILED: {e}", exc_info=True)
        finally:
            os.environ.pop("RF3_RECYCLE_DUMP_DIR", None)
            shutil.rmtree(work, ignore_errors=True)
            torch.cuda.empty_cache()
    log.info(f"[{arm}] DONE {n_done}/{len(todo)} in {(time.time() - t0) / 60:.0f}min")


def main():
    arms = sys.argv[1:] or ARM_ORDER
    subset, wt_seq = load_manifest_subset()
    log.info(f"subset: {len(subset)} entries, arms: {arms}")
    engine = init_engine()
    for arm in arms:
        run_arm(engine, arm, subset, wt_seq)
    log.info("ALL E7 ARMS DONE")


if __name__ == "__main__":
    main()
