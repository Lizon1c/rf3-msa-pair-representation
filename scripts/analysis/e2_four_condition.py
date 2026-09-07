#!/usr/bin/env python3
"""E2: four-condition redo on CLEAN full-set RBD embeddings (replaces report §2.1.1.3).

Conditions (report rows + fusion-as-validator dead-stream controls):
  sp_no        SinglePredictorV2(Z_no)                 [report row 1]
  sp_msa       SinglePredictorV2(Z_msa=full15)         [report row 2 — was −40%, artifacted]
  x_no_msa     XAttn(Z_no, Z_msa)                      [report row 3 — "±0% filter"]
  x_msa_faesm  CrossAttnOrthoConcatFusion(Z_msa, FAESM)[report row 4 — "+7%"]
  x_no_zeros   XAttn(Z_no, zeros)    [validator: architecture-effect control]
  x_msa_zeros  XAttn(Z_msa, zeros)   [validator: architecture-effect control]

Protocol (July benchmark): fusion_v3 architectures, per-residue standardization
(axis=0), cross-position 70/30, split seeds 100-104, inits 7/107/207, 300 ep,
BS=64, AdamW lr1e-4 wd0.05, cosine, MSE, best-epoch val Spearman, raw bind_avg
labels (rbd_repro convention). Orthogonal penalty OFF (E9/E13: decorative/inert).
Validator reporting: split-paired deltas; effects <0.02 = noise floor.

Data: full N=3998 (metadata FILE order == extraction order).
Run: CUDA_VISIBLE_DEVICES=1 python -B e2_four_condition.py [cond ...]
Output: /mnt/k/output_heads/rbd/e2_four_condition/results_<E2_TAG>.json
(default tag 'main'). Archived runs: v3 (the canonical per-channel rerun,
6 cells × 15 runs) and A/B (independent split-GPU run; A = sp_no/sp_msa/
x_no_msa/x_msa_faesm, B = x_no_zeros/x_msa_zeros). Repo copies:
msa_role_properties/results/e2_matrix_rbd/four_condition_{v3,A,B}.json.
"""
import json, math, os, time
import numpy as np
import torch, torch.nn as nn
from pathlib import Path
from scipy import stats

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
torch.set_float32_matmul_precision('high')
import sys
sys.path.insert(0, '/mnt/j/conda_envs/foundry/DMS_Project')
from fusion_v3 import CrossAttnOrthoConcatFusion, SinglePredictorV2, Encoder, AttnPool, PosEnc

RBD = Path('/mnt/k/output_heads/rbd')
OUT = RBD / 'e2_four_condition'
OUT.mkdir(exist_ok=True)
DEVICE = torch.device('cuda')
BS, EPOCHS, LR, WD = 64, 300, 1e-4, 0.05
FINAL_DIM, DM, NH, DROPOUT, FAESM_DIM, L = 128, 256, 8, 0.1, 1280, 201
t0 = time.time()


def log(m):
    print(f'[{time.time()-t0:7.1f}s] {m}', flush=True)


class XAttn(nn.Module):
    """fusion_v3.CrossAttnOrthoConcatFusion with configurable input dims
    (for Z×Z and Z×zeros pairs; mirrors fusion_v3 exactly otherwise)."""

    def __init__(self, d_z=128, d_f=128):
        super().__init__()
        self.z_enc = Encoder(d_z)
        self.f_enc = Encoder(d_f)
        interact_el = nn.TransformerEncoderLayer(d_model=FINAL_DIM, nhead=NH // 2,
                                                 dim_feedforward=FINAL_DIM * 4,
                                                 dropout=DROPOUT, batch_first=True)
        self.interaction = nn.TransformerEncoder(interact_el, num_layers=1)
        self.pe = PosEnc(DM)
        el = nn.TransformerEncoderLayer(d_model=DM, nhead=NH, dim_feedforward=DM * 4,
                                        dropout=DROPOUT, batch_first=True)
        self.joint_enc = nn.TransformerEncoder(el, num_layers=2)
        self.pool = AttnPool(DM)
        self.head = nn.Sequential(nn.Linear(DM, DM // 2), nn.GELU(),
                                  nn.Dropout(DROPOUT), nn.Linear(DM // 2, 1))

    def forward(self, xz, xf):
        z = self.z_enc(xz)
        f = self.f_enc(xf)
        B, L_, D = z.shape
        tokens = torch.stack([z, f], dim=2).view(B * L_, 2, D)
        interacted = self.interaction(tokens).view(B, L_, 2, D)
        j = torch.cat([interacted[:, :, 0, :], interacted[:, :, 1, :]], dim=-1)
        j = self.pe(j)
        j = self.joint_enc(j)
        j = self.pool(j)
        return self.head(j).squeeze(-1)


# ── data ──
meta = [json.loads(l) for l in open(RBD / 'metadata.jsonl') if l.strip()]
N = len(meta)
y = np.array([m['bind_avg'] for m in meta], np.float32)
pos = np.array([m['site_rbd'] - 1 for m in meta])


def load_z_no():
    X = np.empty((N, L, 128), np.float32)
    for i, m in enumerate(meta):
        z = torch.load(RBD / 'zii' / f"{m['mutant_id']}_zii.pt", map_location='cpu', weights_only=True)
        if isinstance(z, dict):
            z = list(z.values())[0]
        X[i] = z[0].float().numpy()
        if i % 1000 == 999:
            log(f'  z_no {i+1}/{N}')
    return torch.from_numpy(X)


def load_z_msa():
    X = np.empty((N, L, 128), np.float32)
    for i in range(N):
        z = torch.load(RBD / 'zii_e1_full15' / f'mutant_{i:04d}_zii.pt',
                       map_location='cpu', weights_only=True)
        if isinstance(z, dict):
            z = list(z.values())[0]
        X[i] = z[0].float().numpy()
        if i % 1000 == 999:
            log(f'  z_msa {i+1}/{N}')
    return torch.from_numpy(X)


def load_faesm():
    X = np.empty((N, L, FAESM_DIM), np.float32)
    for i, m in enumerate(meta):
        X[i] = torch.load(RBD / 'faesm' / f"{m['mutant_id']}.pt",
                          map_location='cpu', weights_only=True)[:L].float().numpy()
        if i % 1000 == 999:
            log(f'  faesm {i+1}/{N}')
    return torch.from_numpy(X)


def std_per_residue(X):
    """Canonical per-position, per-channel z-score (AGENTS.md §4 convention:
    check X.std(axis=0) yields a VECTOR, not a scalar). Scalar-per-position
    std crushes ESM's channel variance hierarchy and specifically degrades
    fusion cells (A/B verified 2026-07-27: x_no_faesm 0.483 -> 0.521)."""
    for i in range(X.shape[1]):
        mu = X[:, i].mean(0)
        sd = X[:, i].std(0).clamp_min(1e-8)
        X[:, i] = (X[:, i] - mu) / sd
    return X


def cross_split(seed):
    rng = np.random.RandomState(seed)
    ap = sorted(np.unique(pos))
    rng.shuffle(ap)
    vp = set(ap[:max(1, int(len(ap) * 0.3))])
    tr = [i for i in range(N) if pos[i] not in vp]
    va = [i for i in range(N) if pos[i] in vp]
    return np.array(tr), np.array(va)


def train_one(model, inputs, tr, va, init):
    torch.manual_seed(init)
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    tr_t = torch.from_numpy(tr).to(DEVICE)
    va_t = torch.from_numpy(va).to(DEVICE)
    yt = torch.from_numpy(y[tr]).to(DEVICE)
    yv_np = y[va]
    best = -1
    ins_tr = [x[tr_t] for x in inputs]
    ins_va = [x[va_t] for x in inputs]

    def fwd(*xs):
        out = model(*xs) if len(xs) > 1 else model(xs[0])
        # fusion_v3.CrossAttnOrthoConcatFusion returns (pred, z_out, f_out)
        return out[0] if isinstance(out, tuple) else out

    for ep in range(EPOCHS):
        model.train()
        idx = torch.randperm(len(tr), device=DEVICE)
        for bi in range(0, len(tr), BS):
            b = idx[bi:bi + BS]
            loss = nn.MSELoss()(fwd(*[x[b] for x in ins_tr]), yt[b])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            pv = torch.cat([fwd(*[x[i:i + BS] for x in ins_va])
                            for i in range(0, len(va), BS)]).cpu().numpy()
        r, _ = stats.spearmanr(pv, yv_np)
        best = max(best, r if not np.isnan(r) else -1)
    del model
    torch.cuda.empty_cache()
    return best


def main():
    conds = sys.argv[1:] or ['sp_no', 'sp_msa', 'x_no_msa',
                             'x_msa_faesm', 'x_no_zeros', 'x_msa_zeros']
    log(f'conds: {conds}')
    Z_no = std_per_residue(load_z_no()).to(DEVICE)
    log('z_no ready')
    Z_msa = std_per_residue(load_z_msa()).to(DEVICE)
    log('z_msa ready')
    Zeros = torch.zeros_like(Z_no)
    F_aesm = None

    def inputs_for(c):
        nonlocal F_aesm
        if c == 'sp_no':
            return SinglePredictorV2(128), [Z_no]
        if c == 'sp_msa':
            return SinglePredictorV2(128), [Z_msa]
        if c == 'x_no_msa':
            return XAttn(128, 128), [Z_no, Z_msa]
        if c == 'x_msa_faesm':
            if F_aesm is None:
                F_aesm = std_per_residue(load_faesm()).to(DEVICE)
                log('faesm ready')
            return CrossAttnOrthoConcatFusion(), [Z_msa, F_aesm]
        if c == 'x_no_zeros':
            return XAttn(128, 128), [Z_no, Zeros]
        if c == 'x_msa_zeros':
            return XAttn(128, 128), [Z_msa, Zeros]
        raise ValueError(c)

    res_path = OUT / f"results_{os.environ.get('E2_TAG', 'main')}.json"
    done = set()
    if res_path.exists():
        prev = json.load(open(res_path))
        done = {(r['cond'], r['split'], r['init']) for r in prev['runs']}
        results = prev['runs']
    else:
        results = []
    for c in conds:
        for sd in [100, 101, 102, 103, 104]:
            tr, va = cross_split(sd)
            for init in [7, 107, 207]:
                if (c, sd, init) in done:
                    continue
                model, inputs = inputs_for(c)
                r = train_one(model, inputs, tr, va, init)
                results.append({'cond': c, 'split': sd, 'init': init, 'val': r})
                log(f'{c} split{sd} init{init}: {r:.4f}')
                with open(res_path, 'w') as f:
                    json.dump({'runs': results}, f)

    summary = {}
    for c in conds:
        vals = [r['val'] for r in results if r['cond'] == c]
        if vals:
            summary[c] = {'mean': float(np.mean(vals)), 'std': float(np.std(vals)),
                          'min': float(np.min(vals)), 'max': float(np.max(vals))}
            log(f"{c}: μ={summary[c]['mean']:.4f} σ={summary[c]['std']:.4f}")
    # split-paired deltas (validator reporting)
    def paired(a, b):
        da = {r['split']: [] for r in results if r['cond'] == a}
        for r in results:
            if r['cond'] == a:
                da[r['split']].append(r['val'])
        db = {r['split']: [] for r in results if r['cond'] == b}
        for r in results:
            if r['cond'] == b:
                db[r['split']].append(r['val'])
        ds = []
        for sd in da:
            if sd in db and da[sd] and db[sd]:
                ds.append(float(np.mean(da[sd]) - np.mean(db[sd])))
        return ds
    pairs = [('sp_msa', 'sp_no'), ('x_no_msa', 'sp_no'), ('x_no_zeros', 'sp_no'),
             ('x_msa_zeros', 'sp_msa'), ('x_msa_faesm', 'sp_msa'), ('x_no_msa', 'x_no_zeros')]
    pd_out = {}
    for a, b in pairs:
        ds = paired(a, b)
        if ds:
            pd_out[f'{a}-{b}'] = {'per_split': ds, 'mean': float(np.mean(ds))}
            log(f'paired {a} - {b}: {["%+.3f" % d for d in ds]} μ={np.mean(ds):+.4f}')
    with open(res_path, 'w') as f:
        json.dump({'runs': results, 'summary': summary, 'paired': pd_out}, f, indent=1)
    log('saved')


if __name__ == '__main__':
    main()
