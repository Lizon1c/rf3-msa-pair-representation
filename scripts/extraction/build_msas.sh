#!/bin/bash
DB=/mnt/j/mmseqs2_db/uniref30_2202_db_seq
OUT=/mnt/j/conda_envs/foundry/DMS_Project/inputs/e2_targets/msas
mkdir -p $OUT
PY=/root/miniforge3/envs/foundry/bin/python
for FA in /tmp/e2/seqs/*.fa; do
  TID=$(basename $FA | cut -d_ -f1)
  W=/tmp/e2/msa_$TID; mkdir -p $W; cd $W
  [ -s $OUT/${TID}.a3m ] && { echo "$TID exists, skip"; continue; }
  echo "=== $TID $(date) ==="
  mmseqs createdb $FA qdb -v 1
  mmseqs search qdb $DB res tmp -s 7.5 --max-seqs 256 -v 1
  mmseqs align qdb $DB res aln -a -v 1
  mmseqs convertalis qdb $DB aln $TID.m8 --format-output "query,target,bits,evalue,qstart,qend,qaln,taln" -v 1
  $PY - "$TID" "$FA" "$W/$TID.m8" "$OUT/${TID}.a3m" <<'PYEOF'
import sys, numpy as np
tid, fa, m8, out = sys.argv[1:5]
rows = [l.rstrip('\n').split('\t') for l in open(m8) if l.strip()]
lines = open(fa).read().split('\n')
query = lines[1].strip(); L = len(query)
o = [lines[0], query]
seen, kept = set(), 0
for q, t, bits, ev, qs, qe, qaln, taln in rows:
    name = t.split()[0]
    if name in seen: continue
    seen.add(name)
    qs, qe = int(qs), int(qe)
    core = [tc if tc != '-' else '-' for qc, tc in zip(qaln, taln) if qc != '-']
    row = '-' * (qs - 1) + ''.join(core) + '-' * (L - qe)
    if len(row) != L: continue
    s = row.replace('-', '')
    if not s or s.upper() == query: continue
    o.append(f'>{name}|bits={bits}')
    o.append(row)
    kept += 1
    if kept >= 256: break
with open(out, 'w') as f:
    f.write('\n'.join(o) + '\n')
seqs = [l for l in open(out).read().split('\n') if l and not l.startswith('>')]
arr = np.array([list(s) for s in seqs], dtype='S')
print(f'{tid}: {kept} homologs, rectangular {arr.shape} OK')
PYEOF
done
echo "=== ALL MSAS DONE $(date) ==="
