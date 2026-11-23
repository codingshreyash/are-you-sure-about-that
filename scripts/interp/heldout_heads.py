"""Held-out check for the own-answer attention heads (E2). Pick top-K heads by |t| (broken vs kept, look_again attention to the
model's own answer) on a random half of the own-right items; score the other half with the mean attention of those heads.
Repeat 200 random splits. Reports mean held-out AUC (lower attention = broken). Also the same score on own-wrong items:
does a larger attention drop go with recovery? Usage: [GROUP=wrong] python3 scripts/interp/heldout_heads.py OUT.npz [K=8]
"""
import os, sys, numpy as np
from scipy.stats import rankdata
d = np.load(sys.argv[1], allow_pickle=True); K = int(sys.argv[2]) if len(sys.argv) > 2 else 8
L = "abcde"; right = np.array([str(c)[-1] for c in d["right_code"]]); ownc = d["own_correct"].astype(bool)
ok = np.array([L[i] for i in d["probs_look_again"].argmax(1)]) == right
la, ne = d["attn_look_again"][:, :, 1, :], d["attn_neutral"][:, :, 1, :]   # own_answer span: items x layers x heads
idx = np.where(~ownc if os.environ.get('GROUP') == 'wrong' else ownc)[0]; y_all = (~ok)[idx].astype(int)
def auc(x, y):
    r = rankdata(x); n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
rng = np.random.default_rng(1684); res = []; res_drop = []; sel = {}
for _ in range(200):
    p = rng.permutation(len(idx)); a, b = idx[p[:len(idx) // 2]], idx[p[len(idx) // 2:]]
    ya, yb = (~ok)[a].astype(int), (~ok)[b].astype(int)
    if min(ya.sum(), len(ya) - ya.sum(), yb.sum(), len(yb) - yb.sum()) < 4: continue
    x, z = la[a][ya == 1], la[a][ya == 0]
    t = (x.mean(0) - z.mean(0)) / np.sqrt(x.var(0, ddof=1) / len(x) + z.var(0, ddof=1) / len(z) + 1e-12)
    top = np.dstack(np.unravel_index(np.argsort(t.ravel())[:K], t.shape))[0]       # most negative t = lower attention when broken
    for l, h in top: sel[(l, h)] = sel.get((l, h), 0) + 1
    s = np.mean([la[b][:, l, h] for l, h in top], 0); sd = np.mean([la[b][:, l, h] - ne[b][:, l, h] for l, h in top], 0)
    res.append(1 - auc(s, yb)); res_drop.append(1 - auc(sd, yb))
print(f"splits used {len(res)}; held-out AUC, look_again attention of top-{K} heads (lower=broken): mean {np.mean(res):.2f}, 5-95% {np.percentile(res,5):.2f}-{np.percentile(res,95):.2f}")
print(f"held-out AUC, drop (look_again - neutral) of the same heads: mean {np.mean(res_drop):.2f}")
print("heads chosen most often:", ", ".join(f"L{l}H{h}({c})" for (l, h), c in sorted(sel.items(), key=lambda kv: -kv[1])[:10]))
