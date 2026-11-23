"""E0-E2 analysis of out/gemma2_full.npz from extract_gemma2.py (plan: docs/interp_plan.md).

E0 gate: does the local, answer-forced run show the break pattern? (own-right items: share of forced answers that are wrong, neutral vs look_again)
E2: per layer and head, attention mass from the last position to the re-ask sentence and to the model's own answer. look_again vs neutral, broken vs kept.
E1-lite: cross-validated logistic probe on the last-position residual per layer: can it tell broken from kept (own-right items)?
Direction: mean(look_again - neutral) per layer; its projection for broken vs kept.
All intervals: 2000 bootstrap resamples over items, seed 1684. Heads: reported with a permutation p, uncorrected, for ranking only.
Usage: [GROUP=wrong] python3 scripts/interp/analyze_e0_e2.py OUT.npz > summary.txt
GROUP=right (default): own-right items; positive = broken. GROUP=wrong: own-wrong items; positive = stuck (still wrong after look_again).
"""
import os, sys, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

d = np.load(sys.argv[1], allow_pickle=True)
rng = np.random.default_rng(1684)
L = "abcde"
right = np.array([str(c)[-1] for c in d["right_code"]]); own = np.array([str(c)[-1] for c in d["own_code"]])
ownc = d["own_correct"].astype(bool)
top = {w: np.array([L[i] for i in d[f"probs_{w}"].argmax(1)]) for w in ("neutral", "look_again")}
ok = {w: top[w] == right for w in top}
api = {w: d[f"api_outcome_{w}"] for w in top}

def boot(x, n=2000):
    x = np.asarray(x, float); m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)

print(f"items {len(right)}; own-right {ownc.sum()}; own-wrong {(~ownc).sum()}")
print("\n== E0 gate: forced-answer behavior (local) vs API run ==")
for name, mask in (("own-right: share WRONG after re-ask (break)", ownc), ("own-wrong: share RIGHT after re-ask (recovery)", ~ownc)):
    for w in ("neutral", "look_again"):
        v = (~ok[w][mask]) if mask is ownc else ok[w][mask]
        m, lo, hi = boot(v); apiv = np.mean([(o in ("broken",)) if mask is ownc else (o == "recovered") for o in api[w][mask]])
        print(f"{name:50s} {w:10s} local {m:.2f} [{lo:.2f},{hi:.2f}]   api {apiv:.2f}  n={mask.sum()}")
bl, bn = (~ok["look_again"][ownc]).mean(), (~ok["neutral"][ownc]).mean()
print(f"GATE (break rate look_again >= 0.20 and >= 2x neutral): {'PASS' if bl >= 0.20 and bl >= 2 * bn else 'FAIL'}  (look_again {bl:.2f}, neutral {bn:.2f})")
agree = np.mean([(ok[w][i] == (api[w][i] in ("kept", "recovered"))) for w in top for i in range(len(right))])
print(f"local forced vs API final answer agree (right/wrong): {agree:.2f}")

# groups among own-right items, by the local forced outcome
G = ~ownc if os.environ.get("GROUP") == "wrong" else ownc
kept = G & ok["look_again"]; broken = G & ~ok["look_again"]   # broken = bad outcome: right->wrong (GROUP=right) or wrong->still wrong (GROUP=wrong)
print(f"\ngroup {os.environ.get('GROUP', 'right')}, local look_again: good outcome {kept.sum()}, bad outcome {broken.sum()}")

print("\n== E2: attention mass (last position -> span), mean over heads' items ==")
spans = list(d["spans"])
for w in ("neutral", "look_again"):
    a = d[f"attn_{w}"]  # items x layers x spans x heads
    print(w, {s: round(float(a[:, :, i, :].mean()), 4) for i, s in enumerate(spans)})
a_la, a_n = d["attn_look_again"], d["attn_neutral"]
for si, s in enumerate(spans):
    if s == "question": continue
    print(f"\n-- span '{s}': head effect = mean over own-right items of (broken - kept) in look_again attention; top 12 heads by |t|")
    x, y = a_la[broken, :, si, :], a_la[kept, :, si, :]
    if len(x) > 2 and len(y) > 2:
        t = (x.mean(0) - y.mean(0)) / np.sqrt(x.var(0, ddof=1) / len(x) + y.var(0, ddof=1) / len(y) + 1e-12)
        idx = np.dstack(np.unravel_index(np.argsort(-np.abs(t).ravel())[:12], t.shape))[0]
        allv = np.concatenate([x, y]); lab = np.array([1] * len(x) + [0] * len(y))
        for l, h in idx:
            obs = abs(x[:, l, h].mean() - y[:, l, h].mean()); c = 0
            for _ in range(500):
                p = rng.permutation(lab); c += abs(allv[p == 1][:, l, h].mean() - allv[p == 0][:, l, h].mean()) >= obs
            print(f"   L{l:02d}H{h:02d}  broken {x[:, l, h].mean():.4f}  kept {y[:, l, h].mean():.4f}  t {t[l, h]:+.2f}  perm p {(c + 1) / 501:.3f}")
    print(f"   look_again - neutral (all items), top 8 heads by mean difference")
    diff = (a_la[:, :, si, :] - a_n[:, :, si, :]).mean(0)
    for l, h in np.dstack(np.unravel_index(np.argsort(-np.abs(diff).ravel())[:8], diff.shape))[0]:
        print(f"   L{l:02d}H{h:02d}  la {a_la[:, l, si, h].mean():.4f}  neutral {a_n[:, l, si, h].mean():.4f}  diff {diff[l, h]:+.4f}")

print("\n== E1-lite: 5-fold CV AUC, residual at last position -> broken vs kept (own-right items) ==")
m = kept | broken
y = broken[m].astype(int)
if y.sum() >= 8 and (1 - y).sum() >= 8:
    for w in ("neutral", "look_again"):
        H = d[f"hidden_{w}"][m].astype(np.float32); res = []
        for l in range(0, H.shape[1], 2):
            pipe = make_pipeline(StandardScaler(), LogisticRegression(C=0.01, max_iter=500))
            p = cross_val_predict(pipe, H[:, l], y, cv=StratifiedKFold(5, shuffle=True, random_state=1684), method="predict_proba")[:, 1]
            res.append((l, roc_auc_score(y, p)))
        print(w, " ".join(f"L{l}:{a:.2f}" for l, a in res), f"(n={len(y)}, broken={y.sum()})")
else:
    print("too few broken or kept items for a probe")

print("\n== Direction: mean(look_again - neutral) per layer; projection, broken vs kept (own-right items) ==")
if broken.sum() > 2 and kept.sum() > 2:
    dh = (d["hidden_look_again"].astype(np.float32) - d["hidden_neutral"].astype(np.float32))
    for l in range(0, dh.shape[1], 3):
        v = dh[ownc, l].mean(0); v /= np.linalg.norm(v) + 1e-9
        pb, pk = d["hidden_look_again"][broken, l].astype(np.float32) @ v, d["hidden_look_again"][kept, l].astype(np.float32) @ v
        print(f"L{l:02d}  broken {pb.mean():9.2f}  kept {pk.mean():9.2f}  diff {pb.mean() - pk.mean():+8.2f}  norm of mean shift {np.linalg.norm(dh[ownc, l].mean(0)):.1f}")
