"""Pick heads for E3 without leakage. Splits own-right items in half (seed 1684). On the SELECT half, ranks heads by t
(broken vs kept, look_again attention to the model's own answer; most negative first). Writes JSON with the top heads,
and the item keys of the TEST half (own-right) and 100 random own-wrong keys (heads were never selected on them).
Usage: [GROUP=wrong] python3 scripts/interp/select_heads.py OUT.npz heads.json [K=16]
"""
import os, sys, json, numpy as np
d = np.load(sys.argv[1], allow_pickle=True); K = int(sys.argv[3]) if len(sys.argv) > 3 else 16
L = "abcde"; right = np.array([str(c)[-1] for c in d["right_code"]]); ownc = d["own_correct"].astype(bool)
ok = d["gen_ok_look_again"] if "gen_ok_look_again" in d.files else np.array([L[i] for i in d["probs_look_again"].argmax(1)]) == right
la = d["attn_look_again"][:, :, 1, :]; keys = [str(k) for k in d["keys"]]
WRONG = os.environ.get("GROUP") == "wrong"   # GROUP=wrong: select on own-wrong items; test = other own-wrong half (capped 100) + 100 random own-right items
idx = np.where(~ownc if WRONG else ownc)[0]; p = np.random.default_rng(1684).permutation(len(idx)); sel, test = idx[p[:len(idx) // 2]], idx[p[len(idx) // 2:]]
x, z = la[sel][~ok[sel]], la[sel][ok[sel]]
t = (x.mean(0) - z.mean(0)) / np.sqrt(x.var(0, ddof=1) / len(x) + z.var(0, ddof=1) / len(z) + 1e-12)
top = [[int(l), int(h)] for l, h in np.dstack(np.unravel_index(np.argsort(t.ravel())[:K], t.shape))[0]]
rng2 = np.random.default_rng(1684)
if WRONG: own_right_keys, own_wrong_keys = [keys[i] for i in rng2.permutation(np.where(ownc)[0])[:100]], [keys[i] for i in test[:100]]
else: own_right_keys, own_wrong_keys = [keys[i] for i in test], [keys[i] for i in rng2.permutation(np.where(~ownc)[0])[:100]]
json.dump({"heads": top, "select_keys": [keys[i] for i in sel], "test_own_right_keys": own_right_keys, "own_wrong_keys": own_wrong_keys}, open(sys.argv[2], "w"))
print("selected on", len(sel), "items (broken", int((~ok[sel]).sum()), "); test", len(test), "(broken", int((~ok[test]).sum()), ")")
print("top heads:", ", ".join(f"L{l}H{h}" for l, h in top))
