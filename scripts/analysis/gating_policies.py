"""Policy comparison for the gating test (design: docs/gating_design.md).
Per model: never / always_LA / gated_LA / vote3 / oracle_LA accuracy, signal quality, gains with 95% paired bootstrap over items
(2000 resamples, seed 1684). Verdict lines follow the rule written in the design doc.
Usage: python3 scripts/analysis/gating_policies.py EXISTING.jsonl GATING.jsonl [--pools medqa=medqa.jsonl,csqa=csqa.jsonl]
"""
import argparse, collections, json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "run")); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
os.environ.setdefault("RANDOMIZE_OPTIONS", "1"); os.environ.setdefault("OPTION_ORDER_SEED", "1684")
import run_own_error_reask as R

def auc(score, y):
    from scipy.stats import rankdata
    y = np.asarray(y, int); n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0: return float("nan")
    r = rankdata(score); return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("existing", help="hub own-error file, or - for a local run"); ap.add_argument("gating", help="hub gating file, or the local file from scripts/interp/gating_local.py")
    ap.add_argument("--scen-dir", default="data/scenarios/realworld"); ap.add_argument("--pools", default="medqa=medqa.jsonl,csqa=csqa.jsonl")
    a = ap.parse_args()
    S = {s["_key"]: s for s in R.build_pool(a.scen_dir, files=[tuple(x.split("=", 1)) for x in a.pools.split(",")])}
    new = [json.loads(l) for l in open(a.gating) if l.strip()]
    local = bool(new) and "a1" in new[0]                      # local format: one row per item with a1, a2, a3, la
    old = [] if local else [json.loads(l) for l in open(a.existing) if l.strip()]
    ans = collections.defaultdict(dict)
    if local:
        for r in new:
            for x in ("a1", "a2", "a3", "la"):
                if r.get(x): ans[r["key"]][x] = "choice_" + r[x]
        new = []
    for r in old:
        if r.get("choice") is None: continue
        if r["phase"] == "turn1": ans[r["key"]]["a1"] = r["choice"]
        if r["phase"] == "reask" and r["wording"] == "look_again": ans[r["key"]]["la"] = r["choice"]
    for r in new:
        if r.get("choice") is None or r.get("stop_reason") == "error": continue
        if r["phase"] == "extra": ans[r["key"]][r["tag"]] = r["choice"]
        if r["phase"] == "fill": ans[r["key"]]["la"] = r["choice"]
    keys = sorted(k for k, d in ans.items() if all(x in d for x in ("a1", "a2", "a3", "la")) and k in S)
    dropped = len([k for k, d in ans.items() if k in S]) - len(keys)
    if not keys: sys.exit(f"no item has all of a1, a2, a3 and a look_again re-ask (dropped {dropped})")
    exp = {k: S[k]["expected"]["base"] for k in keys}
    model_name = (json.loads(open(a.gating).readline())["model"] if local else old[0]["model"])
    c = lambda k, x: float(ans[k][x] == exp[k])
    a1 = np.array([c(k, "a1") for k in keys]); la = np.array([c(k, "la") for k in keys])
    unsure = np.array([not (ans[k]["a1"] == ans[k]["a2"] == ans[k]["a3"]) for k in keys])
    def vote(k):
        v = collections.Counter([ans[k]["a1"], ans[k]["a2"], ans[k]["a3"]]); ch, n = v.most_common(1)[0]
        return ch if n >= 2 else ans[k]["a1"]
    v3 = np.array([float(vote(k) == exp[k]) for k in keys])
    pol = {"never": a1, "always_LA": la, "gated_LA": np.where(unsure, la, a1), "vote3": v3, "oracle_LA": np.where(a1 == 0, la, a1)}
    rng = np.random.default_rng(1684); n = len(keys); idxs = [rng.integers(0, n, n) for _ in range(2000)]
    def ci(x): m = [x[i].mean() for i in idxs]; return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)
    print(f"model {model_name}: items used {n} (dropped for missing/unparsed answers: {dropped}); own accuracy a1 {a1.mean():.3f}")
    print(f"unsure share {unsure.mean():.3f} ({unsure.sum()} items); P(a1 wrong | unsure) {1 - a1[unsure].mean() if unsure.any() else float('nan'):.3f}; "
          f"P(a1 wrong | sure) {1 - a1[~unsure].mean():.3f}; AUC of unsure for a1 wrong {auc(unsure.astype(float), 1 - a1):.2f}")
    print(f"{'policy':10s} accuracy [95% CI]")
    for k, v in pol.items(): m, lo, hi = ci(v); print(f"{k:10s} {m:.3f} [{lo:.3f},{hi:.3f}]")
    print("gains (paired):")
    g = {}
    for name, x, y in (("gated_LA - never", pol["gated_LA"], a1), ("gated_LA - always_LA", pol["gated_LA"], la), ("gated_LA - vote3", pol["gated_LA"], v3),
                       ("always_LA - never", la, a1), ("vote3 - never", v3, a1)):
        m, lo, hi = ci(x - y); g[name] = (m, lo, hi); print(f"  {name:22s} {m:+.3f} [{lo:+.3f},{hi:+.3f}]")
    for p in ("medqa", "csqa"):
        sel = np.array([k.startswith(p + ":") for k in keys])
        if sel.sum(): print(f"  pool {p} (n={sel.sum()}): never {a1[sel].mean():.3f}  always_LA {la[sel].mean():.3f}  gated_LA {pol['gated_LA'][sel].mean():.3f}  vote3 {v3[sel].mean():.3f}")
    gn, ga, gv = g["gated_LA - never"], g["gated_LA - always_LA"], g["gated_LA - vote3"]
    if unsure.sum() < 30 or (gn[2] - gn[1]) > 0.06: verdict = "UNINFORMATIVE (fewer than 30 unsure items or interval wider than 0.06)"
    else: verdict = ("GATING HELPS" if gn[1] > 0 and ga[2] >= 0 else "gating does not help") + ("; BEATS VOTING" if gv[1] > 0 else "; does not beat voting")
    print("VERDICT:", verdict)

if __name__ == "__main__":
    main()
