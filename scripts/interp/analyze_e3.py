"""E3 analysis of patch_gemma2.json (plan: docs/interp_plan.md). Test half of own-right items + 100 own-wrong items.
Groups by the baseline look_again forced answer: broken (own-right, wrong), kept (own-right, right), own-wrong.
Reports, per condition, share forced-right after patching neutral head outputs into the look_again run, with 95% bootstrap CI over items;
random-head conditions are summarized over 10 draws. Also: patching look_again heads into the neutral run (induced breaks).
Usage: python3 scripts/interp/analyze_e3.py patch_gemma2.json
"""
import sys, json, numpy as np
D = json.load(open(sys.argv[1])); items = D["items"]; rng = np.random.default_rng(1684)
def fr(it, key, c=None):
    p = np.array(it[key] if c is None else it[key][c]); return int(p.argmax() == it["right"])
def pr(it, key, c=None):
    p = np.array(it[key] if c is None else it[key][c]); return float(p[it["right"]])
def boot(x, n=2000):
    x = np.asarray(x, float)
    if len(x) == 0: return (float("nan"),) * 3
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]; return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)
own_right = [i for i in items if i["own_correct"]]; own_wrong = [i for i in items if not i["own_correct"]]
broken = [i for i in own_right if not fr(i, "base_look_again")]; kept = [i for i in own_right if fr(i, "base_look_again")]
print(f"items: own-right test {len(own_right)} (broken {len(broken)}, kept {len(kept)}); own-wrong {len(own_wrong)}")
print("baselines  forced-right rate: broken-group neutral %.2f, look_again %.2f | kept-group neutral %.2f, look_again %.2f | own-wrong neutral %.2f, look_again %.2f" % tuple(
    np.mean([fr(i, k) for i in g]) for g in (broken, kept, own_wrong) for k in ("base_neutral", "base_look_again")))
conds = list(items[0]["la_patched"].keys())
def line(name, group, c):
    m, lo, hi = boot([fr(i, "la_patched", c) for i in group]); p, plo, phi = boot([pr(i, "la_patched", c) - pr(i, "base_look_again") for i in group])
    return f"  {name:28s} forced-right {m:.2f} [{lo:.2f},{hi:.2f}]   mean change in P(right) {p:+.2f} [{plo:+.2f},{phi:+.2f}]"
for gname, g in (("BROKEN items: look_again <- neutral heads (does the right answer come back?)", broken),
                 ("KEPT items: same patch (damage check; should stay right)", kept),
                 ("OWN-WRONG items: same patch (does it remove real recoveries?)", own_wrong)):
    print("\n==", gname, f"n={len(g)}")
    for c in ("top8", "top16", "all_24_36"):
        print(line(c, g, c))
    for fam in ("rand8_layers", "rand8_any"):
        cs = [c for c in conds if c.startswith(fam)]; v = [np.mean([fr(i, "la_patched", c) for i in g]) for c in cs]
        pv = [np.mean([pr(i, "la_patched", c) - pr(i, "base_look_again") for i in g]) for c in cs]
        print(f"  {fam+' (10 draws)':28s} forced-right mean {np.mean(v):.2f}, range {min(v):.2f}-{max(v):.2f}   mean change in P(right) {np.mean(pv):+.2f}, range {min(pv):+.2f} to {max(pv):+.2f}")
    t8 = np.mean([fr(i, "la_patched", "top8") for i in g]); rl = [np.mean([fr(i, "la_patched", c) for i in g]) for c in conds if c.startswith("rand8_layers")]
    print(f"  top8 is above {sum(t8 > r for r in rl)} of {len(rl)} random layer-matched sets in forced-right rate")
print("\n== INDUCED BREAKS: neutral <- look_again heads, own-right test items (neutral baseline break rate %.2f)" % (1 - np.mean([fr(i, "base_neutral") for i in own_right])))
for c in ("top8", "top16", "all_24_36"):
    m, lo, hi = boot([1 - fr(i, "n_patched", c) for i in own_right]); print(f"  {c:28s} break rate {m:.2f} [{lo:.2f},{hi:.2f}]")
cs = [c for c in conds if c.startswith("rand8_layers_0")]
for c in cs: print(f"  {c:28s} break rate {np.mean([1 - fr(i, 'n_patched', c) for i in own_right]):.2f}")
