"""Rebuild the exact chat prompts for the interpretability runs (plan: docs/interp_plan.md, E0-E2).

For one model's own-error result file, each item with a turn-1 answer and re-ask rows gets:
  messages_<wording>: the full chat (turn 1 user, the model's own turn-1 text, re-ask user message)
  outcome_<wording>:  kept | broken (own-right items) or recovered | stayed (own-wrong items), from the API run
  right_code / own_code: the acceptable code and the model's own turn-1 code.
Prompts are rebuilt from the scenario file and seed, as in scripts/run/run_own_error_reask.py.
Usage: python3 scripts/interp/build_pairs.py RESULTS.jsonl OUT.jsonl --pools medqa=medqa.jsonl,csqa=csqa.jsonl
"""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "run"))
import run_own_error_reask as R

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results"); ap.add_argument("out")
    ap.add_argument("--scen-dir", default=str(ROOT / "data/scenarios/realworld"))
    ap.add_argument("--pools", default="medqa=medqa.jsonl,csqa=csqa.jsonl")
    a = ap.parse_args()
    files = [tuple(x.split("=", 1)) for x in a.pools.split(",")]
    S = {s["_key"]: s for s in R.build_pool(a.scen_dir, files=files)}
    rows = [json.loads(l) for l in open(a.results) if l.strip()]
    t1 = {r["key"]: r for r in rows if r["phase"] == "turn1" and r.get("choice")}
    items = {}
    for r in rows:
        if r["phase"] != "reask" or not r.get("choice") or r["key"] not in t1 or r["key"] not in S:
            continue
        s, t = S[r["key"]], t1[r["key"]]
        ok = r["choice"] in s["acceptable"]["base"]
        it = items.setdefault(r["key"], {
            "key": r["key"], "pool": s["_pool"], "own_correct": t["own_correct"],
            "own_code": t["choice"], "right_code": s["acceptable"]["base"][0],
            "option_order": t.get("option_order"), "own_text": t["text"]})
        it[f"outcome_{r['wording']}"] = (("kept" if ok else "broken") if t["own_correct"]
                                        else ("recovered" if ok else "stayed"))
        it[f"choice_{r['wording']}"] = r["choice"]
        it[f"messages_{r['wording']}"] = R.reask_msgs(s, t["text"], r["wording"])
    with open(a.out, "w") as f:
        for it in items.values():
            f.write(json.dumps(it) + "\n")
    import collections
    c = collections.Counter((it["own_correct"], it.get("outcome_look_again"), it.get("outcome_neutral")) for it in items.values())
    print(len(items), "items"); [print(k, v) for k, v in sorted(c.items(), key=str)]

if __name__ == "__main__":
    main()
