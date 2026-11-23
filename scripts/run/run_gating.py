"""Gating test (design: docs/gating_design.md). Adds, for one model, the calls the policy comparison needs.

Reads the model's existing own-error result file (turn-1 answers, re-asks), then makes:
  phase "extra": two new independent samples of the turn-1 prompt for every valid item (k = 2, 3; a1 is the old turn-1 answer)
  phase "fill":  a look_again re-ask for every own-right item that has none yet (the old run re-asked only a control sample)
Writes results/gating/<model>__gating.jsonl (new rows only). RESUME=1 skips finished rows and retries error rows.
Usage: CHAT_BACKEND=openai CHAT_BASE_URL=.. CHAT_API_KEY=.. CHAT_MODEL=.. python3 scripts/run/run_gating.py --existing results/realworld/<model>__own_error.jsonl --yes
       --dry-run counts the calls and makes none.
"""
import argparse, json, os, sys, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib")); sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("RANDOMIZE_OPTIONS", "1"); os.environ.setdefault("OPTION_ORDER_SEED", "1684")
import run_own_error_reask as R
from run_second_look import extract_code
from run_seeded_reask import Client

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--existing", required=True); ap.add_argument("--scen-dir", default="data/scenarios/realworld")
    ap.add_argument("--pools", default="medqa=medqa.jsonl,csqa=csqa.jsonl"); ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=2000); ap.add_argument("--limit", type=int)
    ap.add_argument("--yes", action="store_true"); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    files = [tuple(x.split("=", 1)) for x in a.pools.split(",")]
    S = {s["_key"]: s for s in R.build_pool(a.scen_dir, files=files)}
    old = [json.loads(l) for l in open(a.existing) if l.strip()]
    t1 = {r["key"]: r for r in old if r["phase"] == "turn1" and r.get("choice")}
    have_reask = {r["key"] for r in old if r["phase"] == "reask" and r["wording"] == "look_again" and r.get("choice")}
    keys = sorted(t1)[: a.limit]
    extra = [(k, f"a{j}", R.turn1_msgs(S[k])) for k in keys for j in (2, 3)]
    fill = [(k, "look_again", R.reask_msgs(S[k], t1[k]["text"], "look_again")) for k in keys if t1[k]["own_correct"] and k not in have_reask]
    print(f"existing valid items {len(t1)}; extra sample calls {len(extra)}; fill re-asks {len(fill)}; total {len(extra) + len(fill)}")
    if a.dry_run: return
    if os.environ.get("CHAT_BACKEND") != "openai": sys.exit("CHAT_BACKEND=openai required")
    key, base, model = os.environ.get("CHAT_API_KEY"), os.environ.get("CHAT_BASE_URL"), os.environ.get("CHAT_MODEL")
    if not (key and base and model): sys.exit("needs CHAT_BASE_URL, CHAT_API_KEY, CHAT_MODEL")
    if not (a.yes or os.environ.get("CONFIRM") == "1"): sys.exit("Not calling the API: pass --yes.")
    Path("results/gating").mkdir(parents=True, exist_ok=True)
    path = f"results/gating/{model.replace('/', '_')}__gating.jsonl"
    rows = [json.loads(l) for l in open(path) if l.strip()] if os.environ.get("RESUME") == "1" and os.path.exists(path) else []
    rows = [r for r in rows if r.get("stop_reason") != "error"]
    done = {(r["key"], r["phase"], r["tag"]) for r in rows}
    client, lock, errors, out = Client(base, key, model, a.max_tokens, None), threading.Lock(), [0], open(path, "w")
    for r in rows: out.write(json.dumps(r) + "\n")
    def run(phase, tasks):
        todo = [t for t in tasks if (t[0], phase, t[1]) not in done]; cnt = [0]
        print(f"phase {phase}: {len(todo)} calls ({len(tasks) - len(todo)} done)", flush=True)
        def work(t):
            k, tag, msgs = t; s = S[k]
            try: txt, stop = client.ask(msgs); choice = extract_code(txt, s["options"])
            except Exception as e: txt, stop, choice = f"{type(e).__name__}: {e}", "error", None
            row = {"model": model, "key": k, "phase": phase, "tag": tag, "choice": choice, "own_correct": t1[k]["own_correct"], "stop_reason": stop,
                   "text": txt if phase == "fill" else None}
            with lock:
                out.write(json.dumps(row) + "\n"); out.flush(); cnt[0] += 1; errors[0] += stop == "error"
                if cnt[0] % 50 == 0 or cnt[0] == len(todo): print(f"  {phase} {cnt[0]}/{len(todo)} ({errors[0]} errors)", flush=True)
        with ThreadPoolExecutor(a.workers) as ex:
            for f in [ex.submit(work, t) for t in todo]: f.result()
    try:
        run("extra", extra); run("fill", fill)
    finally: out.close()
    print("wrote", path, f"({errors[0]} error rows this run)")

if __name__ == "__main__":
    main()
