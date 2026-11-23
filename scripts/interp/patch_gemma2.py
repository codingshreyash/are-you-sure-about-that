"""E3 patching for Gemma 2 9B IT (plan: docs/interp_plan.md). NEEDS A GPU. Same prompts and answer forcing as extract_gemma2.py.
Intervention: at the LAST position only, replace the per-head attention output (input of o_proj) of chosen heads with the value
from the other wording's run on the same item.
  look_again <- neutral : does it bring back the right answer (broken items) and keep kept items kept?
  neutral <- look_again : does it induce breaks on own-right items?
Conditions: top8 / top16 (selected heads), rand8_layers (8 random heads from the late layers, 10 draws), rand8_any (10 draws),
all_late (every head in the late layers). Also runs on own-wrong items (selectivity cost).
Usage: python3 scripts/interp/patch_gemma2.py pairs.jsonl heads.json OUT.json [--model HF_ID --late-start 24 --late-end 36]
"""
import argparse, json, random
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
MODEL, LETTERS = "google/gemma-2-9b-it", "abcde"

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("pairs"); ap.add_argument("heads"); ap.add_argument("out"); ap.add_argument("--limit", type=int); ap.add_argument("--model", default=MODEL)
    ap.add_argument("--late-start", type=int, default=24); ap.add_argument("--late-end", type=int, default=36)   # inclusive; Gemma 24-36 of 42 layers; Llama use 18-27 of 32 (same fractions)
    a = ap.parse_args()
    H = json.load(open(a.heads)); heads16 = [tuple(x) for x in H["heads"]]; top = {"top8": heads16[:8], "top16": heads16}
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()
    cfg = model.config; NL, NH = cfg.num_hidden_layers, cfg.num_attention_heads; HD = getattr(cfg, "head_dim", None) or cfg.hidden_size // NH
    lid = [tok(l, add_special_tokens=False)["input_ids"][-1] for l in LETTERS]
    state = {"mode": None, "cache": {}, "patch": {}}          # mode: None | "cache" | "patch"
    def make_hook(layer):
        def hook(mod, args):
            x = args[0]
            if state["mode"] == "cache":
                state["cache"][layer] = x[0, -1].view(NH, HD).clone()
            elif state["mode"] == "patch" and layer in state["patch"]:
                x = x.clone(); v = x[0, -1].view(NH, HD)
                for h in state["patch"][layer]: v[h] = state["src"][layer][h]
                x[0, -1] = v.view(-1); return (x,)
        return hook
    for l, layer in enumerate(model.model.layers): layer.self_attn.o_proj.register_forward_pre_hook(make_hook(l))
    def ids_of(msgs):
        return tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + "choice_", add_special_tokens=False, return_tensors="pt").input_ids.cuda()
    @torch.no_grad()
    def probs(ids): return torch.softmax(model(ids, logits_to_keep=1).logits[0, -1, lid].float(), -1).cpu().numpy()
    def run(ids, mode=None, patch=None, src=None):
        state.update(mode=mode, patch=patch or {}, src=src); p = probs(ids); state["mode"] = None; return p
    def as_patch(hs):
        out = {}
        for l, h in hs: out.setdefault(l, []).append(h)
        return out
    rng = random.Random(1684)
    layer_pool = [(l, h) for l in range(a.late_start, a.late_end + 1) for h in range(NH)]; any_pool = [(l, h) for l in range(NL) for h in range(NH)]
    conds = dict(top)
    for i in range(10): conds[f"rand8_layers_{i}"] = rng.sample(layer_pool, 8); conds[f"rand8_any_{i}"] = rng.sample(any_pool, 8)
    conds["all_late"] = layer_pool
    items = [json.loads(l) for l in open(a.pairs) if l.strip()]
    keep = set(H["test_own_right_keys"]) | set(H["own_wrong_keys"])
    items = [it for it in items if it["key"] in keep and "messages_look_again" in it and "messages_neutral" in it][: a.limit]
    res = []
    for n, it in enumerate(items):
        ids = {w: ids_of(it[f"messages_{w}"]) for w in ("neutral", "look_again")}
        right = LETTERS.index(it["right_code"][-1]); own = LETTERS.index(it["own_code"][-1])
        caches = {}
        for w in ids: run(ids[w], "cache"); caches[w] = dict(state["cache"])
        r = {"key": it["key"], "own_correct": it["own_correct"], "right": right, "own": own,
             "base_neutral": run(ids["neutral"]).tolist(), "base_look_again": run(ids["look_again"]).tolist(), "la_patched": {}, "n_patched": {}}
        for c, hs in conds.items():
            r["la_patched"][c] = run(ids["look_again"], "patch", as_patch(hs), caches["neutral"]).tolist()        # neutral heads into look_again
            if c in ("top8", "top16") or c.endswith("_0") or c == "all_late":
                r["n_patched"][c] = run(ids["neutral"], "patch", as_patch(hs), caches["look_again"]).tolist()    # look_again heads into neutral
        res.append(r)
        if n % 20 == 0: print(n, len(items), flush=True)
    json.dump({"heads": H["heads"], "items": res}, open(a.out, "w"))
    print("saved", a.out, len(res))

if __name__ == "__main__":
    main()
