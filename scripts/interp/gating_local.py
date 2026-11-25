"""Local gating run (design: docs/gating_design.md, Amendment A). NEEDS A GPU. NOT YET TESTED.
For each item in data/interp/gating_items.jsonl: three independent turn-1 samples a1, a2, a3 (sampling, model default settings),
then one look_again re-ask of a1 (using a1's own text). Parse the last `choice_x`. Writes one JSON row per item.
Usage: python3 scripts/interp/gating_local.py data/interp/gating_items.jsonl OUT.jsonl --model HF_ID [--batch 48] [--limit N]
"""
import argparse, json, re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
CODE = re.compile(r"choice_([a-e])")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("items"); ap.add_argument("out"); ap.add_argument("--model", required=True)
    ap.add_argument("--batch", type=int, default=48); ap.add_argument("--max-new", type=int, default=400); ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    items = [json.loads(l) for l in open(a.items) if l.strip()][: a.limit]
    tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).cuda().eval()
    print("generation_config:", {k: getattr(model.generation_config, k, None) for k in ("temperature", "top_p", "top_k")}, flush=True)
    def gen(prompts, tag):
        outs = [None] * len(prompts); order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        for s in range(0, len(order), a.batch):
            idx = order[s: s + a.batch]
            enc = tok([prompts[i] for i in idx], add_special_tokens=False, return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad(): o = model.generate(**enc, max_new_tokens=a.max_new, do_sample=True, pad_token_id=tok.pad_token_id)
            for j, i in enumerate(idx): outs[i] = tok.decode(o[j, enc.input_ids.shape[1]:], skip_special_tokens=True)
            print(tag, s + len(idx), len(order), flush=True)
        return outs
    code = lambda t: (lambda m: m[-1].group(1) if m else None)(list(CODE.finditer(t)))
    tpl = lambda msgs: tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    p1 = [tpl(it["messages_turn1"]) for it in items]
    samples = [gen(p1, f"sample{j}") for j in (1, 2, 3)]
    a1_text = samples[0]
    p2 = [tpl(it["messages_turn1"] + [{"role": "assistant", "content": a1_text[i]}, {"role": "user", "content": it["reask_user"]}]) for i, it in enumerate(items)]
    la_text = gen(p2, "reask")
    with open(a.out, "w") as f:
        for i, it in enumerate(items):
            f.write(json.dumps({"key": it["key"], "expected": it["expected"][-1], "a1": code(samples[0][i]), "a2": code(samples[1][i]),
                                "a3": code(samples[2][i]), "la": code(la_text[i]), "model": a.model}) + "\n")
    print("saved", a.out, len(items))

if __name__ == "__main__":
    main()
