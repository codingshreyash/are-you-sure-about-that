"""Generation-based E0/E2 extraction (plan: docs/interp_plan.md; fixes the forced-answer problem found for Llama). NEEDS A GPU. NOT YET TESTED.

For each item and wording (neutral, look_again):
  1. Generate the model's real re-ask reply locally (greedy, up to --max-new tokens).
  2. Parse the final code (last `choice_x` in the reply). Replies without a code are dropped and counted.
  3. Teacher-force prompt + reply up to and including the last `choice_`. Read at that last position: probabilities over a..e,
     residual stream (every layer), attention mass to spans: question, own_answer, reask, generated (the model's new reasoning).
Outcome labels (gen_ok_*) come from the generated code, so they track the API behavior.
Also writes OUT.texts.jsonl (the teacher-forced text per item and wording) for patch_gemma2.py --texts.
Items: all own-right items plus --n-wrong random own-wrong items (seed 1684).
Usage: python3 scripts/interp/gen_extract.py PAIRS.jsonl OUT.npz --model HF_ID [--n-wrong 200] [--batch 24] [--limit N]
"""
import argparse, json, random, re
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
WORDS, LETTERS = ["neutral", "look_again"], "abcde"
CODE = re.compile(r"choice_([a-e])")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs"); ap.add_argument("out"); ap.add_argument("--model", required=True)
    ap.add_argument("--n-wrong", type=int, default=200); ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--max-new", type=int, default=320); ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model); tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()
    lid = [tok(l, add_special_tokens=False)["input_ids"][-1] for l in LETTERS]
    items = [json.loads(l) for l in open(a.pairs) if l.strip()]
    items = [it for it in items if all(f"messages_{w}" in it for w in WORDS)]
    right_items = [it for it in items if it["own_correct"]]
    wrong_items = [it for it in items if not it["own_correct"]]
    random.Random(1684).shuffle(wrong_items)
    items = (right_items + wrong_items[: a.n_wrong])[: a.limit]
    prompts = {w: [tok.apply_chat_template(it[f"messages_{w}"], tokenize=False, add_generation_prompt=True) for it in items] for w in WORDS}
    gen = {w: [None] * len(items) for w in WORDS}
    for w in WORDS:                                   # 1. generation, batched, longest prompts last for stable padding
        order = sorted(range(len(items)), key=lambda i: len(prompts[w][i]))
        for s in range(0, len(order), a.batch):
            idx = order[s: s + a.batch]
            enc = tok([prompts[w][i] for i in idx], add_special_tokens=False, return_tensors="pt", padding=True).to("cuda")
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=a.max_new, do_sample=False, pad_token_id=tok.pad_token_id)
            for j, i in enumerate(idx):
                gen[w][i] = tok.decode(out[j, enc.input_ids.shape[1]:], skip_special_tokens=True)
            print(w, s + len(idx), len(order), flush=True)
    keep, texts = [], []
    for i, it in enumerate(items):                    # 2. parse, 3. teacher-forced pass
        row, ok_all = {}, True
        for w in WORDS:
            ms = list(CODE.finditer(gen[w][i]))
            if not ms: ok_all = False; break
            m = ms[-1]; row[w] = (m.group(1), prompts[w][i] + gen[w][i][: m.end() - 1])   # text up to "choice_" (letter excluded)
        if ok_all: keep.append(i); texts.append(row)
    print("items", len(items), "kept (both replies have a code)", len(keep))
    out = {k: [] for k in ("keys", "own_code", "right_code", "own_correct", "api_outcome_neutral", "api_outcome_look_again", "gen_choice_neutral", "gen_choice_look_again")}
    P, H, A = {w: [] for w in WORDS}, {w: [] for w in WORDS}, {w: [] for w in WORDS}
    with open(a.out + ".texts.jsonl", "w") as tf:
        for i, row in zip(keep, texts):
            it = items[i]; msgs = it["messages_look_again"]
            out["keys"].append(it["key"]); out["own_code"].append(it["own_code"]); out["right_code"].append(it["right_code"]); out["own_correct"].append(it["own_correct"])
            for w in WORDS: out[f"api_outcome_{w}"].append(it[f"outcome_{w}"]); out[f"gen_choice_{w}"].append(row[w][0])
            tf.write(json.dumps({"key": it["key"], **{w: row[w][1] for w in WORDS}}) + "\n")
            for w in WORDS:
                msgs = it[f"messages_{w}"]; full = row[w][1]
                tpl = lambda ms: tok.apply_chat_template(ms, tokenize=False)
                ntok = lambda t: len(tok(t, add_special_tokens=False)["input_ids"])
                q_end, a_end = ntok(tpl(msgs[:1])), ntok(tpl(msgs[:2]))
                prompt_len = ntok(prompts[w][i])
                sentence = msgs[2]["content"].split("\n\n")[0]; pt = prompts[w][i]; cut = pt.rindex(sentence) + len(sentence) if sentence in pt else None   # search the prompt only
                enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
                offs = enc["offset_mapping"]; sent_end = sum(1 for (st, _) in offs if st < cut) if cut else a_end + 1
                total = len(offs)
                spans = [(1, q_end), (q_end, a_end), (a_end, sent_end), (prompt_len, total - 1)]   # question, own_answer, reask, generated
                ids = torch.tensor([enc["input_ids"]]).cuda()
                with torch.no_grad():
                    o = model(ids, output_hidden_states=True, output_attentions=True)
                P[w].append(torch.softmax(o.logits[0, -1, lid].float(), -1).cpu().numpy())
                H[w].append(torch.stack([h[0, -1] for h in o.hidden_states]).float().cpu().numpy().astype(np.float16))
                A[w].append(np.stack([[at[0, :, -1, s:e].sum(-1).float().cpu().numpy() for (s, e) in spans] for at in o.attentions]))
                del o
    gen_ok = {w: np.array([c == r[-1] for c, r in zip(out[f"gen_choice_{w}"], out["right_code"])]) for w in WORDS}
    np.savez_compressed(a.out, spans=np.array(["question", "own_answer", "reask", "generated"]),
                        **{k: np.array(v) for k, v in out.items()}, **{f"gen_ok_{w}": gen_ok[w] for w in WORDS},
                        **{f"probs_{w}": np.array(P[w]) for w in WORDS}, **{f"hidden_{w}": np.array(H[w]) for w in WORDS},
                        **{f"attn_{w}": np.array(A[w]) for w in WORDS})
    print("saved", a.out, len(keep))

if __name__ == "__main__":
    main()
