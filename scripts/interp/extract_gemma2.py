"""E0-E2 extraction for Gemma 2 9B IT (plan: docs/interp_plan.md). NEEDS A GPU. NOT YET TESTED.

For each item and wording (neutral, look_again) from data/interp/*_pairs.jsonl:
  - build the chat with the tokenizer template and record token spans: question (turn-1 user), own answer, re-ask sentence
  - answer forcing: prefill the model turn with "choice_" and read next-token probabilities over a..e (metric used in E0/E1)
  - residual stream at the last position, every layer (E1 probes, E4 direction)
  - attention mass from the last position to each span, per layer and head (E2)
Usage: python3 scripts/interp/extract_gemma2.py PAIRS.jsonl OUT.npz [--limit N] [--model HF_ID]  (default google/gemma-2-9b-it; works for other chat models, e.g. Llama 3.1 8B Instruct)
Needs: torch, transformers, a Hugging Face token with access to google/gemma-2-9b-it. Eager attention is required for attention output.
"""
import argparse, json
import numpy as np, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "google/gemma-2-9b-it"
WORDS = ["neutral", "look_again"]
LETTERS = "abcde"

def text_of(tok, msgs, gen=False):
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen)

def ntok(tok, text):
    return len(tok(text, add_special_tokens=False)["input_ids"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs"); ap.add_argument("out"); ap.add_argument("--limit", type=int); ap.add_argument("--model", default=MODEL)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, attn_implementation="eager").cuda().eval()
    letter_ids = [tok(f"{l}", add_special_tokens=False)["input_ids"][-1] for l in LETTERS]
    items = [json.loads(l) for l in open(a.pairs) if l.strip()]
    items = [it for it in items if all(f"messages_{w}" in it for w in WORDS)][: a.limit]
    out = {"keys": [], "own_code": [], "right_code": [], "own_correct": [], "api_outcome_neutral": [], "api_outcome_look_again": []}
    P, H, A = {w: [] for w in WORDS}, {w: [] for w in WORDS}, {w: [] for w in WORDS}
    for it in items:
        out["keys"].append(it["key"]); out["own_code"].append(it["own_code"]); out["right_code"].append(it["right_code"])
        out["own_correct"].append(it["own_correct"])
        for w in WORDS: out[f"api_outcome_{w}"].append(it[f"outcome_{w}"])
        for w in WORDS:
            msgs = it[f"messages_{w}"]
            full = text_of(tok, msgs, gen=True) + "choice_"           # answer forcing
            ids = tok(full, add_special_tokens=False, return_tensors="pt").input_ids.cuda()  # template already has BOS
            q_end = ntok(tok, text_of(tok, msgs[:1]))                 # turn-1 user ends here
            a_end = ntok(tok, text_of(tok, msgs[:2]))                 # own answer ends here
            sentence = msgs[2]["content"].split("\n\n")[0]; cut = full.rindex(sentence) + len(sentence)   # end of the re-ask sentence, in characters
            offs = tok(full, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
            sent_end = sum(1 for (st, _) in offs if st < cut)                                          # header tokens + sentence tokens
            spans = {"question": (1, q_end), "own_answer": (q_end, a_end), "reask": (a_end, sent_end)}  # skip BOS at 0
            with torch.no_grad():
                o = model(ids, output_hidden_states=True, output_attentions=True)
            P[w].append(torch.softmax(o.logits[0, -1, letter_ids].float(), -1).cpu().numpy())
            H[w].append(torch.stack([h[0, -1] for h in o.hidden_states]).float().cpu().numpy().astype(np.float16))
            A[w].append(np.stack([[at[0, :, -1, s:e].sum(-1).float().cpu().numpy() for (s, e) in spans.values()]
                                  for at in o.attentions]))          # layers x spans x heads
            del o
    np.savez_compressed(a.out, spans=np.array(["question", "own_answer", "reask"]),
                        **{k: np.array(v) for k, v in out.items()},
                        **{f"probs_{w}": np.array(P[w]) for w in WORDS},
                        **{f"hidden_{w}": np.array(H[w]) for w in WORDS},
                        **{f"attn_{w}": np.array(A[w]) for w in WORDS})
    print("saved", a.out, len(items), "items")

if __name__ == "__main__":
    main()
