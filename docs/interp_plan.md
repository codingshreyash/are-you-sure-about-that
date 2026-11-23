# Interpretability plan: why does "look again" change answers?

Status: plan only. No experiment in this file has run. Written 2026-11-21. Written in plain style.

## 1. The question

Our behavior results show three patterns when we ask a model to look again at its own answer, in the same chat:

| Pattern | Model | What we saw |
| --- | --- | --- |
| Anchoring | Llama 3.1 8B, Mistral 7B | The model stays on its own wrong answer. A fresh try fixes more. |
| Doubt flip | Gemma 2 9B | "Look again" breaks 51 of about 100 right answers on real questions. The neutral wording breaks 7. |
| Selective revision | Haiku 5.5, GPT-5.4 Mini | The model fixes errors and breaks few right answers. |

**Core question: what happens inside the model that makes these three patterns?**

We have three untested ideas. We name them so we can test them.

- **H1, anchoring.** The model copies its own turn-1 answer. Late layers attend to the answer tokens.
- **H2, doubt reading.** The model reads "look again" as "you were wrong". It attends to the re-ask words and moves away from its answer, even when the answer is right.
- **H3, selectivity.** A strong model keeps a confidence signal about its own answer. A weak model does not. The signal decides when to revise.

## 2. What others already showed (leads, not verified)

The agent read most of these through a summarizing fetch tool. Check the numbers in the original before we cite any of them.

- **Closest work:** Zhang et al., "Understanding the Dark Side of LLMs' Intrinsic Self-Correction" (arXiv 2412.14959). Llama-3.1-8B overturned 58.8% of its right answers. A tuned lens shows more internal answer changes. "Are you sure?" looks internally like "You are wrong". They did not find heads, did not patch, and did not test Gemma or Mistral. This supports H2 for Llama. Our Llama data show anchoring, not flipping, so the two results differ and we must explain that.
- **Few heads carry persuasion.** Sun et al. (2605.09314): a few mid-layer heads, for example L17H24. Patching attention patterns restores the answer in 36.3% of cases. Models include Gemma-2. The prompt is a single turn with a user claim.
- **Shared sycophancy circuit.** Pandey (2604.19117): the same heads carry "this is wrong" for lying and for sycophancy. Models include Gemma-2 9B and Mistral-7B. Single turn.
- **Truth fades from layer 19.** Wang et al. (2508.02087): the logit-lens preference for the right answer stops growing, then late layers shift it. Llama 3.1 8B and Mistral-7B among the models.
- **Yielding and updating share components.** Ma et al. (2608.26511): turning down sycophancy hurts rational updating. Two turns. The model's own answer is not manipulated.
- **The model may know more than it says.** Orgad et al. (2410.02707): a probe on the answer tokens can find the right answer when the output is wrong. Single turn.
- **Self-answer history effects.** Li et al. (2609.05882): attention knockouts on prior assistant turns, one model (Qwen3-14B). It does not study answer anchoring.
- **Other leads:** Genadi et al. (2601.16644, linear probes, sparse heads), Vennemeyer et al. (2509.21305, separate directions), Lee (2611.11035, reply length matters), Kumaran et al. (2604.22271, cached confidence signal).

### Goodfire (added 2026-11-22; pages read through a summarizing tool, check numbers in the originals)

Goodfire has nothing on sycophancy, self-correction, multi-turn chats or answer flipping. We read their full research list by title. What they give us:

- **One SAE we can use.** `Goodfire/Llama-3.1-8B-Instruct-SAE-l19` is trained on Instruct chat data, at layer 19 only. Use it for Llama. Check the reconstruction error on our own multi-turn prompts. Toxic features were removed from it.
- **Do not depend on Ember.** Their API and demo were deprecated in Feb 2026, and the docs site does not resolve. We load the weights ourselves. Feature labels came from Ember, so we label features ourselves.
- **Chat template tokens carry behavior features.** Their Llama 3 steering post says so. Keep the template, and look at the template tokens next to the re-ask.
- **Steering fails in known ways.** Turning a behavior on often works with one feature. Turning it off often needs several. Straight-line steering can push the model off its manifold ("Steering Along Manifolds", 2026). "Belief Dynamics" (2511.00617, abstract only) reports sharp thresholds when steering adds to in-context evidence. This fits "look again", an in-context cue.
- **Probe methods.** Attention probes predict the final answer early ("Reasoning Theater", 2026). Difference-of-means probes work as a baseline ("Models know when they're reward hacking", 2026). Probes on a frozen base model transferred to the tuned model ("Features as Rewards", Gemma 3 12B IT). A sycophancy direction was found in OLMo-3-7B ("Shaping and Monitoring Models", 2611.07071, page read, paper not opened).
- **A hint to test, not a fact.** A podcast claim says SAE features sometimes underperform raw-activation probes. We run our own comparison.

Their work does not cover our multi-turn own-answer contrast, so it does not close the gap in section 3.

## 3. The gap we can fill

No paper we found does all of these at once:

1. The prior answer is **the model's own**, in a real multi-turn chat. Almost all circuit work uses a user claim ("I believe X").
2. The contrast is **three behavior regimes** (anchor, flip, selective) across models. Papers study one model at a time.
3. The two prompts differ by **one sentence** (neutral vs "look again"), after the same turn-1 text. This is a very clean contrast.
4. We ask if the **right answer survives inside the model** when it flips, and if a **selectivity signal** exists.

We do not claim the gap is empty. The agent searched, but it could miss papers. We must search again before we write a novelty claim.

## 4. Models and tools

| Model | Why | Open SAEs |
| --- | --- | --- |
| Gemma 2 9B IT | Strongest effect (doubt flip). Best tool support. | Gemma Scope: every layer and sublayer, trained on the base model. Model card says they transfer to IT. The IT set has only layers 9, 20, 31. |
| Llama 3.1 8B Instruct | Anchoring. Closest prior work uses it. | Goodfire: one SAE at layer 19, trained on Instruct chat data (LMSYS-Chat-1M, mostly single-turn). Llama Scope and EleutherAI: every layer, base model only, transfer to Instruct not documented. |
| Mistral 7B Instruct v0.3 | Anchoring. | Published SAEs are for v0.1. Treat as weak. Use probes and directions, not SAEs. |
| Gemma 3 12B IT (new) | Possible open model for selective behavior, with Gemma Scope 2 (IT SAEs and transcoders). | Yes, if it shows the selective pattern in a behavior run. We do not know yet. |

Rules for the code:

- Use plain Hugging Face hooks or nnsight. Do not trust a re-implementation until its logits match Hugging Face on a few prompts.
- Use eager attention for Gemma 2.
- Use the tokenizer chat template. Keep the BOS token. Skip position 0 in SAE work. Fix the date in the Llama template.
- Patch by **span**, not by position, because the two prompts have different lengths.
- Compute last-token logits only, to save memory.
- Report SAE reconstruction error on our own prompts.

Data: we already have the rows. Each row has the turn-1 text, the re-ask wording, and the outcome. The prompts are not stored. We rebuild them from the scenario file and seed with `scripts/run/run_own_error_reask.py` (`ordered_scenario`, `chat_prompt_conversational`, `reask_msgs`; `RANDOMIZE_OPTIONS=1`, `OPTION_ORDER_SEED=1684`).

Counts we can use (re-ask stage; own-right is a sample of at most 100 per pool):

| Model, real questions | Own-wrong | Recovered with look_again | Own-right sample | Broken with look_again |
| --- | --- | --- | --- | --- |
| Gemma 2 9B | 359 | 133 | about 100 | 51 |
| Llama 3.1 8B | 476 | 84 | about 100 | 10 |
| Mistral 7B | 589 | 112 | about 100 | 20 |

Gemma 2 9B has only 14 own-wrong items on Arm B. Do not use Arm B for its recovery.

## 5. Experiments

Each experiment lists what we expect and what would prove us wrong. We write these before the run.

### E0. Setup gate (no science yet)
- Rebuild prompts for 300 Gemma 2 9B items. Check that the model, run locally, gives the same choice as the API run for most items. Local runs differ from the hub, so we may not match exactly. We need the local behavior to show the same pattern.
- Check logit match with the chosen tool.
- **Gate:** if local Gemma 2 9B does not break many right answers on "look again" and few on neutral, stop and fix before any analysis.

### E1. Does the right answer survive? (tests the "knows but defers" idea)
- Train a linear probe for "this answer code is correct" on the turn-1 context. Use only the first answers, from items outside the test set. Apply it at the final position of the re-ask prompts.
- Also run logit lens on the answer-code logits by layer.
- Split Gemma 2 9B items: right answer kept vs right answer broken. Split Llama: wrong stayed vs wrong recovered.
- **Prediction (H2/H3):** for broken right answers, the probe and logit lens still show the right option at middle layers, and it falls in late layers.
- **Falsifier:** the right answer is already absent at the middle layers. Then the flip is not a late override.

### E2. Where does attention go? (tests H1 vs H2)
- For each head, measure attention mass from the final position to three spans: (a) the model's own turn-1 answer, (b) the re-ask sentence, (c) the question and options.
- Compare look_again vs neutral on the same turn-1 text. Compare broken vs kept, and stayed vs recovered.
- **Prediction:** Gemma 2 9B puts more mass on (b) under look_again, in mid-layer heads. Llama and Mistral put more mass on (a), in later layers.
- **Falsifier:** no span shows a difference between kept and broken items. Then attention mass is not the cause.

### E3. Does patching move behavior? (the causal test)
- Take the neutral run and the look_again run on the same item. Patch the look_again attention pattern into the neutral run, one head or layer group at a time, by span. Measure the change in the answer-code logit difference (new right minus old).
- Reverse it: patch neutral patterns into look_again.
- Attribution patching first (cheap), full patching on the top heads only.
- **Prediction:** a small set of heads restores most of the effect, as Sun et al. found for persuasion.
- **Falsifier:** no small set. The effect spreads over many heads and layers. That is also a finding, but then we stop head claims.

### E4. A "doubt" direction and its cost
- Take the difference of mean activations (look_again minus neutral) at the final position, mid layers, on right-answer items. Steer neutral runs with it. Remove it from look_again runs.
- Measure both sides: breaks on right answers, and recoveries on wrong answers. Ma et al. warn that removing doubt may also stop real updating.
- Compare with a random direction of the same size.
- Sweep the steering strength in fine steps. Expect a sharp threshold, not a smooth change.
- Report off-target changes (token shifts on unrelated prompts), not only the flip rate. Test a multi-feature or direction removal, not one SAE feature.
- Also compare the probe with raw activations against SAE features.
- **Prediction:** adding the direction raises breaks under neutral. Removing it lowers breaks and lowers recoveries.
- **Falsifier:** a random direction does as well, or removal costs nothing in recoveries.

### E5. SAE features (only if E3 or E4 finds a layer)
- At the chosen layer, list the Gemma Scope features that differ most between look_again and neutral. Read their top activations on Neuronpedia and on our prompts.
- Treat labels as guesses. Test each with a feature steering run. Report reconstruction error.

### E6. Selectivity across models (tests H3)
- Behavior first: run Gemma 3 12B IT on the same real-question protocol. If it shows the selective pattern (fixes more than a fresh try, few breaks), use it. If not, drop this step and say so.
- Compare the probe from E1 across models. Does the answer-correctness signal at the final position predict which answers get revised? Strong model: yes. Weak models: no.
- **Falsifier:** the signal predicts revision equally well in weak and strong models.

### E7. Wording (optional)
- Repeat E2 and E4 for the "review" wording. It gave Gemma 2 9B 155 recoveries but also 52 breaks. Does it use the same direction as "look again"?

## 6. What a win looks like

| Outcome | What we can write |
| --- | --- |
| E1 + E3 find a small head set for Gemma 2 9B | "A few heads read 'look again' as doubt and override a right answer." Strong result. |
| Only E1 and E2 are clean | A descriptive result: where the right answer is lost, and where attention goes. We do not claim a circuit. |
| E3 finds no small set | A negative result: the effect is spread out. Still useful, and we say so. |
| E6 shows the selectivity signal in a strong model only | Support for H3. Weak evidence unless more than one strong model shows it. |
| Nothing is clean | We report the behavior study and the null interpretability result. |

## 7. Cost and order

- GPU: the agent estimates 2 to 10 GPU-hours on one 40 to 80 GB card. It did not measure this. Gemma 2 9B in bf16 is about 18 to 19 GB. Check the budget in `docs/gpu_budget.md` before we rent.
- Order: E0 (gate) → E1 and E2 together (both only read activations) → E3 → E4 → E5 → E6.
- Split for four people: one owns E0 and the prompt rebuild; one owns E1; one owns E2 and E3; one owns E4 and the write-up. Change this as needed.

## 8. Risks

- **Local vs hub behavior.** The hub may use other sampling or a different Gemma 2 build. E0 checks this.
- **SAE fidelity.** The Gemma Scope SAEs were trained on the base model. The reconstruction error on instruction-tuned activations is worse than the loss numbers suggest. Report it.
- **Templated and noisy items.** CommonsenseQA has label noise. The Arm B items are AI-drafted and AI-labeled. Use MedQA and CommonsenseQA for the main run. Do not use Arm B for Gemma 2 recovery.
- **Small and selected samples.** The own-right set is a sample of at most 100 per pool. Own-wrong items depend on the model's own errors, so each model sees different items.
- **Literature risk.** Many sources are 2026 preprints. The agent summarized them through a fetch tool. We must read the top ones in full: Zhang 2412.14959, Sun 2605.09314, Pandey 2604.19117, Ma 2608.26511.
- **Mechanism claims are easy to over-state.** Attention mass is not a cause. Only a patching result can support "cause". Use the word "cause" only for E3 and E4 results that pass their random-control check.
- **Pre-registration.** Commit this file before E0 runs. Mark any later change as made after we saw data.

## 9. Open decisions for the team

1. Rent a GPU, or use an existing one? (See `docs/gpu_budget.md`.)
2. Gemma 2 9B first, or all three models in parallel?
3. Do we add Gemma 3 12B IT as the selective-behavior model, after a behavior run?
