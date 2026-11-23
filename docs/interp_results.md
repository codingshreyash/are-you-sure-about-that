# Interpretability results: Gemma 2 9B IT, E0 to E2

Run 2026-11-22. One model, one real-question set (MedQA and CommonsenseQA, 453 items). All numbers are exploratory. Plan: [`docs/interp_plan.md`](interp_plan.md). Raw numbers: `results/interp/`. Scripts: `scripts/interp/`.

## Setup
- Gemma 2 9B IT, local, bf16, eager attention, one A100 80 GB. The run took about 2 minutes.
- The prompts are rebuilt from the scenario files and the same chat as the API run: turn 1 question, the model's own turn-1 answer, then the re-ask. We compare the **neutral** re-ask with the **look_again** re-ask. They differ by one sentence.
- Answer forcing: we start the model's reply with `choice_` and read the probabilities of the letters a to e. The "forced answer" is the most likely letter.
- "Kept" and "broken" are defined on the 100 own-right items by the **local forced answer** under look_again. This gives 73 kept and 27 broken.

## E0 gate: passed, with a caveat
| | neutral | look_again | API run (sampled text) |
| --- | --- | --- | --- |
| Own-right items turned wrong (n=100) | 0.02 | **0.27** (95% CI 0.18 to 0.36) | 0.07 and 0.51 |
| Own-wrong items turned right (n=353) | 0.06 | **0.35** | 0.07 and 0.38 |

- The pattern reproduces: look_again breaks far more right answers than neutral does.
- Forced answers break fewer right answers than sampled text (0.27 against 0.51). Forcing the first tokens changes the behavior. Local and API final answers agree on right or wrong for 83% of items.

## E2: where attention goes
Attention from the last position to the model's own turn-1 answer, per head.

- **Under look_again, heads in layers 24 to 36 attend much less to the model's own answer on items that get broken.** Example: L30H09 gives 0.055 on broken items and 0.176 on kept items. The 12 strongest heads all have |t| above 5 (uncorrected, permutation p about 0.002, ranking only).
- **The same heads lose attention when we switch from neutral to look_again.** Example: L31H15 goes from 0.415 to 0.168.
- **The score generalizes to held-out items.** We pick the top 8 heads on half of the own-right items and score the other half, over 200 random splits. The AUC for telling broken from kept is **0.92** (5th to 95th percentile 0.86 to 0.97). The same heads are chosen again and again (L30H9 in 192 of 200 splits, L34H14 in 183, L31H15 in 158).
- **The drop in attention (look_again minus neutral) is a weaker predictor**: held-out AUC 0.78. Under neutral alone, the same heads give an AUC of 0.64.
- **The same drop goes with recovery on wrong items.** On own-wrong items, the drop is larger when the answer gets fixed (-0.214 against -0.149). This fits the picture that "look again" loosens the hold on the model's own answer. E3 below shows these heads are markers, not the cause.
- Some heads in layers 14 to 22 (L19H05, L22H11, L14H01) attend more to the re-ask sentence under look_again.

## E1-lite and the direction check (weak evidence)
- A linear probe on the last-position residual predicts broken versus kept with a cross-validated AUC of 0.88 to 0.89 at layers 30 to 39 under look_again, and about 0.7 under neutral.
- **This is close to circular.** The last-position residual in late layers is what produces the answer. We do not read it as evidence of a separate "doubt" signal.
- The direction check (mean look_again minus neutral) shows larger projections on broken items at layers 24 to 39. It is the base for E4, and no steering has run.

## E3: patching (causal test)
Same prompts and answer forcing. At the **last position only**, we replace the per-head attention output of chosen heads in the look_again run with the value from the neutral run on the same item. Heads were chosen on one half of the own-right items. We test on the other half (50 items: 15 broken, 35 kept) and on 100 random own-wrong items. Random controls: 10 draws of 8 heads from layers 24 to 36, and 10 draws from all layers. Numbers: `results/interp/gemma2_e3_summary.txt`.

| Patch (neutral into look_again) | Broken items made right again (n=15) | Kept items still right (n=35) | Own-wrong items right (n=100, baseline 0.45) |
| --- | --- | --- | --- |
| Top 8 heads (E2 heads) | **0.00** | 1.00 | 0.46 |
| Top 16 heads | **0.40** (CI 0.13 to 0.67) | 1.00 | **0.27** |
| All heads in layers 24 to 36 | **0.87** (CI 0.67 to 1.00) | 0.97 | **0.09** |
| 8 random heads, layers 24 to 36 (10 draws) | 0.05 (range 0.00 to 0.40) | 0.99 | 0.45 |
| 8 random heads, any layer (10 draws) | 0.01 | 0.99 | 0.45 |

Reverse patch (look_again heads into the neutral run, own-right items): top 8 break rate 0.06, top 16 0.06, all heads in layers 24 to 36 **0.24** (neutral baseline 0.04).

What this says:
- **The heads with the clearest attention pattern in E2 are markers, not the cause.** Patching them alone restores 0 of 15 answers. High attention to the own answer does not mean the head's output decides the answer.
- **The effect is spread over late layers.** Patching every head in layers 24 to 36 restores 87% of broken answers. The top 16 heads restore 40%, well above random sets (5%). Eight heads alone do nothing, so the top 16 work together or the ranks 9 to 16 matter. We did not test them one by one.
- **The same components carry the useful and the harmful change.** The patch that undoes the breaks also removes real recoveries (0.45 to 0.09 for all heads, 0.45 to 0.27 for the top 16). This matches the Ma et al. warning that "do not yield" and "update when wrong" share components. We have no head set that keeps the fixes and removes the breaks.
- **Late-layer head outputs are not the whole story.** Swapping them into the neutral run causes breaks in 24% of right answers, not 27%+. Earlier layers and earlier positions also matter. We patched the last position only.

## What these results do not show
- **E2 and E1-lite are correlations.** E3 is causal only for the last-position head outputs we patched. It has 15 broken items, so the intervals are wide.
- **One model and one question set.** Llama and Mistral (anchoring) have not run.
- **Own-right sample is 100 items,** with 27 broken. The head selection used 2 passes over the same items; the held-out split reduces, but does not remove, that risk.
- The forced-answer metric is not the same as sampled text.
- The own-right group is a random sample of up to 100, so confidence intervals are wide.
- Hidden states are stored in float16. We found no overflow.

## Next
1. Find which layers and which positions matter: patch by layer block, and patch the re-ask span, not only the last position.
2. Repeat E0 to E2 for Llama 3.1 8B (anchoring) and Mistral 7B.
3. Try a second wording ("review").
