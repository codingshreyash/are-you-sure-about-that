# Design: should a model look again only when it is unsure?

Status: design written 2026-11-23, **before any run**. Plain style. Any change after we see data is marked as such.

## Question
"Look again" helps when the first answer is probably wrong and hurts when it is probably right (second-look results). A cheap signal might tell the two cases apart. **Does re-asking only when the model is unsure beat always re-asking, never re-asking, and a plain vote?**

## Data
- Real questions: MedQA (600) and CommonsenseQA (600). Files: `data/scenarios/realworld/`.
- Models (all through the Inference hub): Haiku 5.5, GPT-5.4 Mini, Gemma 4 31B, Llama 3.1 8B, Mistral 7B, Gemma 2 9B. Results are reported per model. We do not pool models.
- We reuse the turn-1 answer (`a1`) and the existing re-ask results from `results/realworld/`.

## New calls (per model)
1. **Two extra samples** `a2`, `a3` for every valid item: the same turn-1 prompt, same option order, new independent calls. Default sampling, as in the earlier resample baseline.
2. **Re-ask with the look_again wording for the own-right items we have not re-asked yet.** The earlier run re-asked all own-wrong items but only a random sample of at most 100 own-right items per pool. Policies need a re-ask outcome for every item, so we fill the rest. Same prompt builder (`reask_msgs`), same seed and option order.

Cost: about 2 x 1,200 + (own-right items without a re-ask) calls per model.

## The signal
`unsure` = the three first-turn answers `a1, a2, a3` are not all the same option.

## Policies (per item, the final answer)
| Policy | Final answer |
| --- | --- |
| never | `a1` |
| always_LA | the look_again re-ask of `a1` |
| gated_LA | the look_again re-ask if `unsure`, else `a1` |
| vote3 | the most common of `a1, a2, a3`; a three-way split keeps `a1` |
| gated_vote | `vote3` if `unsure`, else `a1` (same as vote3; shown only to check) |
| oracle_LA | the re-ask if `a1` is wrong, else `a1` (upper bound, not a real policy) |

`vote3` is the key baseline. If gating only matches voting, the re-ask adds nothing.

## Metrics and tests
- **Accuracy** of each policy per model and per pool. Paired bootstrap over items (2,000 resamples, seed 1684). Items are the unit.
- **Signal quality:** P(`a1` wrong | unsure) and P(`a1` wrong | not unsure); AUC of `unsure` for "`a1` is wrong"; the share of items that are unsure.
- **Gating gain:** `gated_LA − never`, `gated_LA − always_LA`, `gated_LA − vote3`.
- Items where a re-ask or a sample is unparsed are dropped from all policies for that model, and we report the count.

## Predictions (written before the run)
- **P1.** For Gemma 2 9B, `always_LA` is below `never`, and `gated_LA` is at or above `never`. The model is confident when it is right, so gating avoids the breaks.
- **P2.** For Llama 3.1 8B and Mistral 7B, `unsure` rarely fires (low-variance sampling), so `gated_LA` stays near `never`. Mistral repeats its answer in most neutral re-asks.
- **P3.** For Haiku 5.5 and GPT-5.4 Mini, `gated_LA` is at or above `always_LA`.
- **P4.** `vote3` is at least as good as `gated_LA` for most models. We expect no clear win for gating over voting.

## Verdict rule (per model)
- **Gating helps:** `gated_LA − never` has a 95% interval wholly above 0, and `gated_LA − always_LA` has an interval not wholly below 0.
- **Gating beats voting:** `gated_LA − vote3` has an interval wholly above 0.
- **Uninformative:** fewer than 30 unsure items, or an interval wider than 0.06 on the gain. We state this instead of a pass or fail.
- **Overall claim:** we say "a cheap signal helps" only if gating helps in at least 3 of 6 models. If gating never beats voting, we report that the signal helps only as much as sampling more, and the re-ask adds nothing beyond it.

## Cautions we state in advance
- Default API sampling sets how often `unsure` fires. A model with near-deterministic output gives no signal.
- `a1` is the same turn-1 answer used by the earlier runs, so `a1` is not independent of the earlier sample statistics; `a2` and `a3` are new calls.
- Items are real questions but MedQA and CommonsenseQA are not advice. CommonsenseQA has label noise.
- The oracle row is an upper bound only.
- Six models, one question family. No claim beyond it.
