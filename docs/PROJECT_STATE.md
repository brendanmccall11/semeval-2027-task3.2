# Project state (8 November 2026)

## Scope and evidence

SemEval-2027 Task 3, Subtask 2: classify whether a target word's usage expresses a supplied sense definition. Goal: competitive performance across SV, EN, IT, ES, NL, RU using open-weight models, no paid model APIs. Experiments ran on Google Colab with A100 80 GB GPUs.

Colab filesystem root: `/content/drive/MyDrive/semeval2027/semeval2027_colab`.

## Current primary system

- Base: `google/gemma-4-31B-it`.
- Pinned revision: `842da3794eaa0b77d5f08bae87a17459d91ff475`.
- Actual completed adaptation: **BF16 LoRA**, not QLoRA.
- 45 million trainable parameters; rank 16, alpha 32, dropout 0.05; language-model q/k/v/o projections only.
- Answer-token cross-entropy; score is next-token logit(Yes) minus logit(No), thinking disabled.
- Training: 8,510 examples, RU 6,078 / EN 2,432; 4,275 Yes / 4,235 No. No direct SV/IT/ES/NL adaptation data in this pilot.
- Microbatch 1, effective batch 16, learning rate 1e-4, 27 warmup updates and linear decay. One epoch, 532 optimizer updates.
- Checkpoints 133, 266, 532 assessed on Russian selection data (2,946 examples). Step 532 selected; selection F1 0.875328, threshold -0.25. Original model selection F1 ~0.8680 at threshold 9.0.
- Trial: 20 updates / 320 examples, finite gradients/loss, peak allocated VRAM 59.3 GiB. Full run restarted from a fresh adapter, without merging trial weights.
- Frozen primary config: `outputs/frozen/primary_gemma4_31b_lora.json`.
- Adapter: `outputs/runs/step09_lora_full_pilot/20261105T232211Z_6b04a51b/checkpoint_0532/adapter`.
- Reported/checked adapter hash prefix: `6f3d236ac50e07cc`.
- Original Gemma remains the fallback.

### Thresholds must remain explicit

1. Swedish nested word-CV result: thresholds fitted excluding the held-out word, generally 0.25 or 0.75. Reported F1 0.704306.
2. Final Swedish cutoff fitted using all Swedish development labels: +0.25. Its in-sample F1 is ~0.7148; not the held-out estimate.
3. EN/RU primary external comparison: original threshold 9.0, adapted threshold -0.25, both calibrated on Russian selection.
4. IT/ES/NL notebook 11 comparison: original threshold 10.5, adapted threshold +0.25, transferred from Swedish. These are different thresholds from the EN/RU primary comparison.

Never silently use +0.25 to claim reproduction of the EN/RU numbers obtained at -0.25.

## Completed evaluation results

| Language / dataset | N pairs | Original F1 | Adapted F1 | Original AUC | Adapted AUC |
|---|---:|---:|---:|---:|---:|
| SV competition development | 1,983 | 0.678239 | 0.704306 | n/a | n/a |
| EN WiC-TSV development | 389 | 0.870813 | 0.881517 | 0.909659 | 0.933920 |
| RU AXOLOTL-derived test | 3,092 | 0.867597 | 0.875703 | 0.921908 | 0.947497 |
| IT XL-WSD-derived | 1,660 | 0.7718 | 0.7448 | 0.8615 | 0.8702 |
| ES XL-WSD-derived | 936 | 0.7040 | 0.6968 | 0.8043 | 0.8262 |
| NL XL-WSD-derived | 1,980 | 0.6900 | 0.6944 | 0.7812 | 0.8051 |

SV is mean per-word F1 under the official-compatible metric; the external columns are pooled binary F1. Datasets, construction, and thresholds differ. Do not average these rows into a competition score.

SV improvement +0.0261, paired 95% interval [-0.0140, +0.0709]. EN improvement +0.0107, interval [-0.0101, +0.0321]. RU improvement +0.0081, interval [-0.0011, +0.0174]. All include zero. The Swedish set has been reused across many development decisions, so nested threshold fitting does not remove overall experiment-selection bias.

Notebook 11 reports positive AUC differences for all three new languages (IT +0.0086; ES +0.0219; NL +0.0240) and prints `SAFE` based on a no-more-than-0.02 AUC loss rule. That wording is too strong: Italian F1 fell 0.0270 and Spanish F1 fell 0.0072 at the frozen cutoffs. AUC improvement does not establish F1 improvement.

## Newly verified work in notebook 11

XL-WSD download and inspection, derived evaluation construction, original-model scoring, adapter scoring, and comparison are **already complete** in its saved outputs.

- Archive: 146.5 MiB / 191 files.
- Evaluation inputs: `outputs/runs/step11_xlwsd_evaluation_sets/20261106T032028Z_aa4e2120`.
- Score outputs: `outputs/runs/step11_xlwsd_scores/adapter_6f3d236ac50e`.
- IT: 830 source instances, 343 lemmas, 1,660 constructed balanced pairs.
- ES: 468 source instances, 169 lemmas, 936 constructed balanced pairs.
- NL: 990 source instances, 655 lemmas, 1,980 constructed balanced pairs.
- Gloss construction uses a headword before a colon in translated WNGT gloss sentences and its tagged BabelNet sense ID. Initial attribute-based mapping found zero definitions; the headword-based method produced coverage.
- The preliminary headword/inventory consistency checks were 67.4% IT, 64.0% ES, 56.6% NL; a printed aspirational `90%+` message was not the actual result. These checks and mapping quality warrant review.
- Negative pairs use other inventory senses; overlapping glosses can make negatives ambiguous. Saved examples include near-overlapping definitions. Do not describe the derived binary labels as independently human-verified.
- Dataset event/source years (2010/2013/2015) were used as prompt dates; all Dutch examples default to 2010. These are not verified publication dates. These sets cannot establish historical-period robustness.
- Scoring used Transformers 5.22.0 rather than the earlier 5.22.1. Both original and adapted variants were scored in the same notebook session. The notebook removed torchao to resolve compatibility and loaded the frozen adapter with a matching hash.

## Completed earlier experiments

- Registration, data inspection, project organization and shared evaluation harness.
- Official Swedish development: 10 words, 1,983 usages, 705 positives, five target-offset repairs.
- Baselines: all-positive 0.471630, E5 raw 0.452178, E5 standardized 0.412366, majority 0.0; official scorer agreement recorded.
- English graded-wsd external baseline: pooled F1 ~0.8096; not a competition score.
- Qwen3-8B raw 0.462565 / standardized 0.476132.
- Original Gemma 0.678239; standardized Gemma 0.598939; revised prompt 0.659720.
- Qwen3.8-27B ~0.644679; earlier Gemma/Qwen combinations did not establish a reliable improvement.
- Dictionary-assisted Gemma ~0.684428; uncertain improvement over original.
- XL-DURel usage grouping: original majority-label ceiling corrected to F1-optimal cluster-constant assignment. Actual score-smoothing hybrid 0.675521; no demonstrated benefit.
- Fine-tuning pilot succeeded and checkpoint 532 adopted under the project's practical rule. This adoption is not a statistical-significance claim.
- Post-adaptation EM word-prior/approximate expected-F1 decoding: **0.595631**, rejected. Severe incorrect positive-rate estimates for herre/suga/dumpa. Run: `step10_word_adaptive_decisions/20261106T025056Z_b1cea70c`.
- Post-adaptation original-plus-adapted combination: **0.704306**, no improvement; every fold chose weight 1.0 on trained Gemma. Run: `step10_original_plus_trained/20261106T025756Z_085b1d15`.

## Main artifact references

All relative paths below are under the canonical Colab project root.

- Original model scores: `outputs/runs/step04_gemma4_31b/20261105T164154Z_48096e67/gemma_raw_scores.json`.
- Original SV evaluation: `outputs/runs/step04_gemma4_evaluation/20261105T170910Z_b3fc303a`.
- Frozen word lists: `outputs/runs/step07_frozen_splits/20261105T220410Z_0c2b5b52/splits.json`.
- Pilot train/selection: `outputs/runs/step07_pilot_dataset/20261105T223146Z_1dd5a63b`.
- EN/RU evaluation: `outputs/runs/step07_pilot_evaluation/20261105T223700Z_5259f45a`.
- Original EN/RU baseline: `outputs/runs/step08_gemma_unchanged_baseline/20261105T224131Z_63bba2d9`.
- Completed pilot/adaptation outputs: `outputs/runs/step09_lora_full_pilot/20261105T232211Z_6b04a51b`.

## What remains and how to proceed

Model selection and the first six-language comparison are done. The main open question from notebook 11 is the cutoff: the LoRA model ranks better in every language (AUC), but at the Swedish cutoff its F1 dropped for Italian and Spanish. Before any further training, we should check the quality of the derived multilingual sets and decide how to set cutoffs for other languages. The cutoff should not be tuned on these external evaluation labels, and the failed EM and score-mixing experiments do not need to be repeated.

Substantive unfinished work includes reliable multilingual/historical evaluation, possible improved multilingual training if justified, period-level analysis, consolidated report/presentation, final reproducibility/package checks, and official submission. There is no official test score yet; the test window is January 22 to 31, 2027.
