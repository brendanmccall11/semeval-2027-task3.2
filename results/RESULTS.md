# Results

All Swedish numbers use the official metric: per-word macro-F1 (mean of Yes-F1 and No-F1), averaged over the 10 Swedish dev words (1,983 usages). Unless stated otherwise they come from nested leave-one-word-out cross-validation, so each word's cutoff is chosen without that word's labels. The Swedish data has been used for many decisions, so these numbers still carry some selection bias.

Every experiment after round 1 was judged by a rule written down before it ran.

## Swedish development, macro-F1

| System | Macro-F1 | Notes |
|---|---:|---|
| Gemma 4 31B, no training | 0.6818 | |
| Round 1: Gemma + LoRA, seed 2027 | 0.7529 | +0.071 vs no training (95% range +0.034 to +0.109) |
| Round 2: more training languages (15,610 examples) | 0.7124 | rejected |
| Seed 2028 | 0.7741 | same recipe as round 1 |
| Seed 2029 | 0.7362 | same recipe |
| Seed 2030 | 0.7240 | same recipe |
| Seed 2031 | 0.7557 | same recipe |
| Single model, mean of 5 seeds | 0.7486 ± 0.0192 | spread caused by the random seed alone |
| Average of 3 seeds (2027 to 2029) | 0.7664 | |
| **Average of 5 seeds (2027 to 2031), final** | **0.7789** | adopted on principle, decided before training |
| Option A: training on 5 Swedish dev words per fold | 0.7351 | rejected |
| Per-word cutoff (score minus word median) | 0.6479 | rejected |
| Per-word cutoff (score minus half the word median) | 0.7459 | rejected |
| Option and cutoff chosen inside each fold | 0.7664 | every fold chose the single cutoff |
| Qwen3-32B, same recipe, alone | 0.6558 | |
| Qwen3-32B + average of 3, averaged | 0.7380 | rejected |

Average of 5 minus average of 3: +0.0124 (95% range +0.0051 to +0.0207). Better on 7 words, unchanged on 2, worse on 1 (kvinna, -0.002).

### Per word, macro-F1

| Word | Average of 3 | Average of 5 |
|---|---:|---:|
| dumpa | 0.655 | 0.695 |
| exportera | 0.853 | 0.872 |
| fru | 0.586 | 0.586 |
| fröken | 0.662 | 0.668 |
| förort | 0.887 | 0.887 |
| herre | 0.727 | 0.747 |
| kvinna | 0.749 | 0.746 |
| motiv | 0.961 | 0.966 |
| skär | 0.716 | 0.735 |
| suga | 0.870 | 0.886 |

### Cutoffs

Fitted on all 10 Swedish words (in-sample), predeclared grid from -100 to 100 in steps of 0.25, ties go to the lower cutoff.

| System | Cutoff | In-sample macro-F1 |
|---|---:|---:|
| Seed 2027 alone (backup) | 0.75 | 0.7643 |
| Average of 3 | 0.75 | |
| Average of 5 (main) | 1.00 | 0.7789 |

## External checks for the other languages

Datasets we built, not competition data. All are 50% Yes by construction, unlike the real data (Swedish words range from 6% to 95% Yes). Macro-F1 here is pooled over all pairs; each system uses its own Swedish cutoff. Do not average these with the Swedish score.

| Data | Pairs | Seed 2027 mF1 | Avg of 3 mF1 | Avg of 5 mF1 | Seed 2027 AUC | Avg of 3 AUC | Avg of 5 AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| EN, WiC-TSV dev | 389 | 0.8766 | 0.8740 | 0.8817 | 0.9339 | 0.9346 | 0.9351 |
| RU, AXOLOTL-derived | 3,092 | 0.8698 | 0.8786 | 0.8727 | 0.9475 | 0.9499 | 0.9502 |
| IT, XL-WSD-derived | 1,660 | 0.7237 | 0.7381 | 0.7235 | 0.8702 | 0.8755 | 0.8756 |
| ES, XL-WSD-derived | 936 | 0.7073 | 0.7027 | 0.6866 | 0.8262 | 0.8267 | 0.8254 |
| NL, XL-WSD-derived | 1,980 | 0.6900 | 0.6962 | 0.6907 | 0.8051 | 0.8056 | 0.8065 |

AUC is unchanged across the three systems. The small macro-F1 drops for IT/ES/NL with the average of 5 come from the higher cutoff (1.00 vs 0.75) on these always-50% sets, and all are within the 0.02 limit set before the run.

Caveats:
- XL-WSD pairs use modern sentences and machine-translated WordNet definitions. "No" pairs are other dictionary senses of the same word, and some are very close in meaning to the correct one.
- Dates in the XL-WSD prompts are dataset years, not verified publication dates.

## What did not work, and why (short)

- **More training languages (round 2) and Swedish dev training (option A):** both lowered Swedish. Adding data mostly shifted the model's Yes/No balance rather than teaching anything that transfers to new words.
- **Per-word cutoffs:** a word's average score tracks its true Yes share (for example exportera, 95% Yes, scores +3.6; fröken, 6% Yes, scores -2.8), so removing it throws away information.
- **Qwen3-32B in the ensemble:** different enough from Gemma (correlation 0.816) but much weaker on historical Swedish (0.656 alone, below untrained Gemma), so it pulled the average down.

## Earlier results under the old metric

Before the organizers confirmed macro-F1, results were reported as mean per-word positive-class F1. Those numbers (for example 0.7043 for round 1) are kept in `docs/PROJECT_STATE.md` for history and are not comparable with the table above.
