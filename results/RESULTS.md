# Results

All Swedish numbers are mean per-word F1 on the official Swedish development data (10 words, 1,983 usages, 705 positives), using nested leave-one-word-out cross-validation unless stated otherwise. The Swedish data has been used for many decisions, so these numbers carry some selection bias.

## Swedish development

| System | F1 | Notes |
|---|---|---|
| All-positive baseline | 0.4716 | |
| e5 embedding similarity | 0.4522 | per-word standardized: 0.4124 |
| Qwen3-8B, Yes/No | 0.4626 | per-word standardized: 0.4761 |
| Qwen3.8-27B, Yes/No | 0.6447 | |
| Gemma 4 31B, Yes/No (original) | 0.6782 | 95% CI 0.536 to 0.818 |
| Gemma 4 31B, alternative prompt | 0.6597 | |
| Gemma + dictionary senses | 0.6844 | difference not clear |
| Gemma + usage-group smoothing | 0.6755 | no gain |
| **Gemma 4 31B + LoRA (current)** | **0.7043** | paired change +0.026, 95% range -0.014 to +0.071 |
| LoRA + per-word EM cutoff decision | 0.5956 | rejected |
| LoRA + original Gemma score mix | 0.7043 | identical; every fold chose LoRA alone |

Per word (original -> LoRA): dumpa 0.533 -> 0.549, exportera 0.982 -> 0.982, fru 0.852 -> 0.787, fröken 0.391 -> 0.391, förort 0.872 -> 0.887, herre 0.350 -> 0.514, kvinna 0.522 -> 0.617, motiv 0.929 -> 0.929, skär 0.495 -> 0.586, suga 0.857 -> 0.800. Exportera's gain is one usage (effectively a tie).

Cutoff fitted on all Swedish labels (in-sample, not a held-out estimate): original 0.6853 at 10.5, LoRA 0.7148 at 0.25.

## External checks for the other languages

These are datasets we built, not competition data. Do not average them with the Swedish score.

| Data | Pairs | Original F1 | LoRA F1 | Original AUC | LoRA AUC | Cutoffs (original / LoRA) |
|---|---|---|---|---|---|---|
| EN, WiC-TSV dev | 389 | 0.8708 | 0.8815 | 0.9097 | 0.9339 | 9.0 / -0.25 |
| RU, AXOLOTL-derived | 3,092 | 0.8676 | 0.8757 | 0.9219 | 0.9475 | 9.0 / -0.25 |
| IT, XL-WSD-derived | 1,660 | 0.7718 | 0.7448 | 0.8615 | 0.8702 | 10.5 / 0.25 |
| ES, XL-WSD-derived | 936 | 0.7040 | 0.6968 | 0.8043 | 0.8262 | 10.5 / 0.25 |
| NL, XL-WSD-derived | 1,980 | 0.6900 | 0.6944 | 0.7812 | 0.8051 | 10.5 / 0.25 |

AUC change with 95% range (bootstrap over words): IT +0.009 (-0.005 to +0.022), ES +0.022 (+0.006 to +0.040), NL +0.024 (+0.016 to +0.032).

Caveats:
- EN/RU cutoffs were chosen on a separate Russian selection set; IT/ES/NL use the cutoffs chosen on Swedish.
- XL-WSD pairs use modern sentences and machine-translated WordNet definitions. "No" pairs are other dictionary senses of the same word, and some are very close in meaning to the correct one.
- The XL-WSD sets are balanced (50% Yes), unlike real data, which affects F1 at a fixed cutoff.
