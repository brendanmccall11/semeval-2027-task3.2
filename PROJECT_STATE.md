# Project state (16 November 2026)

## Scope

SemEval-2027 Task 3, Subtask 2: decide whether a target word's usage expresses a given sense definition. Goal: strong performance across SV, EN, IT, ES, NL, RU with open-weight models only (no paid APIs), without favouring Swedish at the other languages' cost. Experiments ran on Google Colab (RTX PRO 6000 "G4", 95 GB; A100 80 GB).

Colab project root: `/content/drive/MyDrive/semeval2027/semeval2027_colab`. Run folders are under `outputs/runs/`, frozen configs under `outputs/frozen/`.

## Final system (frozen 16 November 2026)

- Base: `google/gemma-4-31B-it`, revision `842da3794eaa0b77d5f08bae87a17459d91ff475`, BF16, no quantization.
- 5 LoRA adapters, one per seed (2027 to 2031). Each: rank 16, alpha 32, dropout 0.05, language-model q/k/v/o (230 layers, 45M trainable parameters); answer-token cross-entropy; micro-batch 1, effective batch 16, learning rate 1e-4, 27 warm-up updates then linear decay; one epoch, 532 updates; final update used.
- Training data for every adapter: 8,510 examples (RU 6,078 from AXOLOTL-24, EN 2,432 from WiC-TSV; 4,275 Yes / 4,235 No).
- Prompt v1: language-templated system message ("You are an expert lexicographer of historical {language}.") and user message with lemma, definition, year and the marked sentence; thinking disabled.
- Score: logit(Yes) minus logit(No) (token ids 10784 and 3771), averaged over the 5 adapters. Predict 1 if the average is at least 1.00.
- Config: `outputs/frozen/main_system_ensemble5_gemma4_31b_lora.json` (copy in `configs/`).
- Backup: seed 2027 alone, cutoff 0.75, config `outputs/frozen/primary_gemma4_31b_lora_macro.json`.

| Seed | Adapter folder | SHA-256 (first 16) |
|---|---|---|
| 2027 | `step09_lora_full_pilot/20261105T232211Z_6b04a51b/checkpoint_0532/adapter` | 6f3d236ac50e07cc |
| 2028 | `step15_seed_ensemble/seed_2028/adapter_final` | 9b6d485af7177301 |
| 2029 | `step15_seed_ensemble/seed_2029/adapter_final` | bd5c266eff0d57f0 |
| 2030 | `step19_five_seeds/seed_2030/adapter_final` | a6e593baf7a18852 |
| 2031 | `step19_five_seeds/seed_2031/adapter_final` | 13416fbf49f07abe |

## Metric

Official scorer commit `0e69238`: per word, macro-F1 (mean of Yes-F1 and No-F1); mean over words, then over languages. Results before 13 November used per-word positive-class F1; all were re-scored (step 12) and the old numbers are listed at the end of this file for history only.

Swedish estimates use nested leave-one-word-out cross-validation: the cutoff for each held-out word is chosen on the other 9 words from a predeclared grid (-100 to 100, step 0.25, plus ±1000; ties to the lower cutoff).

## Decision rules

Every experiment after round 1 was judged by a rule saved before it ran. The standard rule for replacing the main system: Swedish improves by at least +0.02 and is better on at least 6 of 10 words, and no other language loses more than 0.02 AUC or 0.02 macro-F1. The 5-seed average was the one exception, decided in advance on principle: it would replace the 3-seed average whatever the Swedish number did, unless a health check failed or a language dropped by more than 0.02.

## Experiment log

| Step | Experiment | Swedish macro-F1 | Outcome |
|---|---|---:|---|
| 04 | Gemma 4 31B, no training | 0.6818 | base model chosen |
| 09 | Round 1 LoRA, seed 2027 | 0.7529 | adopted |
| 10 | Per-word EM decisions; original + trained score mix | (old metric) | rejected |
| 12 | Re-scoring with the official macro-F1 | | cutoff moved 0.25 to 0.75 |
| 14 | Round 2: 15,610 examples incl. IT, ES, NL, SV (Wiktionary), FI | 0.7124 | rejected |
| 15 | Seeds 2028, 2029; average of 3 | 0.7664 | kept as system 2 |
| 16 | Option A: Swedish dev words in training, 2 folds of 5 words | 0.7351 | rejected (also ES and IT F1 drops) |
| 17 | Per-word cutoffs (centered, half-centered, chosen in-fold) | 0.6479 / 0.7459 / 0.7664 | rejected; single cutoff kept |
| 18 | Qwen3-32B, same recipe; averaged with system 2 | 0.6558 alone / 0.7380 mix | rejected |
| 19 | Seeds 2030, 2031; average of 5 | **0.7789** | **adopted and frozen** |

Single-model spread over 5 seeds: 0.7240 to 0.7741 (mean 0.7486, std 0.0192). Differences below about 0.02 between single models are noise.

## Open items

1. Submission pipeline (notebook 16): dry run on Swedish dev must reproduce 0.7789 (main) and 0.7643 (backup, in-sample) through the official scorer. Blocked on 16 November by a Colab Drive-mount bug ("credential propagation was unsuccessful"), not by the code.
2. Test phase, 21 to 31 January 2027: put the test data in `work/data/test/` (same layout as `work/data/dev/`), set `MODE = "test"`, run, submit the main file, then the backup.
3. Report and system paper.

## Main artifact references

All paths are relative to the Colab project root.

- Swedish dev data: `work/data/dev/SV/`.
- Round 1 training and selection data: `outputs/runs/step07_pilot_dataset/20261105T223146Z_1dd5a63b`.
- EN/RU evaluation sets: `outputs/runs/step07_pilot_evaluation/20261105T223700Z_5259f45a`.
- IT/ES/NL evaluation sets: `outputs/runs/step11_xlwsd_evaluation_sets/20261106T032028Z_aa4e2120`.
- Macro-F1 re-scoring: `outputs/runs/step12_macro_f1_rescore/20261113T053308Z_bc99dae6`.
- Round 2: `outputs/runs/step13_round2_dataset/20261114T214116Z_acd7af91`, `outputs/runs/step14_round2_lora/20261114T215454Z_5b5b1dd6`.
- Seeds and decisions: `outputs/runs/step15_seed_ensemble/`, `outputs/runs/step19_five_seeds/` (`five_seed_decision.json`).
- Rejected experiments: `outputs/runs/step16_option_a/`, `outputs/runs/step17_cutoff_check/`, `outputs/runs/step18_qwen3_32b/`.
- Submission files: `outputs/submissions/dry_run_SV_dev/` (dry run), `outputs/submissions/test_phase2/` (January).

## History: results under the old metric (positive-class F1)

| System | Swedish positive F1 |
|---|---:|
| All-positive baseline | 0.4716 |
| e5 embedding similarity | 0.4522 |
| Qwen3-8B, Yes/No | 0.4626 |
| Qwen3.8-27B, Yes/No | 0.6447 |
| Gemma 4 31B, no training | 0.6782 |
| Gemma + dictionary senses | 0.6844 |
| Gemma + usage-group smoothing | 0.6755 |
| Gemma + LoRA (round 1) | 0.7043 |
| LoRA + per-word EM decisions | 0.5956 |
| LoRA + original Gemma mix | 0.7043 |
