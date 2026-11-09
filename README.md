# SemEval-2027 Task 3, Subtask 2: Hypothesis-Driven Change Detection

NLP course project (CU Boulder). Given a sentence with a marked target word, its year, and a sense definition, predict whether the word is used in that sense (1) or not (0). Scoring is positive-class F1 per word, averaged per language, then across languages.

Test languages: Swedish (SV), English (EN), Italian (IT), Spanish (ES), Dutch (NL), Russian (RU). Only Swedish development data with labels has been released so far (10 words, 1,983 usages). The test window is January 22 to 31, 2027.

## Current system

- **Model:** `google/gemma-4-31B-it` (revision `842da3794eaa0b77d5f08bae87a17459d91ff475`) with a LoRA adapter (rank 16, BF16, checkpoint 532).
- **Training data for the adapter:** 8,510 examples from Russian (AXOLOTL-24) and English (WiC-TSV). No Swedish, Italian, Spanish or Dutch data was used for training.
- **Score:** logit(Yes) minus logit(No) for the next token after a Yes/No question. Predict 1 if the score is at least the cutoff (0.25).
- **Fallback:** the same Gemma model without the adapter (cutoff 10.5).
- Configs: `configs/primary_gemma4_31b_lora.json` and `configs/provisional_primary_gemma4_31b.json`.

## Results so far

Swedish is the only official data. Its score is the mean per-word F1 from nested leave-one-word-out cross-validation, so each word's cutoff is chosen without that word's labels.

| Data | Original Gemma | Gemma + LoRA |
|---|---|---|
| Swedish dev, mean per-word F1 (official format) | 0.6782 | **0.7043** |

The other five languages were checked on external datasets we built ourselves. They are **not** competition scores and should not be averaged with the Swedish number.

| Data | Original F1 | LoRA F1 | Original AUC | LoRA AUC |
|---|---|---|---|---|
| EN, WiC-TSV dev | 0.8708 | 0.8815 | 0.9097 | 0.9339 |
| RU, AXOLOTL-derived pairs | 0.8676 | 0.8757 | 0.9219 | 0.9475 |
| IT, XL-WSD-derived pairs | 0.7718 | 0.7448 | 0.8615 | 0.8702 |
| ES, XL-WSD-derived pairs | 0.7040 | 0.6968 | 0.8043 | 0.8262 |
| NL, XL-WSD-derived pairs | 0.6900 | 0.6944 | 0.7812 | 0.8051 |

The adapter ranks examples better in every language (higher AUC). At the fixed cutoffs, F1 dropped for Italian and Spanish, so cutoff choice across languages is still an open problem. Full details, cutoffs and caveats are in `results/RESULTS.md` and `docs/PROJECT_STATE.md`.

## Repository layout

| Path | Contents |
|---|---|
| `notebooks/` | Colab notebooks in run order, outputs cleared, plus `project_utils.py` |
| `src/semeval_task3/` | Data loader, scorer, nested cross-validation and command-line tools |
| `tests/` | Tests for the evaluation harness (compared against the official scorer) |
| `configs/` | Frozen system configurations (model revision, adapter hash, cutoff, prompt) |
| `results/` | Summary tables of all experiments |
| `docs/` | Project state, data and rules notes, environment |

## Notebooks, in order

| Notebook | What it does |
|---|---|
| 00_START_HERE | Project map |
| 01_Data_Inspection | Audit of the Swedish dev data, span repairs |
| 02_Evaluation_Harness | Checks our scorer against the official scorer |
| 03_Baselines | All-positive and embedding (e5) baselines |
| 04_Candidate_Systems_Qwen | Qwen3-8B Yes/No scoring |
| 04b_Candidate_Gemma | Gemma 4 31B Yes/No scoring (became the main system) |
| 04c_Candidate_Qwen38_27B | Second large model for comparison |
| 05_Usage_Groups | Usage clustering experiment (no gain) |
| 05b_Dictionary_Senses | Adding dictionary senses to the prompt (no clear gain) |
| 06_Error_Analysis | Per-word errors and examples |
| 07_Multilingual_Data_Preparation | Builds the Russian/English training and evaluation sets; LoRA training, checkpoint choice, adoption decision and follow-up experiments |
| 11_Multilingual_Evaluation | Builds Italian, Spanish and Dutch test pairs from XL-WSD and compares original vs LoRA Gemma |

The notebooks run on Google Colab and expect the project folder at `/content/drive/MyDrive/semeval2027/semeval2027_colab`. Change `PROJECT` in the setup cell if yours is elsewhere. Gemma 31B needs an A100 80 GB (or larger) GPU.

## Data and model files (not in this repo)

These are kept out of GitHub because of size and licenses:

- **Competition data:** register on the [Codabench competition](https://www.codabench.org/competitions/17510/) and place the Swedish dev files in `work/data/dev/SV/`.
- **Official scorer:** `git clone https://github.com/ChangeIsKey/semeval2027-task3.git work/official-scorer`
- **External data:** [AXOLOTL-24](https://github.com/ltgoslo/axolotl24_shared_task), [WiC-TSV](https://github.com/semantic-web-company/wic-tsv), [XL-WSD](https://sapienzanlp.github.io/xl-wsd/) (CC BY-NC-SA).
- **LoRA adapter and run outputs:** in our shared Google Drive folder (ask Vishwas for access). The adapter's SHA-256 is recorded in `configs/primary_gemma4_31b_lora.json`.

## Running the harness tests (CPU)

Needs the Swedish dev data and the official scorer in `work/` as described above.

```sh
python -m pip install -r requirements.txt
PYTHONPATH=src python -m pytest -q
```

## Next steps

1. Final submission script and checks (same prompt, scores reproduce, valid ZIP, official scorer).
2. Decide how to set the cutoff for languages other than Swedish.
3. Possible second training round with more languages and harder negative examples.
4. Analysis by time period and error type for the paper.
