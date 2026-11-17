# SemEval-2027 Task 3, Subtask 2: Hypothesis-Driven Change Detection

NLP course project (CU Boulder). Given a sentence with a marked target word, its year, and a sense definition, predict whether the word is used in that sense (1) or not (0).

**Official metric** (scorer commit `0e69238`): for each word, the mean of Yes-F1 and No-F1 (macro-F1); averaged over words, then over languages.

Test languages: Swedish (SV), English (EN), Italian (IT), Spanish (ES), Dutch (NL), Russian (RU). Only Swedish development data with labels has been released (10 words, 1,983 usages). The test phase runs from 21 January 2027, 00:00 UTC, to 31 January 2027, 23:59 UTC. The best submission counts.

## Final system

- **Model:** `google/gemma-4-31B-it` (revision `842da3794eaa0b77d5f08bae87a17459d91ff475`) with LoRA adapters (rank 16, alpha 32, dropout 0.05, attention q/k/v/o, BF16 base).
- **Training data:** 8,510 examples from Russian (AXOLOTL-24) and English (WiC-TSV). No Swedish, Italian, Spanish or Dutch data was used for training.
- **Ensemble:** the same recipe trained 5 times with different random seeds (2027 to 2031). Each usage gets the average of the 5 scores.
- **Score:** logit(Yes) minus logit(No) for the next token after a Yes/No question. Predict 1 if the averaged score is at least **1.00**.
- **Backup submission:** seed 2027 alone, cutoff 0.75.
- Configs: `configs/main_system_ensemble5_gemma4_31b_lora.json` (main) and `configs/primary_gemma4_31b_lora_macro.json` (backup).

## Results

Swedish is the only official data. Scores are macro-F1 from nested leave-one-word-out cross-validation: each word's cutoff is chosen without that word's labels.

| System | Swedish dev macro-F1 |
|---|---:|
| Gemma 4 31B, no training | 0.682 |
| Gemma + LoRA, single model (mean of 5 seeds) | 0.749 ± 0.019 |
| Average of 3 seeds | 0.766 |
| **Average of 5 seeds (final)** | **0.779** |

Average of 5 minus average of 3: +0.012 (95% range +0.005 to +0.021, bootstrap over words). Better on 7 of 10 words, worse on 1 (by 0.002).

The other five languages were checked on datasets we built ourselves from public resources. They are **not** competition scores, they are always 50% Yes, and they should not be averaged with the Swedish number.

| Data | Macro-F1 | AUC |
|---|---:|---:|
| EN, WiC-TSV dev | 0.882 | 0.935 |
| RU, AXOLOTL-derived pairs | 0.873 | 0.950 |
| IT, XL-WSD-derived pairs | 0.724 | 0.876 |
| ES, XL-WSD-derived pairs | 0.687 | 0.825 |
| NL, XL-WSD-derived pairs | 0.691 | 0.807 |

Ideas tested and rejected (all against rules fixed before running): adding more training languages, training on Swedish dev words, mixing in a second model family (Qwen3-32B), and per-word cutoffs. Details in `results/RESULTS.md`.

## Repository layout

| Path | Contents |
|---|---|
| `notebooks/` | Colab notebooks in run order, plus `project_utils.py` |
| `src/semeval_task3/` | Data loader, scorer, nested cross-validation, submission writer |
| `tests/` | Tests for the evaluation harness (compared against the official scorer) |
| `configs/` | Frozen system configurations (model revision, adapter hashes, cutoff, prompt) |
| `results/` | Summary tables of all experiments |
| `docs/` | Project state, data and rules notes, environment |

## Notebooks

Notebook numbers do not match the step numbers of the run folders, because several steps ran inside one notebook.

| Notebook | Steps | What it does |
|---|---|---|
| 00_START_HERE | | Project map |
| 01_Data_Inspection | 01 | Audit of the Swedish dev data, span repairs |
| 02_Evaluation_Harness | 02 | Checks our scorer against the official scorer |
| 03_Baselines | 03 | All-positive and embedding (e5) baselines |
| 04_Candidate_Systems_Qwen | 04 | Qwen3-8B Yes/No scoring |
| 04b_Candidate_Gemma | 04 | Gemma 4 31B Yes/No scoring (became the base model) |
| 04c_Candidate_Qwen38_27B | 04 | Second large model for comparison |
| 05_Usage_Groups | 05 | Usage clustering experiment (no gain) |
| 05b_Dictionary_Senses | 06 | Dictionary senses in the prompt (no clear gain) |
| 06_Error_Analysis | | Per-word errors and examples |
| 07_Multilingual_Data_Preparation | 07 to 10 | Russian/English training and evaluation sets, LoRA training (round 1), follow-up experiments |
| 11_Multilingual_Evaluation | 11, 12 | Italian, Spanish, Dutch test pairs from XL-WSD; re-scoring with the official macro-F1 |
| 12_Round2_Training | 13 to 15 | Round 2 (more languages, rejected); seeds 2028 and 2029; average of 3 |
| 13_OptionA_Swedish_Training | 16, 17 | Training with Swedish dev words in two folds (rejected); per-word cutoff check (rejected) |
| 14_Qwen_Second_Model | 18 | Qwen3-32B with the same recipe, averaged with Gemma (rejected) |
| 15_Five_Seeds | 19 | Seeds 2030 and 2031; average of 5 adopted and frozen |
| 16_Submission_Pipeline | | Scores a data folder with the frozen systems and writes the two submission ZIP files |

The notebooks run on Google Colab and expect the project folder at `/content/drive/MyDrive/semeval2027/semeval2027_colab`. Gemma 31B needs a GPU with more than 70 GB (RTX PRO 6000 / G4, or A100 80 GB).

## Data and model files (not in this repo)

Kept out of GitHub because of size and licenses:

- **Competition data:** register on the [Codabench competition](https://www.codabench.org/competitions/17510/) and place the Swedish dev files in `work/data/dev/SV/`.
- **Official scorer:** `git clone https://github.com/ChangeIsKey/semeval2027-task3.git work/official-scorer`
- **External data:** [AXOLOTL-24](https://github.com/ltgoslo/axolotl24_shared_task), [WiC-TSV](https://github.com/semantic-web-company/wic-tsv), [XL-WSD](https://sapienzanlp.github.io/xl-wsd/) (CC BY-NC-SA).
- **LoRA adapters and run outputs:** in our shared Google Drive folder (ask Vishwas for access). Each adapter's SHA-256 is recorded in the configs.

## Running the harness tests (CPU)

Needs the Swedish dev data and the official scorer in `work/` as described above.

```sh
python -m pip install -r requirements.txt
PYTHONPATH=src python -m pytest -q
```

## Next steps

1. Dry run of the submission pipeline on Swedish dev (must reproduce 0.779 through the official scorer).
2. Test phase (21 to 31 January 2027): run the pipeline on the test data, submit the main file, then the backup.
3. Report and system paper: error analysis by word and time period.
