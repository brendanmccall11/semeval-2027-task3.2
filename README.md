# SemEval-2027 Task 3, Subtask 2: Hypothesis-Driven Change Detection

This is our NLP course project at CU Boulder. The task gives a sentence with a marked target word, the year the sentence was written, and a definition of one sense of that word. The system has to decide whether the word is used in that sense (1) or not (0). The test data covers Swedish, English, Italian, Spanish, Dutch and Russian.

Systems are scored with macro-F1 per word (the mean of the Yes-F1 and No-F1), averaged over words and then over languages. So far the organizers have released labeled data only for Swedish: 10 words and 1,983 usages. The test phase runs from 21 to 31 January 2027.

## Our approach

We use Gemma 4 31B (`google/gemma-4-31B-it`, revision `842da379`) and ask it a Yes/No question for each usage. The score is the difference between the logits of "Yes" and "No" for the next token, and we predict 1 when the score is above a cutoff.

To adapt the model to the task, we trained a LoRA adapter (rank 16) on 8,510 examples built from two public datasets: Russian from AXOLOTL-24 and English from WiC-TSV. We deliberately did not train on any Swedish, Italian, Spanish or Dutch data, so the Swedish dev set stays a fair test of how well the model generalizes to a new language.

Training the same adapter with different random seeds gave Swedish scores anywhere from 0.724 to 0.774, so a single model depends quite a lot on luck. Our final system therefore averages the scores of five adapters trained with seeds 2027 to 2031, and predicts 1 when the average is at least 1.00. As a backup submission we keep the seed 2027 adapter on its own, with a cutoff of 0.75.

## Results

All Swedish numbers come from nested leave-one-word-out cross-validation: the cutoff for each word is chosen without looking at that word's labels.

| System | Swedish dev macro-F1 |
|---|---:|
| Gemma 4 31B without training | 0.682 |
| Gemma + LoRA, single model (mean of 5 seeds) | 0.749 ± 0.019 |
| Average of 3 seeds | 0.766 |
| Average of 5 seeds (final system) | **0.779** |

Going from three to five seeds improved the score by 0.012 (95% bootstrap interval 0.005 to 0.021), with gains on 7 of the 10 words.

For the other languages there is no official data yet, so we built our own evaluation sets from WiC-TSV, AXOLOTL-24 and XL-WSD. These sets are balanced 50/50 and look different from the competition data, so the numbers below are a sanity check rather than an estimate of our test score.

| Language | Data | Macro-F1 | AUC |
|---|---|---:|---:|
| English | WiC-TSV dev | 0.882 | 0.935 |
| Russian | AXOLOTL-24 pairs | 0.873 | 0.950 |
| Italian | XL-WSD pairs | 0.724 | 0.876 |
| Spanish | XL-WSD pairs | 0.687 | 0.825 |
| Dutch | XL-WSD pairs | 0.691 | 0.807 |

We also tried several ideas that did not beat the final system: training on more languages, training on part of the Swedish dev data, adding a second model family (Qwen3-32B) to the average, and choosing a separate cutoff for each word. For each experiment we fixed the success criterion before running it. The full numbers and our explanation of why each idea failed are in `results/RESULTS.md`.

## Repository contents

- `notebooks/`: the Colab notebooks, in the order we ran them, plus a small helper module (`project_utils.py`).
- `src/semeval_task3/`: data loading, scoring, nested cross-validation and the submission writer.
- `tests/`: tests that check our scorer against the official one.
- `configs/`: the frozen configurations of the final and backup systems, including the hash of every adapter.
- `results/`: result tables for all experiments.
- `docs/`: a detailed project log, notes on the data and rules, and the environment.

The notebook numbers do not match the step numbers of our run folders, because several steps were run inside the same notebook:

| Notebook | What happens there |
|---|---|
| 01 to 06 | Data inspection, scorer checks, baselines, first Gemma and Qwen experiments, error analysis |
| 07_Multilingual_Data_Preparation | Russian and English training data and the first LoRA training run |
| 11_Multilingual_Evaluation | Italian, Spanish and Dutch evaluation sets; re-scoring with the official metric |
| 12_Round2_Training | Training on more languages (rejected), seeds 2028 and 2029 |
| 13_OptionA_Swedish_Training | Training on Swedish dev words (rejected) and the per-word cutoff check |
| 14_Qwen_Second_Model | Qwen3-32B as a second model (rejected) |
| 15_Five_Seeds | Seeds 2030 and 2031, and the final five-seed system |

The notebooks were written for Google Colab, with the project stored in Google Drive under `semeval2027/semeval2027_colab`. Gemma 31B needs a GPU with more than 70 GB of memory; we used an RTX PRO 6000 (Colab "G4") and an A100 80 GB.

## Data and model files

The datasets, adapters and run outputs are not in this repository because of their size and licenses. The competition data is available after registering on [Codabench](https://www.codabench.org/competitions/17510/), and the official scorer is at [ChangeIsKey/semeval2027-task3](https://github.com/ChangeIsKey/semeval2027-task3). The external datasets are [AXOLOTL-24](https://github.com/ltgoslo/axolotl24_shared_task), [WiC-TSV](https://github.com/semantic-web-company/wic-tsv) and [XL-WSD](https://sapienzanlp.github.io/xl-wsd/) (CC BY-NC-SA). The trained adapters are in our shared Google Drive folder.

To run the scorer tests on a CPU, put the Swedish dev data in `work/data/dev/SV/` and the official scorer in `work/official-scorer`, then run:

```sh
python -m pip install -r requirements.txt
PYTHONPATH=src python -m pytest -q
```

## Status

The final system is frozen. What remains is a full dry run of the submission pipeline on the Swedish dev set, the test-phase submission in January, and the report.
