# Data and task assumptions

## Confirmed from the currently inspected project

- Official development data v2 in `work/data/dev/SV/`: 1,983 usages, 10 words, 705 positive labels.
- Input fields include target word, sentence ID, text, POS, period label, offsets and year.
- One Swedish definition per current development word.
- Original data and IDs remain unchanged. Five whitespace-containing spans are repaired during loading only; repairs are label-independent and inspectable through `original_span`.
- Only Swedish official dev data has been released. All six languages have been evaluated: Swedish on the official dev data, and English, Russian, Italian, Spanish and Dutch on external datasets we built (WiC-TSV, AXOLOTL-24, XL-WSD). The external results are not competition scores.
- Scorer commit used historically: `9763564cd540fda979d84ecaa09a6a0e19b9645b`.

## Questions still needing authoritative resolution

- Are evaluation words disjoint from development words?
- Exact definition languages/formats for each evaluation language.
- Precise permissions for supervised use of development labels and each external/synthetic training resource.
- Any rule/data/scoring changes since the organizer snapshots in `work/sources/`.

Public external models/resources have been described as permitted subject to documentation/licensing; verify the specific resource and current rules before training on it. Development tuning is not permission to inspect hidden evaluation annotations.

## Preprocessing limits

The current repair rule is a narrow heuristic, not a multilingual morphology solution. It requires one unique headword-prefix token with the original span length. Bounds/whitespace checks do not establish correctness of every unflagged span. E5 uses complete sentence text; Qwen uses the marked target.

## Sources

Task: https://www.changeiskey.org/event/2027-semeval/
Competition: https://www.codabench.org/competitions/17510/
Organizer code: https://github.com/ChangeIsKey/semeval2027-task3

This file summarizes inspected evidence, not a fresh verification of all online competition rules.
