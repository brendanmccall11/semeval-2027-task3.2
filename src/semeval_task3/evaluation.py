"""Positive-class F1 macro-averaged over words, then languages."""
from collections import defaultdict
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import zipfile

from .data import Key, LANGUAGES, exact_coverage, json_rows, label_map


def validate_labels(labels: dict[Key, int]) -> None:
    if not labels:
        raise ValueError('Empty labels')
    for key, value in labels.items():
        if len(key) != 3 or key[0] not in LANGUAGES or any(not isinstance(x, str) or not x for x in key):
            raise ValueError(f'Invalid prediction key: {key}')
        if type(value) is not int or value not in (0, 1):
            raise ValueError('Labels must be integer 0 or 1')


def score(gold: dict[Key, int], predictions: dict[Key, int]) -> dict:
    validate_labels(gold)
    validate_labels(predictions)
    exact_coverage(gold, predictions)
    groups = defaultdict(list)
    for key in sorted(gold):
        groups[key[:2]].append(key)
    per_word, languages = [], defaultdict(list)
    for (language, word), keys in sorted(groups.items()):
        tp = sum(gold[k] == 1 and predictions[k] == 1 for k in keys)
        fp = sum(gold[k] == 0 and predictions[k] == 1 for k in keys)
        fn = sum(gold[k] == 1 and predictions[k] == 0 for k in keys)
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        per_word.append(dict(language=language, word=word, n=len(keys), positives=sum(gold[k] for k in keys),
                             tp=tp, fp=fp, fn=fn, f1=f1,
                             precision=tp/(tp+fp) if tp+fp else 0.0,
                             recall=tp/(tp+fn) if tp+fn else 0.0))
        languages[language].append(f1)
    per_language = {lang: sum(values)/len(values) for lang, values in languages.items()}
    return dict(average=sum(per_language.values())/len(per_language),
                per_language=per_language, per_word=per_word)


def word_bootstrap(report: dict, *, other: dict | None = None, seed: int = 2027, draws: int = 10000) -> dict:
    """Stratified word bootstrap; paired differences are report minus other.

    Resamples fixed predictions, not training runs. With 10 words, intervals
    are exploratory and do not account for repeated model searches.
    """
    if type(draws) is not int or draws < 2:
        raise ValueError('At least two bootstrap draws required')
    values = {(r['language'], r['word']): r['f1'] for r in report['per_word']}
    if not values:
        raise ValueError('No word scores')
    if other is not None:
        reference = {(r['language'], r['word']): r['f1'] for r in other['per_word']}
        if values.keys() != reference.keys():
            raise ValueError('Paired bootstrap needs identical language/word groups')
        values = {k: v-reference[k] for k, v in values.items()}
    groups = defaultdict(list)
    for (lang, word), value in sorted(values.items()):
        groups[lang].append(value)
    rng, samples = random.Random(seed), []
    for _ in range(draws):
        samples.append(sum(sum(rng.choices(v, k=len(v)))/len(v) for v in groups.values()) / len(groups))
    samples.sort()
    def quantile(p):
        index = (len(samples)-1)*p
        low, high = math.floor(index), math.ceil(index)
        return samples[low] + (samples[high]-samples[low])*(index-low)
    return dict(estimate=sum(sum(v)/len(v) for v in groups.values())/len(groups),
                lower=quantile(.025), upper=quantile(.975), confidence=.95,
                unit='word, stratified within language', paired=other is not None,
                difference='first minus second' if other is not None else None,
                draws=draws, seed=seed, words=len(values))


def write_submission(path: Path, predictions: dict[Key, int]) -> None:
    validate_labels(predictions)
    grouped = defaultdict(list)
    for (language, word, sid), label in sorted(predictions.items()):
        grouped[language].append(dict(word=word, sentence_id=sid, label=label))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for language, rows in sorted(grouped.items()):
            info = zipfile.ZipInfo(f'{language}_subtask2.jsonl', date_time=(2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in rows))


def read_submission(path: Path, expected: set[Key]) -> dict[Key, int]:
    """Read without extraction and enforce the exact Subtask-2-only contract."""
    labels = {}
    names = {f'{k[0]}_subtask2.jsonl' for k in expected}
    with zipfile.ZipFile(path) as archive:
        files = archive.infolist()
        if len(files) != len(names) or {f.filename for f in files} != names:
            raise ValueError('ZIP must contain exactly one root Subtask 2 file for each expected language')
        if sum(f.file_size for f in files) > 200_000_000:
            raise ValueError('Submission exceeds 200 MB uncompressed limit')
        for file in files:
            language = file.filename.split('_')[0]
            labels.update(label_map(json_rows(archive.read(file).decode('utf-8'), file.filename), language))
    exact_coverage(expected, labels)
    return labels


def official_score(submission: Path, data_root: Path, scorer_root: Path) -> dict:
    """Run the unmodified official CLI in a fresh directory to avoid stale extraction."""
    expected = set()
    for language in sorted(LANGUAGES):
        path = Path(data_root)/language/'subtask2.jsonl'
        if path.exists():
            expected.update(label_map(json_rows(path.read_text(encoding='utf-8'), str(path)), language))
    if not expected:
        raise ValueError('No reference labels found')
    # Validate before the official script extracts the ZIP to disk.
    read_submission(submission, expected)
    with tempfile.TemporaryDirectory(prefix='semeval-official-') as directory:
        output = Path(directory)/'result.json'
        result = subprocess.run([sys.executable, str(Path(scorer_root).resolve()/'evaluation.py'),
                                 str(Path(submission).resolve()), str(Path(data_root).resolve()),
                                 '-o', str(output)], cwd=directory, text=True, capture_output=True)
        if result.returncode or not output.exists():
            raise RuntimeError(f'Official scorer failed: {result.stderr}')
        report = json.loads(output.read_text())
        if report['errors']:
            raise ValueError(f'Official scorer rejected submission: {report["errors"]}')
        return report

