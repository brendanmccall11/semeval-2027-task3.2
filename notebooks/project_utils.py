"""Shared notebook utilities. No model downloads or execution on import."""
from collections import defaultdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
import hashlib
import json
import math
import platform
import subprocess
import sys
import uuid


def run_checked(args, *, cwd):
    result = subprocess.run([str(x) for x in args], cwd=cwd, text=True, capture_output=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)
    result.check_returncode()
    return result


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def new_run(project, step):
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    path = Path(project) / 'outputs' / 'runs' / step / (stamp + '_' + uuid.uuid4().hex[:8])
    path.mkdir(parents=True, exist_ok=False)
    return path


def read_scores(path, dataset):
    from semeval_task3.data import exact_coverage
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Missing saved scores: {path}. Use the complete Drive project or explicitly enable a new model run.')
    raw = json.loads(path.read_text(encoding='utf-8'))
    scores = {tuple(k.split('|')): value for k, value in raw.items()}
    exact_coverage(dataset.keys, scores)
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in scores.values()):
        raise ValueError('Cached scores must be finite numbers.')
    return scores


def save_scores(path, scores):
    if any('|' in part for key in scores for part in key):
        raise ValueError('The legacy score format cannot encode IDs containing |.')
    save_json(path, {'|'.join(k): float(v) for k, v in sorted(scores.items())})


def frozen(scores):
    def fit(train_usages, train_labels):
        return lambda test: {u.key: float(scores[u.key]) for u in test}
    return fit


def standardized_by_word(dataset, scores):
    # Transductive, label-free: uses all input usages of each held-out word.
    import numpy as np
    grouped = defaultdict(list)
    for usage in dataset.usages:
        grouped[usage.group].append(scores[usage.key])
    stats = {g: (np.mean(v), np.std(v) + 1e-8) for g, v in grouped.items()}
    return {u.key: float((scores[u.key] - stats[u.group][0]) / stats[u.group][1]) for u in dataset.usages}


def record_environment(project, destination, **details):
    project = Path(project)
    names = ('numpy', 'scipy', 'scikit-learn', 'bcubed', 'pytest', 'torch', 'transformers', 'sentence-transformers', 'huggingface-hub')
    packages = {}
    for name in names:
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    files = list((project / 'src').rglob('*.py')) + list((project / 'tests').glob('*.py'))
    files += list((project / 'notebooks').glob('*.py')) + list((project / 'notebooks').glob('*.ipynb'))
    files += list((project / 'work/data/dev').rglob('*.jsonl'))
    files += list((project / 'work/official-scorer').glob('*.py'))
    record = dict(created_utc=datetime.now(timezone.utc).isoformat(), python=platform.python_version(),
                  packages=packages, sha256={str(p.relative_to(project)): digest(p) for p in sorted(files)}, **details)
    save_json(destination, record)
    return record


def save_evaluation(destination, name, result, dataset, project, *, official=False):
    from semeval_task3.evaluation import write_submission, score, word_bootstrap, official_score
    from semeval_task3.__main__ import check_agreement
    destination = Path(destination)
    archive = destination / (name + '.zip')
    write_submission(archive, result['predictions'])
    report = {k: v for k, v in result.items() if k != 'predictions'}
    reference = score(dataset.require_gold(), {k: 1 for k in dataset.keys})
    report['versus_all_positive'] = word_bootstrap(result['report'], other=reference)
    report['prediction_sha256'] = digest(archive)
    if official:
        report['official'] = official_score(archive, Path(project)/'work/data/dev', Path(project)/'work/official-scorer')
        check_agreement(result['report'], report['official'])
    save_json(destination / (name + '.json'), report)
    ci = result['confidence_interval']
    delta = report['versus_all_positive']
    print(f"{name}: F1={result['report']['average']:.6f}; 95% CI [{ci['lower']:.4f}, {ci['upper']:.4f}]")
    print(f"  Difference vs all-positive: {delta['estimate']:+.4f} [{delta['lower']:+.4f}, {delta['upper']:+.4f}]")
    return report
