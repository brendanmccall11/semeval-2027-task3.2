"""Commands: prepare, score, compare and verify. No network submissions."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from importlib.metadata import version

from .cv import nested_folds
from .data import load_dataset
from .evaluation import official_score, read_submission, score, word_bootstrap, write_submission


def save(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for name in ('prepare', 'score', 'compare', 'verify'):
        p = sub.add_parser(name)
        p.add_argument('--data', type=Path, default=Path('work/data/dev'))
        p.add_argument('--out', type=Path, required=True)
        if name in ('score', 'compare'):
            p.add_argument('--predictions', type=Path, required=True)
        if name == 'compare':
            p.add_argument('--reference', type=Path, required=True)
        if name in ('score', 'verify'):
            p.add_argument('--official-root', type=Path, required=name == 'verify')
    args = parser.parse_args()
    dataset = load_dataset(args.data, with_gold=args.command != 'prepare')
    if args.command == 'prepare':
        result = dict(usages=len(dataset.usages), languages=sorted({u.language for u in dataset.usages}),
                      folds=nested_folds(dataset),
                      span_flags=[dict(language=u.language, word=u.word, sentence_id=u.sentence_id,
                                       flags=u.span_flags) for u in dataset.usages if u.span_flags],
                      span_repairs=[dict(language=u.language, word=u.word, sentence_id=u.sentence_id,
                                         original=list(u.original_span), repaired=[u.start, u.end],
                                         target=u.text[u.start:u.end])
                                    for u in dataset.usages if u.original_span])
    elif args.command == 'compare':
        first = score(dataset.require_gold(), read_submission(args.predictions, dataset.keys))
        second = score(dataset.require_gold(), read_submission(args.reference, dataset.keys))
        result = dict(first=first, second=second, paired_confidence_interval=word_bootstrap(first, other=second))
    elif args.command == 'score':
        result = score(dataset.require_gold(), read_submission(args.predictions, dataset.keys))
        result['confidence_interval'] = word_bootstrap(result)
        if args.official_root:
            official = official_score(args.predictions, args.data, args.official_root)
            check_agreement(result, official)
            result['official'] = official
    else:
        result = verify(dataset, args.data, args.official_root, args.out.parent/'step2-verification')
    save(args.out, result)
    print(f'{args.command}: saved {args.out}')


def check_agreement(local: dict, official: dict) -> None:
    official = official['results']['subtask2']
    for language, value in {**local['per_language'], 'average': local['average']}.items():
        if abs(value - official[language]) > 1e-12:
            raise AssertionError(f'Official/local mismatch for {language}: {value} vs {official[language]}')


def verify(dataset, data_root, scorer_root, directory):
    directory.mkdir(parents=True, exist_ok=True)
    gold = dataset.require_gold()
    # These are scorer checks, not trained models or leaderboard predictions.
    cases = dict(all_positive={k:1 for k in gold}, all_negative={k:0 for k in gold},
                 id_hash_sanity={k:int(hashlib.sha256(json.dumps(k).encode()).hexdigest(),16)%2 for k in gold})
    checks = {}
    for name, predictions in cases.items():
        archive = directory/f'{name}.zip'
        write_submission(archive, predictions)
        restored = read_submission(archive, dataset.keys)
        local = score(gold, restored)
        official = official_score(archive, data_root, scorer_root)
        check_agreement(local, official)
        checks[name] = dict(local=local, official=official, confidence_interval=word_bootstrap(local),
                            maximum_absolute_difference=max(abs(v-official['results']['subtask2'][lang])
                                                            for lang,v in local['per_language'].items()),
                            archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    return dict(purpose='Scorer verification only; no model trained or submission uploaded',
                python=platform.python_version(),
                dependencies={name:version(name) for name in ('numpy','scipy','scikit-learn','bcubed')},
                scorer_commit=subprocess.check_output(['git','-C',str(scorer_root),'rev-parse','HEAD'],text=True).strip(),
                checks=checks)


if __name__ == '__main__':
    main()
