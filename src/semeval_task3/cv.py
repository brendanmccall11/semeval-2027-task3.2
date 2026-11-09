"""Nested leave-one-word-out selection with explicit fit/predict boundaries."""
from dataclasses import dataclass
import math
from typing import Callable, Mapping

from .data import Dataset, Group, Key, Usage, exact_coverage
from .evaluation import score, word_bootstrap

Predictor = Callable[[tuple[Usage, ...]], Mapping[Key, float]]
Trainer = Callable[[tuple[Usage, ...], dict[Key, int]], Predictor]


@dataclass(frozen=True)
class Candidate:
    name: str
    fit: Trainer
    thresholds: tuple[float, ...]


def nested_folds(dataset: Dataset) -> list[dict]:
    groups = sorted({u.group for u in dataset.usages})
    if len(groups) < 3:
        raise ValueError('Nested leave-one-word-out needs at least three words')
    folds = []
    for held in groups:
        train = [g for g in groups if g != held]
        folds.append(dict(test_group=held, train_groups=train,
                          inner=[dict(validation_group=v, train_groups=[g for g in train if g != v]) for v in train]))
    return folds


def checked_scores(predictor: Predictor, usages: tuple[Usage, ...]) -> dict[Key, float]:
    values = dict(predictor(usages))
    exact_coverage((u.key for u in usages), values)
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values.values()):
        raise ValueError('Model scores must be finite numbers')
    return values


def nested_word_cv(dataset: Dataset, candidates: list[Candidate], *, bootstrap_draws: int = 10000) -> dict:
    """Select a candidate and threshold using inner out-of-fold scores only.

    fit receives training usages + training labels. Its predictor receives
    held-out usages WITHOUT labels. Scores must increase with sense match.
    Frozen zero-shot candidates may ignore fit inputs and use cached scores.
    Trainable candidates must create fresh models for each fit, without shared
    fitted state or caches trained on other folds.
    """
    gold = dataset.require_gold()
    exact_coverage(dataset.keys, gold)
    if not candidates or len({c.name for c in candidates}) != len(candidates):
        raise ValueError('Provide uniquely named candidates')
    for candidate in candidates:
        if not candidate.name or not candidate.thresholds or any(not math.isfinite(t) for t in candidate.thresholds):
            raise ValueError('Each candidate needs a name and finite, prespecified thresholds')
    def subset(groups):
        groups = set(groups)
        return tuple(u for u in dataset.usages if u.group in groups)
    def labels(usages):
        return {u.key: gold[u.key] for u in usages}
    predictions, selections = {}, []
    for fold in nested_folds(dataset):
        options = []
        for candidate in sorted(candidates, key=lambda c: c.name):
            inner_scores = {}
            for inner in fold['inner']:
                train = subset(inner['train_groups'])
                validation = subset([inner['validation_group']])
                predictor = candidate.fit(train, labels(train))
                inner_scores.update(checked_scores(predictor, validation))
            inner_gold = {k: gold[k] for k in inner_scores}
            for threshold in sorted(set(candidate.thresholds)):
                predicted = {k: int(v >= threshold) for k, v in inner_scores.items()}
                value = score(inner_gold, predicted)['average']
                options.append((value, candidate.name, threshold, candidate))
        # Deterministic tie: candidate name, then lower threshold. No test labels.
        value, name, threshold, chosen = sorted(options, key=lambda x: (-x[0], x[1], x[2]))[0]
        train, test = subset(fold['train_groups']), subset([fold['test_group']])
        predictor = chosen.fit(train, labels(train))
        test_scores = checked_scores(predictor, test)
        predictions.update({k: int(v >= threshold) for k, v in test_scores.items()})
        selections.append(dict(test_group=fold['test_group'], candidate=name, threshold=threshold,
                               inner_f1=value, train_words=len(fold['train_groups']), test_usages=len(test)))
    report = score(gold, predictions)
    return dict(predictions=predictions, selections=selections, report=report,
                confidence_interval=word_bootstrap(report, draws=bootstrap_draws))
