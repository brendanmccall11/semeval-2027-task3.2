"""Immutable model inputs; labels are loaded separately and only on request."""
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Iterable

LANGUAGES = frozenset({'EN', 'SV', 'IT', 'ES', 'NL', 'RU'})
Key = tuple[str, str, str]  # language, word, sentence_id
Group = tuple[str, str]  # language, word


def json_rows(text: str, source: str) -> list[dict]:
    rows = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f'{source}:{number}: invalid JSON') from exc
        if not isinstance(row, dict):
            raise ValueError(f'{source}:{number}: expected a JSON object')
        rows.append(row)
    return rows


def nonempty(row: dict, field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f'{field} must be a non-empty string')
    return value


def label_map(rows: Iterable[dict], language: str) -> dict[Key, int]:
    if language not in LANGUAGES:
        raise ValueError(f'Unsupported language: {language}')
    result = {}
    for row in rows:
        word, sid = nonempty(row, 'word'), nonempty(row, 'sentence_id')
        label = row.get('label')
        if type(label) is not int or label not in (0, 1):
            raise ValueError('label must be integer 0 or 1')
        key = language, word, sid
        if key in result:
            raise ValueError(f'Duplicate usage: {key}')
        result[key] = label
    if not result:
        raise ValueError('Empty label file')
    return result


def exact_coverage(expected: Iterable[Key], actual: Iterable[Key]) -> None:
    expected, actual = set(expected), set(actual)
    if expected != actual:
        missing, extra = sorted(expected - actual), sorted(actual - expected)
        raise ValueError(f'ID coverage mismatch: missing={missing[:3]}, unknown={extra[:3]}')


def repair_span(word: str, text: str, start: int, end: int) -> tuple[int, int] | None:
    """Label-free span repair for a flagged target span.

    Looks for tokens that start with the headword (case-insensitive) and have
    exactly the same length as the provided span. Returns the new offsets only
    when there is exactly one such token; otherwise returns None and the usage
    stays flagged. Raw files and sentence IDs are never changed.
    """
    pattern = re.compile(r'(?<!\w)' + re.escape(word) + r'\w*', re.IGNORECASE)
    matches = [m for m in pattern.finditer(text) if m.end() - m.start() == end - start]
    if len(matches) != 1:
        return None
    return matches[0].start(), matches[0].end()


@dataclass(frozen=True)
class Usage:
    language: str
    word: str
    sentence_id: str
    text: str
    definition: str
    start: int
    end: int
    year: int
    period_label: str
    pos: str
    span_flags: tuple[str, ...] = ()
    original_span: tuple[int, int] | None = None  # set only when the span was repaired

    @property
    def key(self) -> Key:
        return self.language, self.word, self.sentence_id

    @property
    def group(self) -> Group:
        return self.language, self.word

    def marked_text(self) -> str:
        if self.span_flags:
            raise ValueError(f'Unresolved target span for {self.key}: {self.span_flags}')
        return self.text[:self.start] + '<t>' + self.text[self.start:self.end] + '</t>' + self.text[self.end:]


@dataclass
class Dataset:
    usages: tuple[Usage, ...]
    gold: dict[Key, int] | None = None

    @property
    def keys(self) -> set[Key]:
        return {u.key for u in self.usages}

    def require_gold(self) -> dict[Key, int]:
        if self.gold is None:
            raise ValueError('Gold labels were not loaded')
        return self.gold


def load_dataset(root: Path, *, with_gold: bool = False, repair_spans: bool = True) -> Dataset:
    root = Path(root)
    dirs = sorted(p for p in root.iterdir() if p.is_dir() and p.name in LANGUAGES)
    if not dirs:
        raise ValueError(f'No supported language folders in {root}')
    usages, gold = [], {}
    for folder in dirs:
        definitions = {}
        for row in json_rows((folder/'definitions.jsonl').read_text(encoding='utf-8'), 'definitions'):
            word = nonempty(row, 'word')
            if word in definitions:
                raise ValueError(f'Duplicate definition: {word}')
            definitions[word] = nonempty(row, 'definition')
        seen = set()
        for row in json_rows((folder/'usages.jsonl').read_text(encoding='utf-8'), 'usages'):
            word, sid = nonempty(row, 'word'), nonempty(row, 'sentence_id')
            text, period, pos = (nonempty(row, f) for f in ('text', 'period_label', 'pos'))
            if word not in definitions:
                raise ValueError(f'Missing definition: {word}')
            start, end = row.get('start'), row.get('end')
            if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
                raise ValueError(f'Invalid offsets: {word}/{sid}')
            year = row.get('year')
            if not (type(year) is int or isinstance(year, str) and year.isdecimal()):
                raise ValueError(f'Invalid year: {word}/{sid}')
            flags = ('whitespace_in_target_span',) if any(c.isspace() for c in text[start:end]) else ()
            original = None
            if flags and repair_spans:
                repaired = repair_span(word, text, start, end)
                if repaired is not None:
                    original, (start, end), flags = (start, end), repaired, ()
            usage = Usage(folder.name, word, sid, text, definitions[word], start, end, int(year), period, pos,
                          flags, original)
            if usage.key in seen:
                raise ValueError(f'Duplicate usage: {usage.key}')
            seen.add(usage.key)
            usages.append(usage)
        if not seen:
            raise ValueError(f'Empty usages: {folder.name}')
        if {k[1] for k in seen} != set(definitions):
            raise ValueError(f'Definitions/usage words differ: {folder.name}')
        if with_gold:
            labels = label_map(json_rows((folder/'subtask2.jsonl').read_text(encoding='utf-8'), 'gold'), folder.name)
            exact_coverage(seen, labels)
            gold.update(labels)
    return Dataset(tuple(sorted(usages, key=lambda u: u.key)), gold if with_gold else None)

