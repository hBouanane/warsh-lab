"""Reading, writing, and quality-checking segment manifests.

A *manifest* is one record per audio segment.  JSONL, JSON, CSV, and TSV are all
accepted and detected from the file extension; JSONL is the default because it
appends cleanly and survives a crashed export.

The interesting part is :func:`check`, which runs the corpus audit you want
before a training run rather than after one:

* structural problems -- missing fields, duplicate ids, empty text
* invalid surah/ayah references, validated against the shipped surah table
* label form -- how many references are already in pausal form
* a charset audit listing every codepoint in the corpus by category, so an
  unknown glyph shows up here instead of inside the tokenizer
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from . import chars as C
from . import surahs
from . import text as T

__all__ = [
    "Issue",
    "CheckReport",
    "read",
    "write",
    "check",
    "join_predictions",
    "REQUIRED_FIELDS",
]

#: Fields every manifest record needs before it is useful for training.
REQUIRED_FIELDS = ("segment_id", "text_warsh")

ERROR = "error"
WARNING = "warning"
INFO = "info"


@dataclass(frozen=True)
class Issue:
    level: str
    code: str
    message: str
    segment_id: Optional[str] = None

    def __str__(self) -> str:
        where = f" [{self.segment_id}]" if self.segment_id else ""
        return f"{self.level.upper():<7} {self.code}{where}: {self.message}"

    def as_dict(self) -> dict:
        return {
            "level": self.level,
            "code": self.code,
            "message": self.message,
            "segment_id": self.segment_id,
        }


@dataclass
class CheckReport:
    """Everything :func:`check` learned about a manifest."""

    n_records: int = 0
    issues: List[Issue] = field(default_factory=list)
    charset: Dict[str, List[tuple]] = field(default_factory=dict)
    unknown_chars: List[tuple] = field(default_factory=list)
    groups: List[tuple] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def errors(self) -> List[Issue]:
        return [i for i in self.issues if i.level == ERROR]

    @property
    def warnings(self) -> List[Issue]:
        return [i for i in self.issues if i.level == WARNING]

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict:
        return {
            "n_records": self.n_records,
            "ok": self.ok,
            "issues": [i.as_dict() for i in self.issues],
            "charset": {k: [list(p) for p in v] for k, v in self.charset.items()},
            "unknown_chars": [list(p) for p in self.unknown_chars],
            "groups": [list(g) for g in self.groups],
            "stats": self.stats,
        }

    def summary(self) -> str:
        lines = [
            f"{self.n_records} record(s): "
            f"{len(self.errors)} error(s), {len(self.warnings)} warning(s)"
        ]
        for issue in self.issues[:40]:
            lines.append(f"  {issue}")
        if len(self.issues) > 40:
            lines.append(f"  ... and {len(self.issues) - 40} more")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def _sniff(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in (".jsonl", ".ndjson"):
        return "jsonl"
    if suffix == ".json":
        return "json"
    if suffix == ".tsv":
        return "tsv"
    if suffix == ".csv":
        return "csv"
    raise ValueError(
        f"cannot tell the format of {path.name!r} from its extension; "
        "use .jsonl, .json, .csv, or .tsv"
    )


def read(path: str | Path) -> List[Dict[str, Any]]:
    """Read a manifest into a list of dicts.

    Blank lines are skipped.  A malformed JSONL line reports its own line
    number, because "Expecting value: line 1 column 1" on a 40k-line file is not
    a useful error message.
    """
    path = Path(path)
    kind = _sniff(path)  # an unsupported extension is a bug, report it first
    if not path.exists():
        raise FileNotFoundError(f"no manifest at {path}")

    if kind == "jsonl":
        records = []
        with path.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{path}:{lineno}: invalid JSON ({exc.msg})"
                    ) from None
        return records

    if kind == "json":
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            for key in ("records", "data", "samples", "segments"):
                if isinstance(data.get(key), list):
                    return list(data[key])
            raise ValueError(
                f"{path}: JSON object has no list under records/data/samples/segments"
            )
        if not isinstance(data, list):
            raise ValueError(f"{path}: expected a JSON list of records")
        return list(data)

    delimiter = "\t" if kind == "tsv" else ","
    with path.open(encoding="utf-8", newline="") as fh:
        return [dict(row) for row in csv.DictReader(fh, delimiter=delimiter)]


def write(path: str | Path, records: Iterable[Mapping[str, Any]]) -> int:
    """Write *records*, format chosen from the extension.  Returns the count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    kind = _sniff(path)
    rows = [dict(r) for r in records]

    if kind == "jsonl":
        with path.open("w", encoding="utf-8", newline="\n") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    elif kind == "json":
        with path.open("w", encoding="utf-8", newline="\n") as fh:
            json.dump(rows, fh, ensure_ascii=False, indent=2)
    else:
        delimiter = "\t" if kind == "tsv" else ","
        fieldnames: List[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with path.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter=delimiter)
            writer.writeheader()
            writer.writerows(rows)

    return len(rows)


# ---------------------------------------------------------------------------
# Quality control
# ---------------------------------------------------------------------------


def check(
    records: Sequence[Mapping[str, Any]],
    *,
    text_key: str = "text_warsh",
    id_key: str = "segment_id",
    group_key: str = "reciter_slug",
    surah_key: str = "surah_number",
    ayah_key: str = "ayah_number",
    min_chars: int = 2,
    max_chars: int = 1200,
) -> CheckReport:
    """Audit a manifest and return everything worth knowing about it."""
    report = CheckReport(n_records=len(records))
    issues = report.issues

    if not records:
        issues.append(Issue(ERROR, "empty-manifest", "the manifest has no records"))
        return report

    present = set()
    for record in records:
        present.update(record.keys())

    for required in (id_key, text_key):
        if required not in present:
            issues.append(
                Issue(ERROR, "missing-field", f"no record carries the field {required!r}")
            )

    seen_ids: Counter = Counter()
    charset: Counter = Counter()
    lengths: List[int] = []
    already_pausal = 0
    group_counter: Counter = Counter()
    surah_counter: Counter = Counter()

    for index, record in enumerate(records):
        segment_id = str(record.get(id_key) or "")
        if not segment_id:
            issues.append(
                Issue(ERROR, "missing-id", f"record {index} has no {id_key!r}", None)
            )
            segment_id = f"#{index}"
        seen_ids[segment_id] += 1

        raw_text = record.get(text_key)
        if raw_text is None or not str(raw_text).strip():
            issues.append(
                Issue(ERROR, "empty-text", f"{text_key!r} is empty", segment_id)
            )
            continue

        value = str(raw_text)
        charset.update(value)
        cleaned = T.to_clean(value)
        lengths.append(len(cleaned))

        if len(cleaned) < min_chars:
            issues.append(
                Issue(
                    WARNING,
                    "text-too-short",
                    f"only {len(cleaned)} character(s) after cleaning",
                    segment_id,
                )
            )
        elif len(cleaned) > max_chars:
            issues.append(
                Issue(
                    WARNING,
                    "text-too-long",
                    f"{len(cleaned)} characters after cleaning, over the {max_chars} limit",
                    segment_id,
                )
            )

        if not T.to_rasm(value):
            issues.append(
                Issue(ERROR, "no-letters", "text contains no Arabic letters", segment_id)
            )

        if T.ends_with_sukun(value):
            already_pausal += 1

        if group_key in record:
            group_counter[str(record[group_key])] += 1

        if surah_key in record and record[surah_key] not in (None, ""):
            surah_counter[str(record[surah_key])] += 1
            problem = surahs.validate_reference(
                record[surah_key], record.get(ayah_key)
            )
            if problem:
                issues.append(Issue(ERROR, "bad-reference", problem, segment_id))

    for segment_id, count in seen_ids.items():
        if count > 1:
            issues.append(
                Issue(
                    ERROR,
                    "duplicate-id",
                    f"appears {count} times -- ids must be unique or splits will leak",
                    segment_id,
                )
            )

    by_category: Dict[str, List[tuple]] = defaultdict(list)
    for char, count in charset.most_common():
        if char.isspace():
            continue
        by_category[C.category_of(char)].append((char, C.describe(char), count))
    report.charset = dict(by_category)
    report.unknown_chars = by_category.get("unknown", [])

    if report.unknown_chars:
        preview = ", ".join(desc for _, desc, _ in report.unknown_chars[:6])
        issues.append(
            Issue(
                WARNING,
                "unknown-chars",
                f"{len(report.unknown_chars)} uncategorised codepoint(s) will pass "
                f"through normalisation untouched: {preview}",
            )
        )

    if group_counter:
        report.groups = group_counter.most_common()
        singletons = [g for g, n in group_counter.items() if n < 3]
        if singletons:
            issues.append(
                Issue(
                    WARNING,
                    "tiny-group",
                    f"{len(singletons)} {group_key} value(s) have fewer than 3 segments "
                    "and cannot be spread across train/val/test: "
                    + ", ".join(sorted(singletons)[:5]),
                )
            )
    elif group_key not in present:
        issues.append(
            Issue(
                WARNING,
                "no-group-key",
                f"no {group_key!r} field -- stratified splitting by speaker is impossible, "
                "so validation scores will not detect voice memorisation",
            )
        )

    n_text = len(lengths)
    report.stats = {
        "records": len(records),
        "with_text": n_text,
        "unique_ids": len(seen_ids),
        "groups": len(group_counter),
        "surahs": len(surah_counter),
        "chars_total": sum(charset.values()),
        "distinct_chars": len([c for c in charset if not c.isspace()]),
        "mean_length": round(sum(lengths) / n_text, 1) if n_text else 0,
        "min_length": min(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
        "already_pausal": already_pausal,
        "already_pausal_pct": round(100 * already_pausal / n_text, 1) if n_text else 0.0,
    }

    if n_text and already_pausal / n_text < 0.5:
        issues.append(
            Issue(
                INFO,
                "labels-not-pausal",
                f"only {report.stats['already_pausal_pct']}% of references already end in "
                "a sukun; run them through warshlab.text.apply_waqf before training so "
                "the model is not taught to invent a final vowel",
            )
        )

    return report


def join_predictions(
    manifest: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    *,
    id_key: str = "segment_id",
    text_key: str = "text_warsh",
    prediction_key: str = "prediction",
) -> List[Dict[str, Any]]:
    """Inner-join a manifest with decoded predictions on *id_key*.

    Produces the ``reference``/``hypothesis`` records :func:`warshlab.metrics.evaluate`
    expects.  Predictions with no matching manifest row are dropped; the caller
    can spot that from the returned length.
    """
    by_id = {str(r.get(id_key)): r for r in manifest}
    joined: List[Dict[str, Any]] = []

    for prediction in predictions:
        key = str(prediction.get(id_key))
        source = by_id.get(key)
        if source is None:
            continue
        record = {
            "segment_id": key,
            "reference": str(source.get(text_key, "")),
            "hypothesis": str(prediction.get(prediction_key, "")),
        }
        for field_name, value in source.items():
            if field_name not in (id_key, text_key):
                record.setdefault(field_name, value)
        joined.append(record)

    return joined
