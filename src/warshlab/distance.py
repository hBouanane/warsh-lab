"""Levenshtein distance, alignment, and error counts over arbitrary sequences.

Written against ``Sequence`` rather than ``str`` so the same code backs both
character-level (CER) and word-level (WER) scoring.  Pure stdlib -- no jiwer, no
editdistance, nothing to install on the training box.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple

__all__ = [
    "EQUAL",
    "SUB",
    "INSERT",
    "DELETE",
    "ErrorCounts",
    "edit_distance",
    "align",
    "opcodes",
    "error_counts",
]

EQUAL = "equal"
SUB = "sub"
INSERT = "insert"  # present in hypothesis, absent from reference
DELETE = "delete"  # present in reference, absent from hypothesis


@dataclass(frozen=True)
class ErrorCounts:
    """Aligned-edit tallies for one reference/hypothesis pair."""

    hits: int = 0
    substitutions: int = 0
    insertions: int = 0
    deletions: int = 0

    @property
    def errors(self) -> int:
        return self.substitutions + self.insertions + self.deletions

    @property
    def ref_length(self) -> int:
        return self.hits + self.substitutions + self.deletions

    @property
    def rate(self) -> float:
        """Error rate against reference length.

        An empty reference scores 0.0 against an empty hypothesis and 1.0
        against any non-empty one, rather than dividing by zero.
        """
        if self.ref_length == 0:
            return 0.0 if self.insertions == 0 else 1.0
        return self.errors / self.ref_length

    def __add__(self, other: "ErrorCounts") -> "ErrorCounts":
        return ErrorCounts(
            hits=self.hits + other.hits,
            substitutions=self.substitutions + other.substitutions,
            insertions=self.insertions + other.insertions,
            deletions=self.deletions + other.deletions,
        )

    def as_dict(self) -> dict:
        return {
            "hits": self.hits,
            "substitutions": self.substitutions,
            "insertions": self.insertions,
            "deletions": self.deletions,
            "errors": self.errors,
            "ref_length": self.ref_length,
            "rate": round(self.rate, 6),
        }


def _matrix(ref: Sequence[Any], hyp: Sequence[Any]) -> List[List[int]]:
    n, m = len(ref), len(hyp)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = i
    for j in range(1, m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        row, prev = d[i], d[i - 1]
        ref_i = ref[i - 1]
        for j in range(1, m + 1):
            if ref_i == hyp[j - 1]:
                row[j] = prev[j - 1]
            else:
                row[j] = 1 + min(prev[j - 1], prev[j], row[j - 1])
    return d


def edit_distance(ref: Sequence[Any], hyp: Sequence[Any]) -> int:
    """Levenshtein distance with unit costs.

    Uses two rolling rows -- no backtrace, so it stays O(min(n, m)) in memory.
    """
    if len(ref) < len(hyp):
        ref, hyp = hyp, ref
    if not hyp:
        return len(ref)

    previous = list(range(len(hyp) + 1))
    for i, ref_i in enumerate(ref, start=1):
        current = [i] + [0] * len(hyp)
        for j, hyp_j in enumerate(hyp, start=1):
            if ref_i == hyp_j:
                current[j] = previous[j - 1]
            else:
                current[j] = 1 + min(previous[j - 1], previous[j], current[j - 1])
        previous = current
    return previous[-1]


def align(ref: Sequence[Any], hyp: Sequence[Any]) -> List[Tuple[str, Any, Any]]:
    """Return the edit script turning *ref* into *hyp*, left to right.

    Each entry is ``(op, ref_item, hyp_item)``; the item not involved in the
    operation is ``None``.  Ties are resolved toward substitution, then
    deletion, then insertion, so the script stays stable across runs.
    """
    d = _matrix(ref, hyp)
    i, j = len(ref), len(hyp)
    script: List[Tuple[str, Any, Any]] = []

    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == hyp[j - 1] and d[i][j] == d[i - 1][j - 1]:
            script.append((EQUAL, ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + 1:
            script.append((SUB, ref[i - 1], hyp[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            script.append((DELETE, ref[i - 1], None))
            i -= 1
        else:
            script.append((INSERT, None, hyp[j - 1]))
            j -= 1

    script.reverse()
    return script


def opcodes(script: Sequence[Tuple[str, Any, Any]]) -> List[Tuple[str, List[Any], List[Any]]]:
    """Collapse an edit script into runs of the same operation.

    Handy for rendering diffs: ``[("equal", [...], [...]), ("sub", [...], [...])]``.
    """
    runs: List[Tuple[str, List[Any], List[Any]]] = []
    for op, ref_item, hyp_item in script:
        if runs and runs[-1][0] == op:
            if ref_item is not None:
                runs[-1][1].append(ref_item)
            if hyp_item is not None:
                runs[-1][2].append(hyp_item)
            continue
        runs.append(
            (
                op,
                [ref_item] if ref_item is not None else [],
                [hyp_item] if hyp_item is not None else [],
            )
        )
    return runs


def error_counts(ref: Sequence[Any], hyp: Sequence[Any]) -> ErrorCounts:
    """Tally hits and the three error types for one pair.

    Fast paths: identical sequences and either side empty are answered without
    building a matrix.
    """
    if list(ref) == list(hyp):
        return ErrorCounts(hits=len(ref))
    if not ref:
        return ErrorCounts(insertions=len(hyp))
    if not hyp:
        return ErrorCounts(deletions=len(ref))

    counts = {EQUAL: 0, SUB: 0, INSERT: 0, DELETE: 0}
    for op, _, _ in align(ref, hyp):
        counts[op] += 1
    return ErrorCounts(
        hits=counts[EQUAL],
        substitutions=counts[SUB],
        insertions=counts[INSERT],
        deletions=counts[DELETE],
    )
