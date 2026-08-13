"""Dataset splitting for speech corpora with a strong speaker signal.

Two split strategies, answering two different questions:

:func:`stratified_split`
    Every reciter appears in every split.  Measures how well the model
    transcribes voices it has heard before -- the number that tracks training
    progress.
:func:`holdout_split`
    Whole reciters are held out.  Measures how well the model transcribes a
    voice it has never heard -- the number that predicts behaviour in the hands
    of an actual user.

Reporting only the first is how a model that has quietly memorised eleven voices
still looks excellent right up until release.  Build both.

Both are deterministic, and stable under corpus growth: shuffling is seeded per
group with a hash that does not depend on ``PYTHONHASHSEED`` or on how many
other groups exist, so adding a reciter next month does not reshuffle the
reciters you already split.
"""

from __future__ import annotations

import hashlib
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

__all__ = [
    "SplitResult",
    "stratified_split",
    "holdout_split",
    "check_leakage",
    "group_summary",
]

DEFAULT_NAMES = ("train", "val", "test")
DEFAULT_RATIOS = (0.8, 0.1, 0.1)


@dataclass
class SplitResult:
    """Named splits plus everything you need to sanity-check them."""

    splits: Dict[str, List[Any]]
    group_key: str
    warnings: List[str] = field(default_factory=list)
    groups_per_split: Dict[str, List[str]] = field(default_factory=dict)
    strategy: str = "stratified"
    seed: int = 42

    def __getitem__(self, name: str) -> List[Any]:
        return self.splits[name]

    @property
    def sizes(self) -> Dict[str, int]:
        return {name: len(items) for name, items in self.splits.items()}

    def summary(self) -> str:
        lines = [f"{self.strategy} split (seed={self.seed}, group={self.group_key})"]
        total = sum(self.sizes.values()) or 1
        for name, size in self.sizes.items():
            groups = len(self.groups_per_split.get(name, []))
            lines.append(
                f"  {name:<6} {size:>7} items  ({100 * size / total:5.1f}%)  "
                f"{groups:>4} groups"
            )
        for warning in self.warnings:
            lines.append(f"  ! {warning}")
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "seed": self.seed,
            "group_key": self.group_key,
            "sizes": self.sizes,
            "groups_per_split": self.groups_per_split,
            "warnings": self.warnings,
        }


def _seeded_shuffle(items: List[Any], seed: int, salt: str) -> List[Any]:
    """Shuffle deterministically from (seed, salt), independent of PYTHONHASHSEED.

    ``hash()`` on strings is randomised per process, so it cannot be used here
    without splits changing between runs.
    """
    digest = hashlib.sha256(f"{seed}:{salt}".encode("utf-8")).hexdigest()
    rng = random.Random(int(digest[:16], 16))
    shuffled = list(items)
    rng.shuffle(shuffled)
    return shuffled


def _allocate(n: int, ratios: Sequence[float]) -> List[int]:
    """Split *n* items across *ratios* by largest remainder.

    Guarantees the counts sum to *n* exactly, and gives every split at least one
    item when ``n >= len(ratios)`` -- a split that silently ends up empty turns
    into a divide-by-zero three steps later.
    """
    k = len(ratios)
    total_ratio = sum(ratios)
    exact = [n * r / total_ratio for r in ratios]
    counts = [int(x) for x in exact]
    remainder = n - sum(counts)

    order = sorted(range(k), key=lambda i: exact[i] - counts[i], reverse=True)
    for i in range(remainder):
        counts[order[i % k]] += 1

    if n >= k:
        while any(c == 0 for c in counts):
            donor = max(range(k), key=lambda i: counts[i])
            receiver = min(range(k), key=lambda i: counts[i])
            if counts[donor] <= 1:
                break
            counts[donor] -= 1
            counts[receiver] += 1

    return counts


def _key_getter(group_key: str) -> Callable[[Any], str]:
    def get(item: Any) -> str:
        if isinstance(item, Mapping):
            value = item.get(group_key)
        else:
            value = getattr(item, group_key, None)
        if value is None:
            raise KeyError(f"item is missing group key {group_key!r}: {item!r}")
        return str(value)

    return get


def stratified_split(
    items: Sequence[Any],
    *,
    group_key: str = "reciter_slug",
    ratios: Sequence[float] = DEFAULT_RATIOS,
    names: Sequence[str] = DEFAULT_NAMES,
    seed: int = 42,
    shuffle_output: bool = True,
) -> SplitResult:
    """Split *items* so that every group is represented in every split.

    Groups smaller than the number of splits cannot be spread across all of them;
    those go entirely to the first split and are reported in
    :attr:`SplitResult.warnings` rather than raising, so one two-clip reciter
    does not block the whole run.

    Output order is interleaved across groups (``shuffle_output=True``), so
    training never sees one voice for a long consecutive stretch.
    """
    if len(ratios) != len(names):
        raise ValueError(
            f"got {len(ratios)} ratios for {len(names)} split names -- they must match"
        )
    if any(r < 0 for r in ratios) or sum(ratios) <= 0:
        raise ValueError(f"ratios must be non-negative and sum to > 0, got {ratios!r}")

    get_group = _key_getter(group_key)
    by_group: Dict[str, List[Any]] = defaultdict(list)
    for item in items:
        by_group[get_group(item)].append(item)

    splits: Dict[str, List[Any]] = {name: [] for name in names}
    warnings: List[str] = []

    for group in sorted(by_group):
        members = _seeded_shuffle(by_group[group], seed, group)
        n = len(members)

        if n < len(names):
            splits[names[0]].extend(members)
            warnings.append(
                f"group {group!r} has {n} item(s), fewer than the {len(names)} splits; "
                f"all of it went to {names[0]!r} and it cannot be evaluated"
            )
            continue

        cursor = 0
        for name, count in zip(names, _allocate(n, ratios)):
            splits[name].extend(members[cursor : cursor + count])
            cursor += count

    if shuffle_output:
        for name in names:
            splits[name] = _seeded_shuffle(splits[name], seed, f"__output__{name}")

    groups_per_split = {
        name: sorted({get_group(item) for item in bucket})
        for name, bucket in splits.items()
    }

    all_groups = set(by_group)
    for name in names[1:]:
        missing = all_groups - set(groups_per_split[name])
        if missing:
            warnings.append(
                f"{len(missing)} group(s) absent from {name!r}: "
                + ", ".join(sorted(missing)[:5])
                + ("..." if len(missing) > 5 else "")
            )

    return SplitResult(
        splits=splits,
        group_key=group_key,
        warnings=warnings,
        groups_per_split=groups_per_split,
        strategy="stratified",
        seed=seed,
    )


def holdout_split(
    items: Sequence[Any],
    *,
    group_key: str = "reciter_slug",
    holdout: Sequence[str] = (),
    n_holdout: int = 1,
    name: str = "unseen",
    seed: int = 42,
) -> SplitResult:
    """Hold out whole groups, giving a zero-shot evaluation set.

    Pass explicit group names via *holdout*, or let it pick *n_holdout* groups
    deterministically.  Groups are picked from the middle of the size ranking
    rather than the extremes, so the unseen set is neither the one reciter with
    half the corpus nor a group with nine clips in it.
    """
    get_group = _key_getter(group_key)
    by_group: Dict[str, List[Any]] = defaultdict(list)
    for item in items:
        by_group[get_group(item)].append(item)

    known = set(by_group)
    warnings: List[str] = []

    if holdout:
        chosen = []
        for group in holdout:
            if group in known:
                chosen.append(group)
            else:
                warnings.append(f"requested holdout group {group!r} is not in the data")
    else:
        if n_holdout >= len(known):
            raise ValueError(
                f"cannot hold out {n_holdout} of {len(known)} group(s) -- "
                "nothing would be left to train on"
            )
        ranked = sorted(known, key=lambda g: (-len(by_group[g]), g))
        middle = ranked[1:-1] or ranked
        chosen = _seeded_shuffle(middle, seed, "__holdout__")[:n_holdout]

    chosen_set = set(chosen)
    if not chosen_set:
        warnings.append("no groups were held out; the unseen split is empty")

    seen_items = [i for i in items if get_group(i) not in chosen_set]
    unseen_items = [i for i in items if get_group(i) in chosen_set]

    splits = {
        "seen": _seeded_shuffle(seen_items, seed, "__output__seen"),
        name: _seeded_shuffle(unseen_items, seed, f"__output__{name}"),
    }
    groups_per_split = {
        key: sorted({get_group(i) for i in bucket}) for key, bucket in splits.items()
    }

    return SplitResult(
        splits=splits,
        group_key=group_key,
        warnings=warnings,
        groups_per_split=groups_per_split,
        strategy="holdout",
        seed=seed,
    )


def check_leakage(
    splits: Mapping[str, Sequence[Any]], *, id_key: str = "segment_id"
) -> List[str]:
    """Return a problem list: ids appearing in more than one split, or duplicated.

    An empty list means the splits are disjoint.  Call it after every split and
    after every manual edit to one -- overlapping splits inflate validation
    scores in a way nothing downstream will flag.
    """
    get_id = _key_getter(id_key)
    seen: Dict[str, str] = {}
    problems: List[str] = []
    duplicate_within: Counter = Counter()

    for split_name, bucket in splits.items():
        local: set = set()
        for item in bucket:
            item_id = get_id(item)
            if item_id in local:
                duplicate_within[(split_name, item_id)] += 1
                continue
            local.add(item_id)
            if item_id in seen:
                problems.append(
                    f"id {item_id!r} appears in both {seen[item_id]!r} and {split_name!r}"
                )
            else:
                seen[item_id] = split_name

    for (split_name, item_id), count in duplicate_within.items():
        problems.append(
            f"id {item_id!r} appears {count + 1} times within {split_name!r}"
        )

    return problems


def group_summary(items: Sequence[Any], *, group_key: str = "reciter_slug") -> List[Tuple[str, int]]:
    """``(group, count)`` pairs, largest first."""
    get_group = _key_getter(group_key)
    return Counter(get_group(i) for i in items).most_common()
