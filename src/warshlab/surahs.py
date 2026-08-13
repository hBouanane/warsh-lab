"""Surah metadata, used to validate corpus references.

Data ships with the package (``warshlab/data/surahs.json``), so lookups work
offline.  The point is catching bad references before training: a manifest with
``surah_number: 115`` or ``ayah: 8`` in Al-Fatiha is a bug in the segmentation
pipeline, and it is much cheaper to find here than in a confusion matrix.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

__all__ = ["Surah", "SURAHS", "get", "find", "ayah_count", "validate_reference", "total_ayahs"]

_DATA = Path(__file__).parent / "data" / "surahs.json"


@dataclass(frozen=True)
class Surah:
    number: int
    name: str
    english_name: str
    english_translation: str
    ayah_count: int
    revelation_type: str

    def __str__(self) -> str:
        return f"{self.number}. {self.english_name} ({self.ayah_count} ayat)"


def _load() -> List[Surah]:
    with _DATA.open(encoding="utf-8") as fh:
        raw = json.load(fh)
    return [
        Surah(
            number=int(entry["number"]),
            name=entry["name"],
            english_name=entry["englishName"],
            english_translation=entry["englishNameTranslation"],
            ayah_count=int(entry["numberOfAyahs"]),
            revelation_type=entry["revelationType"],
        )
        for entry in raw
    ]


SURAHS: List[Surah] = _load()

_BY_NUMBER: Dict[int, Surah] = {s.number: s for s in SURAHS}


def get(number: int) -> Surah:
    """Look up a surah by its number (1-114). Raises :class:`KeyError` otherwise."""
    try:
        return _BY_NUMBER[int(number)]
    except (KeyError, TypeError, ValueError):
        raise KeyError(f"no surah numbered {number!r}; expected 1-114") from None


@lru_cache(maxsize=256)
def find(query: str) -> Optional[Surah]:
    """Find a surah by number, Arabic name, or English name (case-insensitive)."""
    text = str(query).strip()
    if not text:
        return None
    if text.isdigit():
        return _BY_NUMBER.get(int(text))

    needle = text.casefold().replace("-", "").replace(" ", "")
    for surah in SURAHS:
        candidates = (surah.english_name, surah.english_translation, surah.name)
        for candidate in candidates:
            if candidate.casefold().replace("-", "").replace(" ", "") == needle:
                return surah
    for surah in SURAHS:
        if needle in surah.english_name.casefold().replace("-", "").replace(" ", ""):
            return surah
    return None


def ayah_count(number: int) -> int:
    """Number of ayat in the given surah."""
    return get(number).ayah_count


def total_ayahs() -> int:
    """6236 -- the ayah count of the whole mushaf, computed from the table."""
    return sum(s.ayah_count for s in SURAHS)


def validate_reference(surah_number: object, ayah_number: object = None) -> Optional[str]:
    """Return an error message if the reference is invalid, else ``None``."""
    try:
        number = int(str(surah_number))
    except (TypeError, ValueError):
        return f"surah_number {surah_number!r} is not an integer"

    if number not in _BY_NUMBER:
        return f"surah_number {number} is out of range (1-114)"

    if ayah_number is None or ayah_number == "":
        return None

    try:
        ayah = int(str(ayah_number))
    except (TypeError, ValueError):
        return f"ayah number {ayah_number!r} is not an integer"

    limit = _BY_NUMBER[number].ayah_count
    if not 1 <= ayah <= limit:
        return (
            f"ayah {ayah} is out of range for surah {number} "
            f"({_BY_NUMBER[number].english_name} has {limit} ayat)"
        )
    return None
