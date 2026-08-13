"""A synthetic corpus, so the whole pipeline runs before your data arrives.

``warsh demo`` builds a manifest, splits it, fabricates predictions with a
seeded error model, scores them, and writes a report -- proving the toolchain
end to end on a machine with no dataset, no GPU, and no network.

The sample text is short, well-known Quranic passages in common Uthmani
diacritisation.  It is here to exercise the code paths, **not** as a Warsh
reference: Warsh orthography differs from what is written below in ways that
matter.  Point the tools at your own corpus for real work.

The error model is deliberately shaped like a real ASR failure mode -- mostly
diacritic slips, some orthographic variant confusion, occasional dropped or
doubled letters -- with a different error rate per reciter, so the per-reciter
table in the report has something to show.
"""

from __future__ import annotations

import hashlib
import random
from typing import Dict, List, Sequence, Tuple

from . import chars as C
from . import text as T

__all__ = ["VERSES", "RECITERS", "build_manifest", "fake_predictions"]

#: ``(surah, ayah, text)``.  Illustrative sample text -- see the module docstring.
VERSES: Tuple[Tuple[int, int, str], ...] = (
    (1, 1, "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"),
    (1, 2, "الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ"),
    (1, 3, "الرَّحْمَٰنِ الرَّحِيمِ"),
    (1, 4, "مَالِكِ يَوْمِ الدِّينِ"),
    (1, 5, "إِيَّاكَ نَعْبُدُ وَإِيَّاكَ نَسْتَعِينُ"),
    (1, 6, "اهْدِنَا الصِّرَاطَ الْمُسْتَقِيمَ"),
    (1, 7, "صِرَاطَ الَّذِينَ أَنْعَمْتَ عَلَيْهِمْ غَيْرِ الْمَغْضُوبِ عَلَيْهِمْ وَلَا الضَّالِّينَ"),
    (103, 1, "وَالْعَصْرِ"),
    (103, 2, "إِنَّ الْإِنْسَانَ لَفِي خُسْرٍ"),
    (103, 3, "إِلَّا الَّذِينَ آمَنُوا وَعَمِلُوا الصَّالِحَاتِ وَتَوَاصَوْا بِالْحَقِّ وَتَوَاصَوْا بِالصَّبْرِ"),
    (112, 1, "قُلْ هُوَ اللَّهُ أَحَدٌ"),
    (112, 2, "اللَّهُ الصَّمَدُ"),
    (112, 3, "لَمْ يَلِدْ وَلَمْ يُولَدْ"),
    (112, 4, "وَلَمْ يَكُنْ لَهُ كُفُوًا أَحَدٌ"),
    (113, 1, "قُلْ أَعُوذُ بِرَبِّ الْفَلَقِ"),
    (113, 2, "مِنْ شَرِّ مَا خَلَقَ"),
    (114, 1, "قُلْ أَعُوذُ بِرَبِّ النَّاسِ"),
    (114, 2, "مَلِكِ النَّاسِ"),
    (114, 3, "إِلَٰهِ النَّاسِ"),
    (114, 4, "مِنْ شَرِّ الْوَسْوَاسِ الْخَنَّاسِ"),
    (114, 5, "الَّذِي يُوَسْوِسُ فِي صُدُورِ النَّاسِ"),
    (114, 6, "مِنَ الْجِنَّةِ وَالنَّاسِ"),
)

#: ``(slug, relative error rate)`` -- the spread is what makes the per-reciter
#: table in the report worth looking at.
RECITERS: Tuple[Tuple[str, float], ...] = (
    ("reciter-alpha", 0.6),
    ("reciter-beta", 1.0),
    ("reciter-gamma", 1.9),
    ("reciter-delta", 3.2),
)

_VOWELS = ["َ", "ُ", "ِ", "ْ", "ً", "ٌ", "ٍ"]
_ORTHOGRAPHIC = {"ة": "ه", "ى": "ي", "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا"}


def _rng(*parts: object) -> random.Random:
    """A generator seeded from the arguments -- reproducible across processes."""
    digest = hashlib.sha256(":".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def build_manifest(
    *, repeats: int = 1, reciters: Sequence[Tuple[str, float]] = RECITERS
) -> List[Dict[str, object]]:
    """Build a manifest: every reciter reads every verse, *repeats* times over."""
    records: List[Dict[str, object]] = []
    for slug, _ in reciters:
        for take in range(repeats):
            for surah, ayah, verse in VERSES:
                suffix = f"-t{take + 1}" if repeats > 1 else ""
                records.append(
                    {
                        "segment_id": f"{slug}-{surah:03d}-{ayah:03d}{suffix}",
                        "reciter_slug": slug,
                        "surah_number": surah,
                        "ayah_number": ayah,
                        "text_warsh": verse,
                        "duration_s": round(0.45 * len(T.to_rasm(verse)) / 3.2 + 1.2, 2),
                    }
                )
    return records


def _corrupt(reference: str, severity: float, rng: random.Random) -> str:
    """Apply ASR-shaped noise to *reference* at the given *severity*."""
    out: List[str] = []
    base = 0.045 * severity

    for char in reference:
        roll = rng.random()

        if C.category_of(char) == C.HARAKA:
            if roll < base * 1.8:  # dropped diacritic -- the commonest failure
                continue
            if roll < base * 2.6:  # wrong diacritic
                out.append(rng.choice(_VOWELS))
                continue

        elif char in _ORTHOGRAPHIC and roll < base * 1.4:
            out.append(_ORTHOGRAPHIC[char])
            continue

        elif C.is_letter(char):
            if roll < base * 0.30:  # dropped letter
                continue
            if roll < base * 0.45:  # doubled letter
                out.append(char)

        out.append(char)

    text = "".join(out)
    if rng.random() < 0.05 * severity:  # truncated tail
        text = text[: max(1, int(len(text) * rng.uniform(0.6, 0.95)))]
    return T.collapse_whitespace(text)


def fake_predictions(
    manifest: Sequence[Dict[str, object]],
    *,
    seed: int = 7,
    reciters: Sequence[Tuple[str, float]] = RECITERS,
) -> List[Dict[str, object]]:
    """Fabricate a decoded-predictions file for *manifest*.

    References are put into pausal form first, so the synthetic hypotheses look
    like the output of a model trained on waqf labels.
    """
    severity = dict(reciters)
    predictions: List[Dict[str, object]] = []

    for record in manifest:
        segment_id = str(record["segment_id"])
        reference = T.apply_waqf(str(record["text_warsh"]))
        rng = _rng(seed, segment_id)
        level = severity.get(str(record.get("reciter_slug")), 1.0)
        predictions.append(
            {"segment_id": segment_id, "prediction": _corrupt(reference, level, rng)}
        )

    return predictions
