"""warsh-lab -- tooling for Warsh Quran recitation ASR.

A zero-dependency Python library and CLI for the parts of a speech-recognition
project that sit either side of the GPU: normalising mushaf text into labels,
splitting a corpus without leaking speakers, scoring transcriptions in a way
that separates acoustic errors from diacritisation errors, and turning the
result into a report someone can actually read.

Quick start
-----------
>>> from warshlab import text, metrics
>>> text.apply_waqf("مِنْ غَيْرِكُمُۥٓ")
'مِنْ غَيْرِكُمْ'
>>> round(metrics.cer("ٱلْحَمْدُ", "الحمد"), 3)
0.5

Everything runs offline and imports nothing outside the standard library, so it
works on a training box with no wheels installed and no network.
"""

from __future__ import annotations

__version__ = "0.1.0"

from . import chars, distance, manifest, metrics, report, splits, surahs, text
from .metrics import cer, evaluate, wer
from .splits import holdout_split, stratified_split
from .text import apply_waqf, normalize, to_rasm, to_skeleton

__all__ = [
    "__version__",
    "chars",
    "distance",
    "manifest",
    "metrics",
    "report",
    "splits",
    "surahs",
    "text",
    "apply_waqf",
    "normalize",
    "to_rasm",
    "to_skeleton",
    "cer",
    "wer",
    "evaluate",
    "stratified_split",
    "holdout_split",
]
