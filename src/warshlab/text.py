"""Normalisation of Warsh mushaf text into ASR-friendly label forms.

The functions here are deterministic, dependency-free, and idempotent where that
makes sense (``to_rasm(to_rasm(x)) == to_rasm(x)``).

Normal forms
------------
``raw``
    Input with invisible formatting removed and whitespace collapsed.  Nothing
    else is touched.
``clean``
    ``raw`` minus editorial glyphs the reciter never pronounces: waqf pause
    signs, ayah/hizb/sajdah ornaments, and (by default) tajweed indicators.
``waqf``
    ``clean`` with the utterance put into pausal form -- the final vowel is
    replaced by a sukun.  This is the label form for segment-level ASR, where
    every clip ends at a stop.
``rasm``
    Letters and spaces only; every vowel and mark stripped.  Use it to score
    consonantal accuracy independently of diacritisation.
``skeleton``
    ``rasm`` with letter variants unified (alef forms, teh marbuta, alef
    maksura).  The most forgiving comparison form.
``harakat``
    Only the vowel marks, in order, with word boundaries kept.  Scoring this
    isolates diacritisation accuracy.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, List, Set

from . import chars as C

__all__ = [
    "WaqfConfig",
    "DEFAULT_WAQF",
    "NOTEBOOK_WAQF",
    "FORMS",
    "remove_categories",
    "collapse_whitespace",
    "to_raw",
    "to_clean",
    "to_rasm",
    "to_skeleton",
    "to_harakat",
    "apply_waqf",
    "normalize",
    "unify_letters",
    "words",
    "ends_with_sukun",
]

_WS = re.compile(r"\s+")

#: Normal forms understood by :func:`normalize` and the ``--form`` CLI flag.
FORMS = ("raw", "clean", "waqf", "rasm", "skeleton", "harakat")


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def remove_categories(text: str, categories: Iterable[str]) -> str:
    """Drop every character whose category is in *categories*."""
    drop = set(categories)
    return "".join(ch for ch in text if C.category_of(ch) not in drop)


def collapse_whitespace(text: str) -> str:
    """Collapse runs of whitespace to a single space and strip the ends."""
    return _WS.sub(" ", text).strip()


def to_raw(text: str) -> str:
    """Remove invisible controls and tatweel; collapse whitespace; NFC.

    This is the floor every other form is built on -- it never changes what is
    pronounced, so it is always safe to apply.
    """
    text = unicodedata.normalize("NFC", text)
    text = remove_categories(text, [C.FORMATTING])
    return collapse_whitespace(text)


def to_clean(text: str, *, drop_tajweed: bool = True) -> str:
    """``to_raw`` minus editorial marks the reciter does not pronounce.

    Waqf pause signs and ornaments (end-of-ayah, rub el hizb, sajdah) are always
    removed: they are typographic guidance, not sound.  Tajweed indicators
    (iqlab meem, silent-letter zeros, sakta seen) are removed by default because
    an acoustic model cannot reliably distinguish them, but keep them with
    ``drop_tajweed=False`` if your labels encode them deliberately.
    """
    drop = [C.WAQF_MARK, C.ORNAMENT]
    if drop_tajweed:
        drop.append(C.TAJWEED_MARK)
    return collapse_whitespace(remove_categories(to_raw(text), drop))


def to_rasm(text: str) -> str:
    """Letters and spaces only -- the consonantal skeleton, marks stripped."""
    kept = "".join(
        ch if (C.is_letter(ch) or ch.isspace()) else "" for ch in to_raw(text)
    )
    return collapse_whitespace(kept)


#: Letter-variant folding used by :func:`unify_letters`.
_UNIFY = {
    "أ": "ا",  # ALEF WITH HAMZA ABOVE
    "إ": "ا",  # ALEF WITH HAMZA BELOW
    "آ": "ا",  # ALEF WITH MADDA ABOVE
    "ٱ": "ا",  # ALEF WASLA
    "ى": "ي",  # ALEF MAKSURA
    "ة": "ه",  # TEH MARBUTA
    "ؤ": "و",  # WAW WITH HAMZA
    "ئ": "ي",  # YEH WITH HAMZA
    "ے": "ي",  # YEH BARREE
    "ۓ": "ي",  # YEH BARREE WITH HAMZA
}


def unify_letters(text: str) -> str:
    """Fold orthographic letter variants onto a single representative.

    Hamzated alefs and alef wasla become bare alef, alef maksura becomes yeh,
    teh marbuta becomes heh.  These distinctions are orthographic rather than
    acoustic, so folding them keeps a model from being penalised for a spelling
    choice it cannot hear.
    """
    return "".join(_UNIFY.get(ch, ch) for ch in text)


def to_skeleton(text: str) -> str:
    """``to_rasm`` with letter variants unified -- the most forgiving form."""
    return unify_letters(to_rasm(text))


def to_harakat(text: str) -> str:
    """Keep only vowel marks (and word boundaries), dropping the letters.

    Comparing two texts in this form scores diacritisation on its own: a model
    that hears every consonant but guesses the vowels will look perfect in
    :func:`to_rasm` and poor here.
    """
    keep = {C.HARAKA, C.SUPERSCRIPT_LETTER, C.HAMZA_MARK}
    out: List[str] = []
    for ch in to_raw(text):
        if ch.isspace():
            out.append(" ")
        elif C.category_of(ch) in keep:
            out.append(ch)
    return collapse_whitespace("".join(out))


def words(text: str) -> List[str]:
    """Split on whitespace after :func:`to_raw`."""
    cleaned = to_raw(text)
    return cleaned.split() if cleaned else []


# ---------------------------------------------------------------------------
# Pausal form (waqf)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WaqfConfig:
    """Rules for :func:`apply_waqf`.

    Attributes
    ----------
    drop_tajweed:
        Passed through to :func:`to_clean`.
    sukun_carriers:
        Which characters may receive the final sukun.  ``"letters"`` (default)
        restricts it to base letters.  ``"letters+superscript"`` also allows the
        miniature letters (small waw/yeh, superscript alef) -- the behaviour of
        the original training notebook, kept for label-set reproducibility.
    skip_madd_carriers:
        When True, a word ending in a bare long-vowel carrier (alef, waw, yeh,
        alef maksura) that is *not* itself vowelled gets no sukun, matching the
        recitation rule that a final madd is lengthened rather than stopped.
        Off by default because it changes an existing label convention.
    teh_marbuta_to_heh:
        When True, a final teh marbuta is rewritten as heh, which is how it is
        pronounced in pause.  Off by default -- it makes labels diverge from the
        mushaf orthography.
    sukun:
        The character appended.  Defaults to U+0652 ARABIC SUKUN; set it to
        :data:`warshlab.chars.QURANIC_SUKUN` to match mushaf typography.
    """

    drop_tajweed: bool = True
    sukun_carriers: str = "letters"
    skip_madd_carriers: bool = False
    teh_marbuta_to_heh: bool = False
    sukun: str = C.SUKUN

    def carriers(self) -> Set[str]:
        if self.sukun_carriers == "letters":
            return C.LETTERS
        if self.sukun_carriers == "letters+superscript":
            return C.LETTERS | C.SUPERSCRIPT_LETTERS
        raise ValueError(
            f"unknown sukun_carriers {self.sukun_carriers!r}; "
            "expected 'letters' or 'letters+superscript'"
        )


#: Recommended defaults.
DEFAULT_WAQF = WaqfConfig()

#: Reproduces the behaviour of the original ``whisper_warsh_training`` notebook,
#: so previously generated labels stay comparable.
NOTEBOOK_WAQF = WaqfConfig(sukun_carriers="letters+superscript")

_MADD_CARRIERS = {"ا", "و", "ي", "ى", "ٱ", "ے"}


def apply_waqf(text: str, config: WaqfConfig = DEFAULT_WAQF) -> str:
    """Put *text* into pausal form: replace the final vowel with a sukun.

    Every clip in a segment-level recitation corpus ends at a stop, so its label
    must end in pausal form.  Training on non-pausal labels teaches the model to
    hallucinate a final vowel it never hears.

    The transformation is: clean the text, walk back to the last sukun carrier,
    discard the marks that trailed it (they spell the vowel the pause removes),
    then append the sukun.

    Returns the empty string for input with no letters.
    """
    cleaned = to_clean(text, drop_tajweed=config.drop_tajweed)
    if not cleaned:
        return ""

    carriers = config.carriers()
    last = -1
    for i in range(len(cleaned) - 1, -1, -1):
        if cleaned[i] in carriers:
            last = i
            break

    if last < 0:  # nothing to hang a sukun on
        return cleaned

    stem = cleaned[: last + 1]
    final = cleaned[last]

    if config.teh_marbuta_to_heh and final == C.TEH_MARBUTA:
        stem = stem[:-1] + C.HEH
        final = C.HEH

    if config.skip_madd_carriers and final in _MADD_CARRIERS:
        # A final madd is drawn out, not stopped -- but only when the carrier is
        # bare.  A vowelled or shadda'd carrier is a real consonant.
        preceding_marks = _marks_before(cleaned, last)
        if not preceding_marks:
            return stem

    if stem.endswith(config.sukun):
        return stem
    return stem + config.sukun


def _marks_before(text: str, index: int) -> List[str]:
    """Marks attached to the character at *index* (i.e. sitting right after it).

    Combining marks follow their base in logical order, so the marks belonging
    to ``text[index]`` are the run starting at ``index + 1``.
    """
    out: List[str] = []
    for ch in text[index + 1 :]:
        if C.is_mark(ch):
            out.append(ch)
        else:
            break
    return out


def ends_with_sukun(text: str, config: WaqfConfig = DEFAULT_WAQF) -> bool:
    """True if *text* is already in pausal form."""
    stripped = to_clean(text, drop_tajweed=config.drop_tajweed)
    return stripped.endswith(config.sukun) or stripped.endswith(C.QURANIC_SUKUN)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


def normalize(text: str, form: str = "waqf", *, config: WaqfConfig = DEFAULT_WAQF) -> str:
    """Apply the normal form named by *form* (see :data:`FORMS`)."""
    if form == "raw":
        return to_raw(text)
    if form == "clean":
        return to_clean(text, drop_tajweed=config.drop_tajweed)
    if form == "waqf":
        return apply_waqf(text, config)
    if form == "rasm":
        return to_rasm(text)
    if form == "skeleton":
        return to_skeleton(text)
    if form == "harakat":
        return to_harakat(text)
    raise ValueError(f"unknown form {form!r}; expected one of {', '.join(FORMS)}")
