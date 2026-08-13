"""Unicode character tables for Quranic Arabic (Warsh orthography).

Every codepoint that occurs in Warsh mushaf text is assigned to exactly one
category.  The categories are *disjoint by construction* -- ``tests/test_chars.py``
asserts it -- which is the whole point of this module.

The motivating bug: hand-rolled normalisers usually keep two ad-hoc sets, one of
"letters" and one of "marks to strip", written as opaque string literals.  Those
sets silently overlap (e.g. U+06E7 SMALL HIGH YEH landing in both), and which
behaviour wins then depends on the order the sets happen to be applied in.  Here
each codepoint is declared once, with its Unicode name next to it.

Categories
----------
``LETTER``
    Base consonants and long-vowel carriers.  These are the only characters that
    can carry a sukun, and the only ones that survive :func:`~warshlab.text.to_rasm`.
``SUPERSCRIPT_LETTER``
    Miniature letters that spell a vowel that is pronounced but not written in
    the rasm (superscript alef, small waw, small yeh).  Combining marks, *not*
    sukun carriers.
``HARAKA``
    Short vowels, tanween, shadda, sukun.
``HAMZA_MARK``
    Maddah and the free-standing hamza marks.
``TAJWEED_MARK``
    Idgham / ikhfa / iqlab indicators and the "silent letter" zeros.
``WAQF_MARK``
    Pause signs -- editorial guidance for the reciter, never pronounced.
``ORNAMENT``
    Structural glyphs: end of ayah, rub el hizb, place of sajdah.
``FORMATTING``
    Zero-width and bidi controls, tatweel.
``DIGIT`` / ``PUNCT``
    Arabic-Indic digits and punctuation.
"""

from __future__ import annotations

from typing import Dict, Iterable, Set

__all__ = [
    "LETTER",
    "SUPERSCRIPT_LETTER",
    "HARAKA",
    "HAMZA_MARK",
    "TAJWEED_MARK",
    "WAQF_MARK",
    "ORNAMENT",
    "FORMATTING",
    "DIGIT",
    "PUNCT",
    "CATEGORIES",
    "CATEGORY_OF",
    "LETTERS",
    "SUPERSCRIPT_LETTERS",
    "HARAKAT",
    "HAMZA_MARKS",
    "TAJWEED_MARKS",
    "WAQF_MARKS",
    "ORNAMENTS",
    "FORMATTING_CHARS",
    "DIGITS",
    "PUNCTUATION",
    "ALL_MARKS",
    "SUKUN",
    "SHADDA",
    "TATWEEL",
    "category_of",
    "is_letter",
    "is_mark",
    "describe",
    "unknown_chars",
]

# --------------------------------------------------------------------------
# Category names
# --------------------------------------------------------------------------

LETTER = "letter"
SUPERSCRIPT_LETTER = "superscript_letter"
HARAKA = "haraka"
HAMZA_MARK = "hamza_mark"
TAJWEED_MARK = "tajweed_mark"
WAQF_MARK = "waqf_mark"
ORNAMENT = "ornament"
FORMATTING = "formatting"
DIGIT = "digit"
PUNCT = "punct"
SPACE = "space"

# --------------------------------------------------------------------------
# Frequently referenced single codepoints
# --------------------------------------------------------------------------

SUKUN = "ْ"  # ARABIC SUKUN
QURANIC_SUKUN = "ۡ"  # ARABIC SMALL HIGH DOTLESS HEAD OF KHAH
SHADDA = "ّ"  # ARABIC SHADDA
TATWEEL = "ـ"  # ARABIC TATWEEL
SUPERSCRIPT_ALEF = "ٰ"  # ARABIC LETTER SUPERSCRIPT ALEF
ALEF = "ا"  # ARABIC LETTER ALEF
ALEF_WASLA = "ٱ"  # ARABIC LETTER ALEF WASLA
TEH_MARBUTA = "ة"  # ARABIC LETTER TEH MARBUTA
HEH = "ه"  # ARABIC LETTER HEH
ALEF_MAKSURA = "ى"  # ARABIC LETTER ALEF MAKSURA
YEH = "ي"  # ARABIC LETTER YEH

# --------------------------------------------------------------------------
# LETTER -- base consonants and long-vowel carriers
# --------------------------------------------------------------------------

LETTERS: Set[str] = {
    "ء",  # HAMZA
    "آ",  # ALEF WITH MADDA ABOVE
    "أ",  # ALEF WITH HAMZA ABOVE
    "ؤ",  # WAW WITH HAMZA ABOVE
    "إ",  # ALEF WITH HAMZA BELOW
    "ئ",  # YEH WITH HAMZA ABOVE
    "ا",  # ALEF
    "ب",  # BEH
    "ة",  # TEH MARBUTA
    "ت",  # TEH
    "ث",  # THEH
    "ج",  # JEEM
    "ح",  # HAH
    "خ",  # KHAH
    "د",  # DAL
    "ذ",  # THAL
    "ر",  # REH
    "ز",  # ZAIN
    "س",  # SEEN
    "ش",  # SHEEN
    "ص",  # SAD
    "ض",  # DAD
    "ط",  # TAH
    "ظ",  # ZAH
    "ع",  # AIN
    "غ",  # GHAIN
    "ف",  # FEH
    "ق",  # QAF
    "ك",  # KAF
    "ل",  # LAM
    "م",  # MEEM
    "ن",  # NOON
    "ه",  # HEH
    "و",  # WAW
    "ى",  # ALEF MAKSURA
    "ي",  # YEH
    "ٱ",  # ALEF WASLA  (hamzat al-wasl, very common in Warsh text)
    "ے",  # YEH BARREE  (appears in some Warsh typesettings, e.g. fi-y)
    "ۓ",  # YEH BARREE WITH HAMZA ABOVE
}

# --------------------------------------------------------------------------
# SUPERSCRIPT_LETTER -- pronounced, written as a combining mark
# --------------------------------------------------------------------------

SUPERSCRIPT_LETTERS: Set[str] = {
    "ٰ",  # SUPERSCRIPT ALEF
    "ٖ",  # SUBSCRIPT ALEF
    "ۥ",  # SMALL WAW
    "ۦ",  # SMALL YEH
}

# --------------------------------------------------------------------------
# HARAKA -- short vowels, tanween, shadda, sukun
# --------------------------------------------------------------------------

HARAKAT: Set[str] = {
    "ً",  # FATHATAN
    "ٌ",  # DAMMATAN
    "ٍ",  # KASRATAN
    "َ",  # FATHA
    "ُ",  # DAMMA
    "ِ",  # KASRA
    "ّ",  # SHADDA
    "ْ",  # SUKUN
    "ٗ",  # INVERTED DAMMA
    "٘",  # MARK NOON GHUNNA
    "ٙ",  # ZWARAKAY
    "ٚ",  # VOWEL SIGN SMALL V ABOVE
    "ٛ",  # VOWEL SIGN INVERTED SMALL V ABOVE
    "ٜ",  # VOWEL SIGN DOT BELOW
    "ٝ",  # REVERSED DAMMA
    "ٞ",  # FATHA WITH TWO DOTS
    "ٟ",  # WAVY HAMZA BELOW
    "ۡ",  # SMALL HIGH DOTLESS HEAD OF KHAH  (Quranic sukun)
}

# --------------------------------------------------------------------------
# HAMZA_MARK -- maddah and free-standing hamzas
# --------------------------------------------------------------------------

HAMZA_MARKS: Set[str] = {
    "ٓ",  # MADDAH ABOVE
    "ٔ",  # HAMZA ABOVE
    "ٕ",  # HAMZA BELOW
    "ۤ",  # SMALL HIGH MADDA
}

# --------------------------------------------------------------------------
# TAJWEED_MARK -- assimilation / nasalisation / silent-letter indicators
# --------------------------------------------------------------------------

TAJWEED_MARKS: Set[str] = {
    "۟",  # SMALL HIGH ROUNDED ZERO       (silent letter)
    "۠",  # SMALL HIGH UPRIGHT RECTANGULAR ZERO (silent in wasl only)
    "ۢ",  # SMALL HIGH MEEM ISOLATED FORM (iqlab)
    "ۣ",  # SMALL LOW SEEN                (sakta / seen under sad)
    "ۧ",  # SMALL HIGH YEH
    "ۨ",  # SMALL HIGH NOON
    "۪",  # EMPTY CENTRE LOW STOP
    "۫",  # EMPTY CENTRE HIGH STOP
    "۬",  # ROUNDED HIGH STOP WITH FILLED CENTRE
    "ۭ",  # SMALL LOW MEEM                (iqlab, low form)
    "ۜ",  # SMALL HIGH SEEN               (sakta indicator)
}

# --------------------------------------------------------------------------
# WAQF_MARK -- pause signs.  Editorial, never pronounced.
# --------------------------------------------------------------------------

WAQF_MARKS: Set[str] = {
    "ؕ",  # SMALL HIGH TAH
    "ؖ",  # SMALL HIGH LIGATURE ALEF WITH LAM WITH YEH
    "ؗ",  # SMALL HIGH ZAIN
    "ؘ",  # SMALL FATHA
    "ؙ",  # SMALL DAMMA
    "ؚ",  # SMALL KASRA
    "ۖ",  # SMALL HIGH LIGATURE SAD WITH LAM WITH ALEF MAKSURA
    "ۗ",  # SMALL HIGH LIGATURE QAF WITH LAM WITH ALEF MAKSURA
    "ۘ",  # SMALL HIGH MEEM INITIAL FORM
    "ۙ",  # SMALL HIGH LAM ALEF
    "ۚ",  # SMALL HIGH JEEM
    "ۛ",  # SMALL HIGH THREE DOTS
}

# --------------------------------------------------------------------------
# ORNAMENT -- structural glyphs
# --------------------------------------------------------------------------

ORNAMENTS: Set[str] = {
    "۝",  # END OF AYAH
    "۞",  # START OF RUB EL HIZB
    "۩",  # PLACE OF SAJDAH
    "࣢",  # DISPUTED END OF AYAH
}

# --------------------------------------------------------------------------
# FORMATTING -- invisible controls and the elongation dash
# --------------------------------------------------------------------------

FORMATTING_CHARS: Set[str] = {
    "ـ",  # TATWEEL
    "؜",  # ARABIC LETTER MARK
    "​",  # ZERO WIDTH SPACE
    "‌",  # ZERO WIDTH NON-JOINER
    "‍",  # ZERO WIDTH JOINER
    "‎",  # LEFT-TO-RIGHT MARK
    "‏",  # RIGHT-TO-LEFT MARK
    "‪",  # LEFT-TO-RIGHT EMBEDDING
    "‫",  # RIGHT-TO-LEFT EMBEDDING
    "‬",  # POP DIRECTIONAL FORMATTING
    "‭",  # LEFT-TO-RIGHT OVERRIDE
    "‮",  # RIGHT-TO-LEFT OVERRIDE
    "﻿",  # ZERO WIDTH NO-BREAK SPACE / BOM
}

DIGITS: Set[str] = {chr(c) for c in range(0x0660, 0x066A)} | {
    chr(c) for c in range(0x06F0, 0x06FA)
} | {chr(c) for c in range(0x0030, 0x003A)}

PUNCTUATION: Set[str] = {
    "،",  # ARABIC COMMA
    "؛",  # ARABIC SEMICOLON
    "؟",  # ARABIC QUESTION MARK
    "٪",  # ARABIC PERCENT SIGN
    "٫",  # ARABIC DECIMAL SEPARATOR
    "٬",  # ARABIC THOUSANDS SEPARATOR
    "۔",  # ARABIC FULL STOP
    ".",
    ",",
    ";",
    ":",
    "!",
    "?",
    "-",
    "(",
    ")",
    "[",
    "]",
    '"',
    "'",
}

# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

CATEGORIES: Dict[str, Set[str]] = {
    LETTER: LETTERS,
    SUPERSCRIPT_LETTER: SUPERSCRIPT_LETTERS,
    HARAKA: HARAKAT,
    HAMZA_MARK: HAMZA_MARKS,
    TAJWEED_MARK: TAJWEED_MARKS,
    WAQF_MARK: WAQF_MARKS,
    ORNAMENT: ORNAMENTS,
    FORMATTING: FORMATTING_CHARS,
    DIGIT: DIGITS,
    PUNCT: PUNCTUATION,
}

#: Reverse index, built once at import time.
CATEGORY_OF: Dict[str, str] = {
    char: name for name, chars in CATEGORIES.items() for char in chars
}

#: Every combining mark, regardless of sub-category.
ALL_MARKS: Set[str] = (
    SUPERSCRIPT_LETTERS | HARAKAT | HAMZA_MARKS | TAJWEED_MARKS | WAQF_MARKS
)


def category_of(char: str) -> str:
    """Return the category name for *char*, or ``"unknown"``.

    Whitespace maps to ``"space"``.
    """
    if not char:
        raise ValueError("category_of() expects a single character, got empty string")
    if len(char) != 1:
        raise ValueError(f"category_of() expects a single character, got {char!r}")
    if char.isspace():
        return SPACE
    return CATEGORY_OF.get(char, "unknown")


def is_letter(char: str) -> bool:
    """True if *char* is a base letter that can carry a sukun."""
    return char in LETTERS


def is_mark(char: str) -> bool:
    """True if *char* is any combining mark (haraka, tajweed, waqf, ...)."""
    return char in ALL_MARKS


def describe(char: str) -> str:
    """Human-readable ``U+XXXX <category> <unicode name>`` for diagnostics."""
    import unicodedata

    try:
        name = unicodedata.name(char)
    except ValueError:
        name = "<unnamed>"
    return f"U+{ord(char):04X} [{category_of(char)}] {name}"


def unknown_chars(text: Iterable[str]) -> Set[str]:
    """Return the set of characters in *text* that no category claims.

    Run this over a corpus before training: anything it returns is a codepoint
    the normaliser will pass through untouched and the tokenizer will have to
    deal with on its own.
    """
    return {c for c in text if category_of(c) == "unknown"}
