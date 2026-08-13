"""The category tables must stay disjoint -- that is the module's whole promise."""

import pytest

from warshlab import chars as C


def test_categories_are_disjoint():
    """No codepoint may belong to two categories.

    This is the bug the module exists to prevent: when a character sits in both
    a "letters" set and a "strip these" set, behaviour depends on which set the
    normaliser happens to consult first.
    """
    seen = {}
    overlaps = []
    for name, members in C.CATEGORIES.items():
        for char in members:
            if char in seen:
                overlaps.append(f"U+{ord(char):04X} in {seen[char]} and {name}")
            seen[char] = name
    assert overlaps == []


def test_every_category_is_populated():
    for name, members in C.CATEGORIES.items():
        assert members, f"category {name} is empty"


def test_category_of_known_characters():
    assert C.category_of("ب") == C.LETTER
    assert C.category_of("ٱ") == C.LETTER
    assert C.category_of("َ") == C.HARAKA
    assert C.category_of("ْ") == C.HARAKA
    assert C.category_of("ٰ") == C.SUPERSCRIPT_LETTER
    assert C.category_of("ۖ") == C.WAQF_MARK
    assert C.category_of("۩") == C.ORNAMENT
    assert C.category_of("ـ") == C.FORMATTING
    assert C.category_of(" ") == C.SPACE
    assert C.category_of("Z") == "unknown"


def test_small_high_yeh_is_a_tajweed_mark_not_a_letter():
    """U+06E7 was in both sets in the original notebook; here it is a mark."""
    assert C.category_of("ۧ") == C.TAJWEED_MARK
    assert not C.is_letter("ۧ")
    assert C.is_mark("ۧ")


def test_small_waw_and_yeh_are_superscript_letters():
    for char in ("ۥ", "ۦ"):
        assert C.category_of(char) == C.SUPERSCRIPT_LETTER
        assert not C.is_letter(char)
        assert C.is_mark(char)


def test_all_marks_excludes_base_letters():
    assert not (C.ALL_MARKS & C.LETTERS)


def test_describe_includes_codepoint_and_category():
    described = C.describe("ب")
    assert "U+0628" in described
    assert C.LETTER in described


def test_describe_handles_unnamed_codepoint():
    assert "U+0378" in C.describe("͸")


@pytest.mark.parametrize("bad", ["", "abc"])
def test_category_of_rejects_non_single_characters(bad):
    with pytest.raises(ValueError):
        C.category_of(bad)


def test_unknown_chars_finds_foreign_codepoints():
    found = C.unknown_chars("بِسْمِ Z اللَّهِ")
    assert found == {"Z"}


def test_unknown_chars_empty_for_pure_quranic_text():
    assert C.unknown_chars("بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ") == set()
