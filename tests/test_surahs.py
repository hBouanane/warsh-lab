"""Surah metadata and reference validation."""

import pytest

from warshlab import surahs


def test_there_are_114_surahs():
    assert len(surahs.SURAHS) == 114


def test_the_ayah_counts_sum_to_6236():
    """A well-known total -- if the shipped table drifts, this catches it."""
    assert surahs.total_ayahs() == 6236


def test_numbers_are_contiguous():
    assert [s.number for s in surahs.SURAHS] == list(range(1, 115))


def test_get_returns_the_expected_surah():
    fatiha = surahs.get(1)
    assert fatiha.english_name == "Al-Faatiha"
    assert fatiha.ayah_count == 7


def test_get_rejects_an_out_of_range_number():
    with pytest.raises(KeyError, match="1-114"):
        surahs.get(115)


def test_find_by_number_string():
    assert surahs.find("112").number == 112


def test_find_by_english_name_is_case_and_dash_insensitive():
    assert surahs.find("al faatiha").number == 1
    assert surahs.find("AL-FAATIHA").number == 1


def test_find_by_arabic_name():
    assert surahs.find(surahs.get(114).name).number == 114


def test_find_returns_none_for_nonsense():
    assert surahs.find("not a surah") is None
    assert surahs.find("") is None


def test_ayah_count_shortcut():
    assert surahs.ayah_count(114) == 6


def test_validate_reference_accepts_a_valid_pair():
    assert surahs.validate_reference(1, 7) is None


def test_validate_reference_accepts_a_surah_with_no_ayah():
    assert surahs.validate_reference(1) is None


def test_validate_reference_rejects_an_out_of_range_surah():
    assert "out of range" in surahs.validate_reference(115)


def test_validate_reference_rejects_an_out_of_range_ayah():
    message = surahs.validate_reference(1, 8)
    assert "Al-Faatiha has 7 ayat" in message


def test_validate_reference_rejects_ayah_zero():
    assert surahs.validate_reference(1, 0) is not None


def test_validate_reference_rejects_non_numeric_input():
    assert "not an integer" in surahs.validate_reference("first")
    assert "not an integer" in surahs.validate_reference(1, "seven")


def test_validate_reference_accepts_numeric_strings():
    assert surahs.validate_reference("1", "7") is None
