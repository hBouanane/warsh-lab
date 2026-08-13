"""Edit distance, alignment, and the error tallies built on them."""

import pytest

from warshlab.distance import (
    EQUAL,
    INSERT,
    SUB,
    ErrorCounts,
    align,
    edit_distance,
    error_counts,
    opcodes,
)


@pytest.mark.parametrize(
    "ref, hyp, expected",
    [
        ("", "", 0),
        ("abc", "abc", 0),
        ("abc", "", 3),
        ("", "abc", 3),
        ("kitten", "sitting", 3),
        ("flaw", "lawn", 2),
        ("a", "b", 1),
    ],
)
def test_edit_distance(ref, hyp, expected):
    assert edit_distance(ref, hyp) == expected


def test_edit_distance_is_symmetric():
    assert edit_distance("kitten", "sitting") == edit_distance("sitting", "kitten")


def test_edit_distance_works_on_word_lists():
    assert edit_distance(["قل", "هو", "الله"], ["قل", "هي", "الله"]) == 1


def test_align_reproduces_the_distance():
    script = align("kitten", "sitting")
    edits = sum(1 for op, _, _ in script if op != EQUAL)
    assert edits == edit_distance("kitten", "sitting")


def test_align_reconstructs_both_sequences():
    ref, hyp = "الرحمن", "الرحيم"
    script = align(ref, hyp)
    assert "".join(r for _, r, _ in script if r is not None) == ref
    assert "".join(h for _, _, h in script if h is not None) == hyp


def test_align_marks_each_operation_type():
    script = align("abc", "axcd")
    ops = {op for op, _, _ in script}
    assert SUB in ops and INSERT in ops and EQUAL in ops


def test_align_of_identical_sequences_is_all_equal():
    assert all(op == EQUAL for op, _, _ in align("abc", "abc"))


def test_align_is_deterministic():
    assert align("abcdef", "azcdxf") == align("abcdef", "azcdxf")


def test_opcodes_group_consecutive_operations():
    runs = opcodes(align("abcdef", "abXXef"))
    assert [op for op, _, _ in runs] == [EQUAL, SUB, EQUAL]
    assert runs[1][1] == ["c", "d"]
    assert runs[1][2] == ["X", "X"]


def test_error_counts_on_identical_input():
    counts = error_counts("abc", "abc")
    assert counts.errors == 0
    assert counts.hits == 3
    assert counts.rate == 0.0


def test_error_counts_classifies_each_error():
    counts = error_counts("abc", "axc")
    assert counts.substitutions == 1
    assert counts.hits == 2
    assert counts.insertions == counts.deletions == 0


def test_error_counts_counts_deletions_and_insertions():
    assert error_counts("abc", "ab").deletions == 1
    assert error_counts("ab", "abc").insertions == 1


def test_error_counts_ref_length_matches_the_reference():
    for ref, hyp in [("abcdef", "az"), ("abc", "abcdef"), ("", "abc")]:
        assert error_counts(ref, hyp).ref_length == len(ref)


def test_rate_against_an_empty_reference_does_not_divide_by_zero():
    assert error_counts("", "").rate == 0.0
    assert error_counts("", "abc").rate == 1.0


def test_rate_can_exceed_one_when_the_hypothesis_rambles():
    counts = error_counts("a", "abcdefgh")
    assert counts.rate > 1.0


def test_counts_add_up():
    total = error_counts("abc", "axc") + error_counts("de", "d")
    assert total.substitutions == 1
    assert total.deletions == 1
    assert total.ref_length == 5


def test_empty_counts_are_the_additive_identity():
    counts = error_counts("abc", "axc")
    assert (counts + ErrorCounts()).as_dict() == counts.as_dict()


def test_as_dict_is_json_ready():
    payload = error_counts("abc", "axc").as_dict()
    assert payload["errors"] == 1
    assert set(payload) == {
        "hits",
        "substitutions",
        "insertions",
        "deletions",
        "errors",
        "ref_length",
        "rate",
    }
