"""Scoring: the metric spread, corpus aggregation, grouping, and confusions."""

import pytest

from warshlab import metrics as M
from warshlab import text as T

REF = "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمْ"


def _pairs(*triples):
    return [
        {"segment_id": sid, "reference": ref, "hypothesis": hyp, "reciter_slug": who}
        for sid, ref, hyp, who in triples
    ]


# --------------------------------------------------------------------------
# One-shot scorers
# --------------------------------------------------------------------------


def test_cer_of_identical_text_is_zero():
    assert M.cer(REF, REF) == 0.0


def test_wer_of_identical_text_is_zero():
    assert M.wer(REF, REF) == 0.0


def test_cer_is_positive_when_text_differs():
    assert M.cer("الرَّحِيمْ", "الرَّحْمَٰنْ") > 0


def test_cer_ignores_pause_marks():
    """A waqf sign in the reference must not count as a model error."""
    assert M.cer("قُلْۖ", "قُلْ") == 0.0


def test_stripping_diacritics_hurts_cer_but_not_rasm_cer():
    hypothesis = T.to_rasm(REF)
    assert M.cer(REF, hypothesis) > 0.3
    assert M.cer(REF, hypothesis, form="rasm") == 0.0


def test_orthographic_variants_survive_the_skeleton_form():
    assert M.cer("مُصِيبَةْ", "مُصِيبَهْ", form="rasm") > 0
    assert M.cer("مُصِيبَةْ", "مُصِيبَهْ", form="skeleton") == 0.0


def test_wer_counts_whole_words():
    assert M.wer("قُلْ هُوَ اللَّهْ", "قُلْ هِيَ اللَّهْ") == pytest.approx(1 / 3)


# --------------------------------------------------------------------------
# score_pair
# --------------------------------------------------------------------------


def test_score_pair_produces_every_registered_metric():
    scores = M.score_pair(REF, REF)
    assert set(scores) == {spec.name for spec in M.METRICS}


def test_score_pair_separates_consonant_and_vowel_errors():
    """Dropping the diacritics should show up in harakat_cer, not rasm_cer."""
    scores = M.score_pair(REF, T.to_rasm(REF))
    assert scores["rasm_cer"].rate == 0.0
    assert scores["harakat_cer"].rate == 1.0


# --------------------------------------------------------------------------
# evaluate
# --------------------------------------------------------------------------


def test_evaluate_scores_every_sample():
    ev = M.evaluate(_pairs(("a", REF, REF, "x"), ("b", REF, "قُلْ", "y")))
    assert len(ev.samples) == 2
    assert ev.samples[0].exact


def test_evaluate_skips_records_with_no_reference():
    ev = M.evaluate(
        [
            {"segment_id": "a", "reference": REF, "hypothesis": REF},
            {"segment_id": "b", "reference": "", "hypothesis": REF},
            {"segment_id": "c", "reference": REF},
        ]
    )
    assert len(ev.samples) == 1
    assert {r["segment_id"] for r in ev.skipped} == {"b", "c"}


def test_evaluate_of_perfect_predictions_is_zero_everywhere():
    ev = M.evaluate(_pairs(("a", REF, REF, "x"), ("b", "قُلْ هُوَ", "قُلْ هُوَ", "y")))
    assert all(rate == 0.0 for rate in ev.corpus.values())
    assert ev.buckets["exact_match"] == 1.0


def test_corpus_rate_pools_errors_rather_than_averaging_rates():
    """A long good segment must outweigh a short bad one in the corpus number."""
    long_ref = "الرَّحْمَٰنِ الرَّحِيمِ الْعَالَمِينْ"
    ev = M.evaluate(
        _pairs(("long", long_ref, long_ref, "x"), ("short", "قُلْ", "هَلْ", "x"))
    )
    assert ev.corpus["cer"] < ev.mean["cer"]


def test_groups_are_built_for_each_requested_field():
    ev = M.evaluate(
        _pairs(("a", REF, REF, "alpha"), ("b", REF, "قُلْ", "beta")),
        group_by=("reciter_slug",),
    )
    keys = {g.key for g in ev.groups["reciter_slug"]}
    assert keys == {"alpha", "beta"}


def test_groups_are_sorted_best_first():
    ev = M.evaluate(
        _pairs(
            ("a", REF, REF, "clean"),
            ("b", REF, "قُلْ", "noisy"),
        ),
        group_by=("reciter_slug",),
    )
    ordered = [g.key for g in ev.groups["reciter_slug"]]
    assert ordered == ["clean", "noisy"]


def test_missing_group_values_are_left_out_rather_than_bucketed_as_none():
    ev = M.evaluate(
        [
            {"segment_id": "a", "reference": REF, "hypothesis": REF, "reciter_slug": "x"},
            {"segment_id": "b", "reference": REF, "hypothesis": REF},
        ],
        group_by=("reciter_slug",),
    )
    assert [g.key for g in ev.groups["reciter_slug"]] == ["x"]


def test_confusions_record_the_substituted_pair():
    ev = M.evaluate(_pairs(("a", "مُصِيبَةْ", "مُصِيبَهْ", "x")))
    assert ("ة", "ه", 1) in ev.confusions


def test_deletions_and_insertions_are_tracked_separately():
    ev = M.evaluate(_pairs(("a", "قُلْ", "قْ", "x"), ("b", "قْ", "قُلْ", "x")))
    assert dict(ev.deletions)
    assert dict(ev.insertions)


def test_buckets_report_the_score_distribution():
    ev = M.evaluate(
        _pairs(
            ("a", REF, REF, "x"),
            ("b", REF, REF, "x"),
            ("c", REF, "لَا", "x"),
        )
    )
    assert ev.buckets["exact_match"] == pytest.approx(2 / 3)
    assert ev.buckets["cer_gt_0.50"] == pytest.approx(1 / 3)


def test_worst_and_best_are_opposite_ends_of_the_same_ordering():
    ev = M.evaluate(_pairs(("good", REF, REF, "x"), ("bad", REF, "لَا", "x")))
    assert ev.worst(1)[0].segment_id == "bad"
    assert ev.best(1)[0].segment_id == "good"


def test_evaluate_of_an_empty_set_does_not_crash():
    ev = M.evaluate([])
    assert ev.samples == []
    assert ev.buckets["exact_match"] == 0.0


def test_as_dict_round_trips_through_json():
    import json

    ev = M.evaluate(_pairs(("a", REF, "قُلْ", "x")))
    payload = json.loads(json.dumps(ev.as_dict(), ensure_ascii=False))
    assert payload["summary"]["n"] == 1
    assert payload["samples"][0]["segment_id"] == "a"


def test_as_dict_can_omit_samples():
    ev = M.evaluate(_pairs(("a", REF, "قُلْ", "x")))
    assert "samples" not in ev.as_dict(include_samples=False)


def test_metadata_travels_with_the_sample():
    ev = M.evaluate(
        [
            {
                "segment_id": "a",
                "reference": REF,
                "hypothesis": REF,
                "surah_number": 1,
                "reciter_slug": "x",
            }
        ]
    )
    assert ev.samples[0].metadata["surah_number"] == 1
