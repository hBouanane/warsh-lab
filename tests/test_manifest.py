"""Manifest I/O round-trips and the pre-training audit."""

import json

import pytest

from warshlab import manifest as MF

RECORDS = [
    {
        "segment_id": "alpha-001",
        "reciter_slug": "alpha",
        "surah_number": 1,
        "ayah_number": 1,
        "text_warsh": "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ",
    },
    {
        "segment_id": "alpha-002",
        "reciter_slug": "alpha",
        "surah_number": 112,
        "ayah_number": 1,
        "text_warsh": "قُلْ هُوَ اللَّهُ أَحَدٌ",
    },
    {
        "segment_id": "beta-001",
        "reciter_slug": "beta",
        "surah_number": 114,
        "ayah_number": 6,
        "text_warsh": "مِنَ الْجِنَّةِ وَالنَّاسِ",
    },
]


# --------------------------------------------------------------------------
# I/O
# --------------------------------------------------------------------------


@pytest.mark.parametrize("suffix", [".jsonl", ".json", ".csv", ".tsv"])
def test_round_trip(tmp_path, suffix):
    path = tmp_path / f"manifest{suffix}"
    assert MF.write(path, RECORDS) == 3
    loaded = MF.read(path)
    assert len(loaded) == 3
    assert loaded[0]["segment_id"] == "alpha-001"
    assert loaded[0]["text_warsh"] == RECORDS[0]["text_warsh"]


def test_jsonl_preserves_arabic_unescaped(tmp_path):
    path = tmp_path / "m.jsonl"
    MF.write(path, RECORDS)
    assert "بِسْمِ" in path.read_text(encoding="utf-8")


def test_read_skips_blank_lines(tmp_path):
    path = tmp_path / "m.jsonl"
    path.write_text(
        json.dumps(RECORDS[0], ensure_ascii=False) + "\n\n\n", encoding="utf-8"
    )
    assert len(MF.read(path)) == 1


def test_malformed_jsonl_reports_its_line_number(tmp_path):
    path = tmp_path / "m.jsonl"
    path.write_text('{"a": 1}\nnot json\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r":2:"):
        MF.read(path)


def test_json_object_with_a_records_array(tmp_path):
    path = tmp_path / "m.json"
    path.write_text(
        json.dumps({"records": RECORDS}, ensure_ascii=False), encoding="utf-8"
    )
    assert len(MF.read(path)) == 3


def test_unknown_extension_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="extension"):
        MF.read(tmp_path / "m.parquet")


def test_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        MF.read(tmp_path / "nope.jsonl")


def test_write_creates_parent_directories(tmp_path):
    path = tmp_path / "deep" / "nested" / "m.jsonl"
    MF.write(path, RECORDS)
    assert path.exists()


def test_csv_columns_cover_every_key(tmp_path):
    path = tmp_path / "m.csv"
    MF.write(path, [{"a": 1}, {"a": 2, "b": 3}])
    header = path.read_text(encoding="utf-8").splitlines()[0]
    assert "a" in header and "b" in header


# --------------------------------------------------------------------------
# check
# --------------------------------------------------------------------------


def test_a_clean_manifest_passes():
    report = MF.check(RECORDS)
    assert report.ok
    assert report.stats["records"] == 3
    assert report.stats["groups"] == 2


def test_an_empty_manifest_is_an_error():
    report = MF.check([])
    assert not report.ok
    assert report.errors[0].code == "empty-manifest"


def test_duplicate_ids_are_errors():
    report = MF.check(RECORDS + [RECORDS[0]])
    assert any(i.code == "duplicate-id" for i in report.errors)


def test_empty_text_is_an_error():
    report = MF.check([{"segment_id": "x", "text_warsh": "   "}])
    assert any(i.code == "empty-text" for i in report.errors)


def test_text_without_letters_is_an_error():
    report = MF.check([{"segment_id": "x", "text_warsh": "12345"}])
    assert any(i.code in ("no-letters", "empty-text") for i in report.errors)


def test_an_out_of_range_surah_is_an_error():
    report = MF.check(
        [{"segment_id": "x", "text_warsh": "قُلْ", "surah_number": 115}]
    )
    assert any(i.code == "bad-reference" for i in report.errors)


def test_an_out_of_range_ayah_is_an_error():
    """Al-Fatiha has 7 ayat, so ayah 8 is a segmentation bug."""
    report = MF.check(
        [
            {
                "segment_id": "x",
                "text_warsh": "قُلْ",
                "surah_number": 1,
                "ayah_number": 8,
            }
        ]
    )
    assert any("out of range" in i.message for i in report.errors)


def test_missing_id_field_is_an_error():
    report = MF.check([{"text_warsh": "قُلْ"}])
    assert any(i.code == "missing-field" for i in report.errors)


def test_unknown_codepoints_are_surfaced_as_warnings():
    report = MF.check([{"segment_id": "x", "text_warsh": "قُلْ hello"}])
    assert any(i.code == "unknown-chars" for i in report.warnings)
    assert report.unknown_chars


def test_charset_is_grouped_by_category():
    report = MF.check(RECORDS)
    assert "letter" in report.charset
    assert "haraka" in report.charset


def test_a_tiny_group_is_warned_about():
    report = MF.check(RECORDS)
    assert any(i.code == "tiny-group" for i in report.warnings)


def test_a_manifest_without_a_group_key_is_warned_about():
    report = MF.check([{"segment_id": "x", "text_warsh": "قُلْ هُوَ"}])
    assert any(i.code == "no-group-key" for i in report.warnings)


def test_non_pausal_labels_are_flagged():
    report = MF.check(RECORDS)
    assert any(i.code == "labels-not-pausal" for i in report.issues)


def test_pausal_labels_are_not_flagged():
    pausal = [{**r, "text_warsh": r["text_warsh"][:-1] + "ْ"} for r in RECORDS]
    report = MF.check(pausal)
    assert not any(i.code == "labels-not-pausal" for i in report.issues)


def test_report_serialises_to_json():
    payload = json.loads(json.dumps(MF.check(RECORDS).as_dict(), ensure_ascii=False))
    assert payload["n_records"] == 3


def test_summary_is_printable():
    assert "record(s)" in MF.check(RECORDS).summary()


# --------------------------------------------------------------------------
# join_predictions
# --------------------------------------------------------------------------


def test_join_predictions_pairs_on_segment_id():
    predictions = [{"segment_id": "alpha-001", "prediction": "بسم الله"}]
    joined = MF.join_predictions(RECORDS, predictions)
    assert len(joined) == 1
    assert joined[0]["reference"] == RECORDS[0]["text_warsh"]
    assert joined[0]["hypothesis"] == "بسم الله"


def test_join_predictions_carries_metadata_through():
    joined = MF.join_predictions(
        RECORDS, [{"segment_id": "beta-001", "prediction": "x"}]
    )
    assert joined[0]["reciter_slug"] == "beta"
    assert joined[0]["surah_number"] == 114


def test_join_predictions_drops_unmatched_predictions():
    joined = MF.join_predictions(RECORDS, [{"segment_id": "ghost", "prediction": "x"}])
    assert joined == []
