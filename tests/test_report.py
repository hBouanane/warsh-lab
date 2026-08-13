"""The HTML report must be valid, self-contained, and safe to open."""

import pytest

from warshlab import demo, manifest as MF, metrics as M, report as R

REF = "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمْ"


@pytest.fixture()
def evaluation():
    records = demo.build_manifest()
    predictions = demo.fake_predictions(records)
    return M.evaluate(MF.join_predictions(records, predictions))


def test_render_produces_a_complete_document(evaluation):
    html = R.render(evaluation)
    assert html.startswith("<!doctype html>")
    assert html.rstrip().endswith("</html>")
    assert html.count("<body>") == 1


def test_report_has_no_external_references(evaluation):
    """It has to open from a training box with no network."""
    html = R.render(evaluation)
    assert "http://" not in html
    assert "https://" not in html
    assert "<script" not in html
    assert "@import" not in html


def test_report_shows_every_headline_metric(evaluation):
    html = R.render(evaluation)
    for label in ("CER", "WER", "Rasm CER", "Harakat CER"):
        assert label in html


def test_report_includes_the_per_reciter_table(evaluation):
    html = R.render(evaluation)
    assert "By reciter slug" in html
    for slug, _ in demo.RECITERS:
        assert slug in html


def test_report_includes_a_histogram(evaluation):
    assert "<svg" in R.render(evaluation)


def test_report_marks_diffs(evaluation):
    html = R.render(evaluation)
    assert 'class="sub"' in html or 'class="del"' in html or 'class="ins"' in html


def test_worst_n_limits_the_sample_blocks(evaluation):
    assert R.render(evaluation, worst_n=3).count('<div class="pair">') == 3


def test_html_is_escaped_so_a_prediction_cannot_inject_markup():
    ev = M.evaluate(
        [
            {
                "segment_id": "<img src=x onerror=alert(1)>",
                "reference": REF,
                "hypothesis": "<script>alert('xss')</script>",
            }
        ]
    )
    html = R.render(ev)
    # The diff view interleaves <mark> tags between characters, so the escaped
    # payload is not contiguous -- what matters is that no tag survives intact.
    assert "<script" not in html
    assert "<img" not in html
    assert "&lt;" in html and "&gt;" in html


def test_report_renders_for_a_single_perfect_sample():
    ev = M.evaluate([{"segment_id": "a", "reference": REF, "hypothesis": REF}])
    html = R.render(ev)
    assert "0.00%" in html


def test_report_renders_for_an_empty_evaluation():
    html = R.render(M.evaluate([]))
    assert "<!doctype html>" in html


def test_skipped_records_are_listed():
    ev = M.evaluate(
        [
            {"segment_id": "ok", "reference": REF, "hypothesis": REF},
            {"segment_id": "broken", "reference": "", "hypothesis": REF},
        ]
    )
    html = R.render(ev)
    assert "Skipped (1)" in html
    assert "broken" in html


def test_diagnosis_names_the_dominant_error_mode():
    """Diacritics-only errors should be called out as such."""
    from warshlab import text as T

    ev = M.evaluate(
        [{"segment_id": "a", "reference": REF, "hypothesis": T.to_rasm(REF)}]
    )
    assert "diacritisation" in R.render(ev)


def test_write_report_creates_the_file(tmp_path, evaluation):
    path = R.write_report(evaluation, tmp_path / "nested" / "report.html")
    assert path.exists()
    assert path.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_title_and_subtitle_appear(evaluation):
    html = R.render(evaluation, title="Run 42", subtitle="checkpoint-8000")
    assert "Run 42" in html
    assert "checkpoint-8000" in html


def test_generated_at_is_injectable_for_reproducible_output(evaluation):
    first = R.render(evaluation, generated_at="2026-01-01 00:00")
    second = R.render(evaluation, generated_at="2026-01-01 00:00")
    assert first == second
