"""End-to-end CLI behaviour, including the demo pipeline."""

import json

import pytest

from warshlab import demo, manifest as MF
from warshlab.cli import main


@pytest.fixture()
def corpus(tmp_path):
    records = demo.build_manifest()
    path = tmp_path / "manifest.jsonl"
    MF.write(path, records)
    return path


@pytest.fixture()
def predictions(tmp_path, corpus):
    path = tmp_path / "predictions.jsonl"
    MF.write(path, demo.fake_predictions(MF.read(corpus)))
    return path


# --------------------------------------------------------------------------
# Basics
# --------------------------------------------------------------------------


def test_no_arguments_prints_help(capsys):
    assert main([]) == 2
    assert "warsh" in capsys.readouterr().out


def test_version_exits_zero():
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0


def test_doctor_passes(capsys):
    assert main(["doctor"]) == 0
    assert "All checks passed" in capsys.readouterr().out


def test_unknown_command_is_a_usage_error():
    with pytest.raises(SystemExit) as exc:
        main(["frobnicate"])
    assert exc.value.code == 2


# --------------------------------------------------------------------------
# normalize
# --------------------------------------------------------------------------


def test_normalize_text_argument(capsys):
    assert main(["normalize", "الرَّحِيمِ"]) == 0
    assert capsys.readouterr().out.strip() == "الرَّحِيمْ"


def test_normalize_rasm_form(capsys):
    assert main(["normalize", "--form", "rasm", "بِسْمِ اللَّهِ"]) == 0
    assert capsys.readouterr().out.strip() == "بسم الله"


def test_normalize_notebook_compat_differs(capsys):
    main(["normalize", "غَيْرِكُمُۥ"])
    default = capsys.readouterr().out.strip()
    main(["normalize", "--notebook-compat", "غَيْرِكُمُۥ"])
    compat = capsys.readouterr().out.strip()
    assert default != compat


def test_normalize_a_manifest_column(tmp_path, corpus):
    output = tmp_path / "normalized.jsonl"
    assert main(["normalize", "--manifest", str(corpus), "--output", str(output)]) == 0
    records = MF.read(output)
    assert all(r["text_normalized"].endswith("ْ") for r in records)


# --------------------------------------------------------------------------
# check
# --------------------------------------------------------------------------


def test_check_a_clean_manifest(capsys, corpus):
    assert main(["check", str(corpus)]) == 0
    assert "STATS" in capsys.readouterr().out


def test_check_json_output(capsys, corpus):
    assert main(["check", str(corpus), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True


def test_check_fails_on_a_broken_manifest(tmp_path, capsys):
    path = tmp_path / "broken.jsonl"
    MF.write(path, [{"segment_id": "x", "text_warsh": ""}])
    assert main(["check", str(path)]) == 1


def test_check_charset_listing(capsys, corpus):
    assert main(["check", str(corpus), "--charset"]) == 0
    assert "CHARSET" in capsys.readouterr().out


def test_check_reports_a_missing_file():
    assert main(["check", "does-not-exist.jsonl"]) == 1


# --------------------------------------------------------------------------
# split
# --------------------------------------------------------------------------


def test_split_writes_every_bucket(tmp_path, corpus):
    out_dir = tmp_path / "splits"
    assert main(["split", str(corpus), "--out-dir", str(out_dir)]) == 0
    for name in ("train", "val", "test"):
        assert (out_dir / f"{name}.jsonl").exists()
    assert (out_dir / "split_report.json").exists()


def test_split_preserves_the_corpus(tmp_path, corpus):
    out_dir = tmp_path / "splits"
    main(["split", str(corpus), "--out-dir", str(out_dir)])
    total = sum(
        len(MF.read(out_dir / f"{name}.jsonl")) for name in ("train", "val", "test")
    )
    assert total == len(MF.read(corpus))


def test_split_holdout_mode(tmp_path, corpus, capsys):
    out_dir = tmp_path / "holdout"
    assert main(["split", str(corpus), "--n-holdout", "1", "--out-dir", str(out_dir)]) == 0
    assert (out_dir / "unseen.jsonl").exists()
    assert "holdout" in capsys.readouterr().out


def test_split_custom_ratios(tmp_path, corpus):
    out_dir = tmp_path / "two"
    main(
        [
            "split", str(corpus),
            "--ratios", "0.9,0.1",
            "--names", "train,val",
            "--out-dir", str(out_dir),
        ]
    )
    assert (out_dir / "val.jsonl").exists()
    assert not (out_dir / "test.jsonl").exists()


# --------------------------------------------------------------------------
# eval / report
# --------------------------------------------------------------------------


def test_eval_prints_corpus_rates(capsys, corpus, predictions):
    assert main(["eval", str(corpus), "--predictions", str(predictions)]) == 0
    out = capsys.readouterr().out
    assert "corpus rates" in out
    assert "rasm_cer" in out


def test_eval_writes_json_and_html(tmp_path, corpus, predictions):
    json_out = tmp_path / "results.json"
    html_out = tmp_path / "report.html"
    assert (
        main(
            [
                "eval", str(corpus),
                "--predictions", str(predictions),
                "--json-out", str(json_out),
                "--html-out", str(html_out),
            ]
        )
        == 0
    )
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["summary"]["n"] > 0
    assert html_out.read_text(encoding="utf-8").startswith("<!doctype html>")


def test_eval_fail_over_threshold_gates_the_exit_code(tmp_path, corpus, predictions):
    args = ["eval", str(corpus), "--predictions", str(predictions)]
    assert main(args + ["--fail-over", "0.99"]) == 0
    assert main(args + ["--fail-over", "0.0"]) == 1


def test_eval_errors_when_nothing_joins(tmp_path, corpus):
    orphans = tmp_path / "orphans.jsonl"
    MF.write(orphans, [{"segment_id": "ghost", "prediction": "x"}])
    assert main(["eval", str(corpus), "--predictions", str(orphans)]) == 1


def test_eval_reads_both_columns_from_one_file(tmp_path):
    path = tmp_path / "pairs.jsonl"
    MF.write(
        path,
        [{"segment_id": "a", "text_warsh": "قُلْ هُوَ", "prediction": "قُلْ هُوَ"}],
    )
    assert main(["eval", str(path)]) == 0


def test_report_renders_from_a_results_file(tmp_path, corpus, predictions):
    json_out = tmp_path / "results.json"
    main(
        ["eval", str(corpus), "--predictions", str(predictions), "--json-out", str(json_out)]
    )
    html_out = tmp_path / "from_json.html"
    assert main(["report", str(json_out), "-o", str(html_out)]) == 0
    assert "Warsh ASR evaluation" in html_out.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# surah
# --------------------------------------------------------------------------


def test_surah_lookup(capsys):
    assert main(["surah", "112"]) == 0
    assert "Al-Ikhlaas" in capsys.readouterr().out or "112" in capsys.readouterr().out


def test_surah_listing(capsys):
    assert main(["surah"]) == 0
    assert "114 surahs" in capsys.readouterr().out


def test_surah_unknown_query():
    assert main(["surah", "nonsense-name"]) == 1


# --------------------------------------------------------------------------
# demo
# --------------------------------------------------------------------------


def test_demo_produces_every_artefact(tmp_path, capsys):
    out_dir = tmp_path / "demo"
    assert main(["demo", "--out-dir", str(out_dir)]) == 0

    for name in (
        "manifest.jsonl",
        "predictions.jsonl",
        "results.json",
        "report.html",
        "train.jsonl",
        "val.jsonl",
        "test.jsonl",
        "holdout_unseen.jsonl",
    ):
        assert (out_dir / name).exists(), f"{name} was not written"

    assert "Done." in capsys.readouterr().out


def test_demo_report_is_self_contained(tmp_path):
    out_dir = tmp_path / "demo"
    main(["demo", "--out-dir", str(out_dir)])
    html = (out_dir / "report.html").read_text(encoding="utf-8")
    assert "http://" not in html
    assert "https://" not in html
    assert "<script" not in html


def test_demo_is_reproducible(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    main(["demo", "--out-dir", str(first)])
    main(["demo", "--out-dir", str(second)])
    assert (first / "predictions.jsonl").read_text(encoding="utf-8") == (
        second / "predictions.jsonl"
    ).read_text(encoding="utf-8")
