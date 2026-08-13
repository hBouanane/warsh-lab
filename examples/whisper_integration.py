"""Dropping warsh-lab into the Whisper fine-tuning notebook.

Each section below replaces a block of ``whisper_warsh_training.ipynb`` with the
tested equivalent.  Nothing here imports torch or transformers, so the file is
readable (and the snippets are copy-pasteable) without a GPU box.

Run it as a script to see the label-normalisation and scoring pieces work on the
built-in sample corpus:

    python examples/whisper_integration.py
"""

from __future__ import annotations

import json
from pathlib import Path

from warshlab import demo, manifest, metrics, report, splits, text


# ---------------------------------------------------------------------------
# 1. Labels  --  replaces the notebook's `apply_waqf` cell
# ---------------------------------------------------------------------------
#
# The notebook kept a LETTERS set and a REMOVE set that overlapped on U+06E7 and
# U+06E8, so whether those marks counted as sukun carriers depended on the order
# the sets were applied in.  `warshlab.chars` assigns every codepoint exactly one
# category and a test asserts the categories stay disjoint.
#
#     def prepare(batch):
#         batch["labels"] = tokenizer(text.apply_waqf(batch["text_warsh"])).input_ids
#         return batch
#
# To keep labels bit-identical to a previous run, pass the compatibility config:
#
#     text.apply_waqf(raw, text.NOTEBOOK_WAQF)


def show_label_forms(raw: str) -> None:
    print(f"  raw       {text.to_raw(raw)}")
    print(f"  clean     {text.to_clean(raw)}")
    print(f"  waqf      {text.apply_waqf(raw)}   <- train on this")
    print(f"  rasm      {text.to_rasm(raw)}")
    print(f"  harakat   {text.to_harakat(raw)}")


# ---------------------------------------------------------------------------
# 2. Splits  --  replaces the notebook's manual stratification cell
# ---------------------------------------------------------------------------
#
# The notebook asserted `val_reciters == train_reciters`, which crashes the run
# whenever a reciter has too few segments to appear in every split.  Here that
# case is a warning on the result object, and the split still happens.
#
# The second call is the one the notebook never made.  A stratified split shows
# every reciter to the model during training, so it cannot tell you what happens
# on a voice the model has never heard -- which is the only number that predicts
# what a user will experience.


def build_splits(records):
    seen_voices = splits.stratified_split(records, group_key="reciter_slug", seed=42)
    unseen_voice = splits.holdout_split(records, group_key="reciter_slug", n_holdout=1)

    leaks = splits.check_leakage(seen_voices.splits)
    assert not leaks, leaks

    return seen_voices, unseen_voice


# ---------------------------------------------------------------------------
# 3. Metrics  --  replaces `compute_metrics` in the Seq2SeqTrainer
# ---------------------------------------------------------------------------
#
#     from warshlab import metrics as M
#
#     def compute_metrics(pred):
#         label_ids = pred.label_ids.copy()
#         label_ids[label_ids == -100] = tokenizer.pad_token_id
#         hyps = tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
#         refs = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
#
#         ev = M.evaluate(
#             {"segment_id": str(i), "reference": r, "hypothesis": h}
#             for i, (r, h) in enumerate(zip(refs, hyps))
#         )
#         # rasm_cer and harakat_cer split the error into "misheard the
#         # consonants" and "guessed the vowels" -- they move independently, and
#         # a single CER curve cannot tell you which one is improving.
#         return {name: round(100 * rate, 3) for name, rate in ev.corpus.items()}
#
# No jiwer or evaluate import, so nothing downloads a metric script mid-training.


# ---------------------------------------------------------------------------
# 4. Test-set report  --  replaces the notebook's final evaluation cell
# ---------------------------------------------------------------------------
#
#     rows = [
#         {"segment_id": ex["segment_id"], "prediction": decode(ex)}
#         for ex in test_ds
#     ]
#     manifest.write("output/predictions.jsonl", rows)
#
# then, from a shell:
#
#     warsh eval test.jsonl --predictions output/predictions.jsonl \
#         --json-out output/results.json --html-out output/report.html
#
# `--fail-over 0.15` makes the command exit non-zero when corpus CER regresses,
# so a training run can gate itself in CI.


def score_and_report(records, predictions, out_dir: Path):
    pairs = manifest.join_predictions(records, predictions)
    evaluation = metrics.evaluate(pairs, group_by=("reciter_slug", "surah_number"))

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps(evaluation.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return evaluation, report.write_report(
        evaluation, out_dir / "report.html", title="Warsh ASR -- integration example"
    )


# ---------------------------------------------------------------------------


def main() -> None:
    records = demo.build_manifest()

    print("1. label forms")
    show_label_forms(records[0]["text_warsh"])

    print("\n2. splits")
    seen_voices, unseen_voice = build_splits(records)
    print("  " + seen_voices.summary().replace("\n", "\n  "))
    print("  " + unseen_voice.summary().replace("\n", "\n  "))

    print("\n3. scoring")
    predictions = demo.fake_predictions(records)
    evaluation, path = score_and_report(records, predictions, Path("integration_out"))
    for name, rate in evaluation.corpus.items():
        print(f"  {name:<14} {100 * rate:6.2f}%")

    print(f"\n4. report -> {path}")


if __name__ == "__main__":
    main()
