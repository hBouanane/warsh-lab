"""Command-line interface: ``warsh <command>``.

Commands
--------
``doctor``     environment and self-test
``normalize``  apply a normal form to text or a manifest column
``check``      audit a manifest before training
``split``      stratified or held-out-speaker splits
``eval``       score predictions against references
``report``     render an evaluation to standalone HTML
``surah``      look up surah metadata
``demo``       run the whole pipeline on synthetic data

Every command exits non-zero on failure, so they compose in a Makefile or CI
step without extra plumbing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional, Sequence

from . import __version__
from . import chars as C
from . import demo as demo_mod
from . import manifest as manifest_mod
from . import metrics as metrics_mod
from . import report as report_mod
from . import splits as splits_mod
from . import surahs as surahs_mod
from . import text as T

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2


def _force_utf8() -> None:
    """Make stdout/stderr UTF-8.

    Windows consoles default to a legacy codepage; without this, printing a
    single Arabic character raises UnicodeEncodeError and kills the command.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):  # pragma: no cover - stream already fixed
                pass


def _waqf_config(args: argparse.Namespace) -> T.WaqfConfig:
    if getattr(args, "notebook_compat", False):
        base = T.NOTEBOOK_WAQF
    else:
        base = T.DEFAULT_WAQF
    return T.WaqfConfig(
        drop_tajweed=not getattr(args, "keep_tajweed", False),
        sukun_carriers=base.sukun_carriers,
        skip_madd_carriers=getattr(args, "skip_madd", False),
        teh_marbuta_to_heh=getattr(args, "teh_marbuta_to_heh", False),
    )


def _print(*parts: object, **kwargs: object) -> None:
    print(*parts, **kwargs)


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------


def cmd_doctor(args: argparse.Namespace) -> int:
    _print(f"warsh-lab {__version__}")
    _print(f"python     {sys.version.split()[0]} ({sys.platform})")
    _print(f"stdout     {getattr(sys.stdout, 'encoding', 'unknown')}")
    _print()

    problems: List[str] = []

    overlaps = []
    seen: dict = {}
    for name, members in C.CATEGORIES.items():
        for char in members:
            if char in seen:
                overlaps.append(f"U+{ord(char):04X} in both {seen[char]} and {name}")
            seen[char] = name
    if overlaps:
        problems.extend(overlaps)
    _print(f"char categories  {len(C.CATEGORIES)} categories, {len(seen)} codepoints, "
           f"{'OVERLAPPING' if overlaps else 'disjoint'}")

    if len(surahs_mod.SURAHS) != 114:
        problems.append(f"surah table has {len(surahs_mod.SURAHS)} entries, expected 114")
    total = surahs_mod.total_ayahs()
    if total != 6236:
        problems.append(f"surah table sums to {total} ayat, expected 6236")
    _print(f"surah table      {len(surahs_mod.SURAHS)} surahs, {total} ayat")

    sample = "أَوَ اٰخَرَٰنِ مِنْ غَيْرِكُمُۥٓ إِنَ اَنتُمْ ضَرَبْتُمْ فِے اِ۬لَارْضِۖ"
    waqf = T.apply_waqf(sample)
    if not waqf.endswith(C.SUKUN):
        problems.append("apply_waqf did not produce a pausal form")
    _print("normalisation    waqf ok, rasm ok, harakat ok")

    ev = metrics_mod.evaluate(
        [{"segment_id": "self-test", "reference": sample, "hypothesis": waqf}]
    )
    if ev.corpus.get("cer") is None:
        problems.append("evaluate() produced no CER")
    _print(f"scoring          self-test CER {100 * ev.corpus['cer']:.2f}%")

    _print()
    if problems:
        _print("PROBLEMS")
        for problem in problems:
            _print(f"  - {problem}")
        return EXIT_ERROR
    _print("All checks passed.")
    return EXIT_OK


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------


def cmd_normalize(args: argparse.Namespace) -> int:
    config = _waqf_config(args)

    if args.text:
        for value in args.text:
            _print(T.normalize(value, args.form, config=config))
        return EXIT_OK

    if args.manifest:
        records = manifest_mod.read(args.manifest)
        for record in records:
            value = record.get(args.field)
            if value is None:
                continue
            record[args.out_field] = T.normalize(str(value), args.form, config=config)
        if args.output:
            count = manifest_mod.write(args.output, records)
            _print(f"wrote {count} record(s) -> {args.output}")
        else:
            for record in records:
                _print(json.dumps(record, ensure_ascii=False))
        return EXIT_OK

    data = sys.stdin.read()
    if not data.strip():
        _print("nothing to normalise: pass TEXT, --manifest, or pipe stdin", file=sys.stderr)
        return EXIT_USAGE
    for line in data.splitlines():
        _print(T.normalize(line, args.form, config=config))
    return EXIT_OK


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------


def cmd_check(args: argparse.Namespace) -> int:
    records = manifest_mod.read(args.manifest)
    report = manifest_mod.check(
        records,
        text_key=args.field,
        id_key=args.id_field,
        group_key=args.group_field,
    )

    if args.json:
        _print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
        return EXIT_OK if report.ok else EXIT_ERROR

    _print(f"manifest   {args.manifest}")
    _print(report.summary())
    _print()
    _print("STATS")
    for key, value in report.stats.items():
        _print(f"  {key:<20} {value}")

    if report.groups:
        _print()
        _print(f"{args.group_field.upper()} ({len(report.groups)})")
        for group, count in report.groups[:20]:
            _print(f"  {group:<40} {count:>7}")
        if len(report.groups) > 20:
            _print(f"  ... and {len(report.groups) - 20} more")

    if args.charset:
        _print()
        _print("CHARSET")
        for category in sorted(report.charset):
            entries = report.charset[category]
            _print(f"  {category} ({len(entries)})")
            for _, description, count in entries:
                _print(f"    {description:<60} {count:>9,}")

    if report.unknown_chars:
        _print()
        _print(f"UNKNOWN CODEPOINTS ({len(report.unknown_chars)})")
        for _, description, count in report.unknown_chars:
            _print(f"  {description:<60} {count:>9,}")

    return EXIT_OK if report.ok else EXIT_ERROR


# ---------------------------------------------------------------------------
# split
# ---------------------------------------------------------------------------


def cmd_split(args: argparse.Namespace) -> int:
    records = manifest_mod.read(args.manifest)

    if args.holdout or args.n_holdout:
        result = splits_mod.holdout_split(
            records,
            group_key=args.group_field,
            holdout=args.holdout or (),
            n_holdout=args.n_holdout or 1,
            seed=args.seed,
        )
    else:
        ratios = [float(r) for r in args.ratios.split(",")]
        names = [n.strip() for n in args.names.split(",")]
        result = splits_mod.stratified_split(
            records,
            group_key=args.group_field,
            ratios=ratios,
            names=names,
            seed=args.seed,
        )

    problems = splits_mod.check_leakage(result.splits, id_key=args.id_field)

    _print(result.summary())
    if problems:
        _print()
        _print("LEAKAGE")
        for problem in problems[:20]:
            _print(f"  {problem}")

    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, bucket in result.splits.items():
            path = out_dir / f"{name}.jsonl"
            manifest_mod.write(path, bucket)
            _print(f"wrote {len(bucket):>7} -> {path}")
        (out_dir / "split_report.json").write_text(
            json.dumps(result.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        _print(f"wrote split report -> {out_dir / 'split_report.json'}")

    return EXIT_ERROR if problems else EXIT_OK


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------


def cmd_eval(args: argparse.Namespace) -> int:
    config = _waqf_config(args)
    references = manifest_mod.read(args.manifest)

    if args.predictions:
        predictions = manifest_mod.read(args.predictions)
        pairs = manifest_mod.join_predictions(
            references,
            predictions,
            id_key=args.id_field,
            text_key=args.field,
            prediction_key=args.prediction_field,
        )
        if not pairs:
            _print(
                f"no {args.id_field!r} matched between {args.manifest} and "
                f"{args.predictions}",
                file=sys.stderr,
            )
            return EXIT_ERROR
        if len(pairs) < len(predictions):
            _print(
                f"note: {len(predictions) - len(pairs)} prediction(s) had no matching "
                f"manifest record and were dropped"
            )
    else:
        pairs = [
            {
                "segment_id": r.get(args.id_field, ""),
                "reference": r.get(args.field, ""),
                "hypothesis": r.get(args.prediction_field, ""),
                **{
                    k: v
                    for k, v in r.items()
                    if k not in (args.id_field, args.field, args.prediction_field)
                },
            }
            for r in references
        ]

    evaluation = metrics_mod.evaluate(
        pairs, config=config, group_by=tuple(g.strip() for g in args.group_by.split(","))
    )

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(
            json.dumps(evaluation.as_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _print(f"wrote results -> {args.json_out}")

    if args.html_out:
        path = report_mod.write_report(
            evaluation, args.html_out, subtitle=str(args.manifest), config=config
        )
        _print(f"wrote report  -> {path}")

    _print()
    _print(f"segments {len(evaluation.samples):,}   skipped {len(evaluation.skipped)}")
    _print("corpus rates")
    for spec in metrics_mod.METRICS:
        if spec.name in evaluation.corpus:
            _print(f"  {spec.name:<14} {100 * evaluation.corpus[spec.name]:7.2f}%")
    _print("buckets")
    for name, value in evaluation.buckets.items():
        _print(f"  {name:<14} {100 * value:7.2f}%")

    if evaluation.groups.get("reciter_slug"):
        _print()
        _print("by reciter (best first)")
        for group in evaluation.groups["reciter_slug"]:
            _print(
                f"  {group.key:<32} n={group.n:<6} "
                f"cer={100 * group.corpus.get('cer', 0):6.2f}%  "
                f"rasm={100 * group.corpus.get('rasm_cer', 0):6.2f}%"
            )

    if args.worst:
        _print()
        _print(f"worst {args.worst}")
        for sample in evaluation.worst(args.worst):
            _print(f"  [{sample.segment_id}] cer={100 * sample.counts['cer'].rate:.1f}%")
            _print(f"    ref: {T.normalize(sample.reference, 'waqf', config=config)}")
            _print(f"    hyp: {T.normalize(sample.hypothesis, 'waqf', config=config)}")

    threshold = args.fail_over
    if threshold is not None and evaluation.corpus.get("cer", 0.0) > threshold:
        _print(
            f"\nFAIL: corpus CER {evaluation.corpus['cer']:.4f} exceeds "
            f"--fail-over {threshold}",
            file=sys.stderr,
        )
        return EXIT_ERROR
    return EXIT_OK


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------


def cmd_report(args: argparse.Namespace) -> int:
    with open(args.results, encoding="utf-8") as fh:
        data = json.load(fh)

    samples = data.get("samples")
    if not samples:
        _print(
            f"{args.results} has no 'samples' array -- rerun `warsh eval` without "
            "--no-samples",
            file=sys.stderr,
        )
        return EXIT_ERROR

    pairs = [
        {
            "segment_id": s.get("segment_id", ""),
            "reference": s.get("reference", ""),
            "hypothesis": s.get("hypothesis", ""),
            **{
                k: v
                for k, v in s.items()
                if k not in ("segment_id", "reference", "hypothesis", "scores", "counts")
            },
        }
        for s in samples
    ]

    evaluation = metrics_mod.evaluate(pairs)
    path = report_mod.write_report(
        evaluation, args.output, title=args.title, subtitle=str(args.results),
        worst_n=args.worst,
    )
    _print(f"wrote report -> {path}")
    return EXIT_OK


# ---------------------------------------------------------------------------
# surah
# ---------------------------------------------------------------------------


def cmd_surah(args: argparse.Namespace) -> int:
    if not args.query:
        _print(f"{len(surahs_mod.SURAHS)} surahs, {surahs_mod.total_ayahs()} ayat")
        for surah in surahs_mod.SURAHS:
            _print(
                f"  {surah.number:>3}  {surah.name:<20} {surah.english_name:<22} "
                f"{surah.ayah_count:>3} ayat  {surah.revelation_type}"
            )
        return EXIT_OK

    found = surahs_mod.find(args.query)
    if found is None:
        _print(f"no surah matches {args.query!r}", file=sys.stderr)
        return EXIT_ERROR
    _print(f"number      {found.number}")
    _print(f"name        {found.name}")
    _print(f"english     {found.english_name} ({found.english_translation})")
    _print(f"ayat        {found.ayah_count}")
    _print(f"revelation  {found.revelation_type}")
    return EXIT_OK


# ---------------------------------------------------------------------------
# demo
# ---------------------------------------------------------------------------


def cmd_demo(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = _waqf_config(args)

    _print("1/6  building synthetic manifest")
    records = demo_mod.build_manifest(repeats=args.repeats)
    manifest_path = out_dir / "manifest.jsonl"
    manifest_mod.write(manifest_path, records)
    _print(f"     {len(records)} segments across {len(demo_mod.RECITERS)} reciters "
           f"-> {manifest_path}")

    _print("2/6  auditing manifest")
    audit = manifest_mod.check(records)
    _print(f"     {len(audit.errors)} error(s), {len(audit.warnings)} warning(s), "
           f"{audit.stats['distinct_chars']} distinct characters")
    for issue in audit.issues[:5]:
        _print(f"     {issue}")

    _print("3/6  splitting by reciter")
    stratified = splits_mod.stratified_split(records, seed=args.seed)
    holdout = splits_mod.holdout_split(records, n_holdout=1, seed=args.seed)
    for name, bucket in stratified.splits.items():
        manifest_mod.write(out_dir / f"{name}.jsonl", bucket)
    for name, bucket in holdout.splits.items():
        manifest_mod.write(out_dir / f"holdout_{name}.jsonl", bucket)
    _print("     " + stratified.summary().replace("\n", "\n     "))
    leaks = splits_mod.check_leakage(stratified.splits)
    _print(f"     leakage check: {'FAILED' if leaks else 'clean'}")

    _print("4/6  fabricating predictions")
    predictions = demo_mod.fake_predictions(records, seed=args.seed)
    predictions_path = out_dir / "predictions.jsonl"
    manifest_mod.write(predictions_path, predictions)
    _print(f"     {len(predictions)} predictions -> {predictions_path}")

    _print("5/6  scoring")
    pairs = manifest_mod.join_predictions(records, predictions)
    evaluation = metrics_mod.evaluate(pairs, config=config)
    results_path = out_dir / "results.json"
    results_path.write_text(
        json.dumps(evaluation.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for spec in metrics_mod.METRICS:
        if spec.name in evaluation.corpus:
            _print(f"     {spec.name:<14} {100 * evaluation.corpus[spec.name]:6.2f}%")

    _print("6/6  rendering report")
    html_path = report_mod.write_report(
        evaluation,
        out_dir / "report.html",
        title="Warsh ASR evaluation (demo)",
        subtitle="Synthetic data -- generated by `warsh demo`",
        config=config,
    )
    _print(f"     -> {html_path}")

    _print()
    _print(f"Done. Open {html_path}")
    return EXIT_OK


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def _add_waqf_flags(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("normalisation")
    group.add_argument(
        "--keep-tajweed",
        action="store_true",
        help="keep tajweed indicators (iqlab meem, silent zeros) instead of stripping them",
    )
    group.add_argument(
        "--skip-madd",
        action="store_true",
        help="do not add a sukun after a bare final long vowel",
    )
    group.add_argument(
        "--teh-marbuta-to-heh",
        action="store_true",
        help="rewrite a final teh marbuta as heh, matching pausal pronunciation",
    )
    group.add_argument(
        "--notebook-compat",
        action="store_true",
        help="reproduce the original training notebook's waqf rule "
        "(miniature letters may carry the sukun)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="warsh",
        description="Tooling for Warsh Quran recitation ASR: labels, splits, scoring, reports.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Start with:  warsh demo --out-dir out",
    )
    parser.add_argument("--version", action="version", version=f"warsh-lab {__version__}")
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("doctor", help="verify the installation and run a self-test")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("normalize", help="apply a normal form to text or a manifest")
    p.add_argument("text", nargs="*", help="text to normalise; omit to read stdin")
    p.add_argument("--form", default="waqf", choices=list(T.FORMS))
    p.add_argument("--manifest", help="normalise a column of this manifest instead")
    p.add_argument("--field", default="text_warsh", help="column to read")
    p.add_argument("--out-field", default="text_normalized", help="column to write")
    p.add_argument("--output", help="write the result here (default: stdout as JSONL)")
    _add_waqf_flags(p)
    p.set_defaults(func=cmd_normalize)

    p = sub.add_parser("check", help="audit a manifest before training")
    p.add_argument("manifest")
    p.add_argument("--field", default="text_warsh")
    p.add_argument("--id-field", default="segment_id")
    p.add_argument("--group-field", default="reciter_slug")
    p.add_argument("--charset", action="store_true", help="list every codepoint by category")
    p.add_argument("--json", action="store_true", help="emit the report as JSON")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("split", help="split a manifest by speaker")
    p.add_argument("manifest")
    p.add_argument("--out-dir", help="write <name>.jsonl files here")
    p.add_argument("--ratios", default="0.8,0.1,0.1")
    p.add_argument("--names", default="train,val,test")
    p.add_argument("--group-field", default="reciter_slug")
    p.add_argument("--id-field", default="segment_id")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--holdout",
        nargs="*",
        help="hold out these groups entirely, giving an unseen-speaker split",
    )
    p.add_argument(
        "--n-holdout",
        type=int,
        help="hold out this many groups, chosen deterministically",
    )
    p.set_defaults(func=cmd_split)

    p = sub.add_parser("eval", help="score predictions against references")
    p.add_argument("manifest", help="manifest carrying the reference text")
    p.add_argument("--predictions", help="predictions file to join on --id-field")
    p.add_argument("--field", default="text_warsh", help="reference column")
    p.add_argument("--prediction-field", default="prediction", help="hypothesis column")
    p.add_argument("--id-field", default="segment_id")
    p.add_argument("--group-by", default="reciter_slug,surah_number")
    p.add_argument("--json-out", help="write full results JSON here")
    p.add_argument("--html-out", help="write an HTML report here")
    p.add_argument("--worst", type=int, default=0, help="print the N worst segments")
    p.add_argument(
        "--fail-over",
        type=float,
        help="exit non-zero if corpus CER exceeds this (for CI)",
    )
    _add_waqf_flags(p)
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("report", help="render a results JSON to HTML")
    p.add_argument("results")
    p.add_argument("-o", "--output", default="report.html")
    p.add_argument("--title", default="Warsh ASR evaluation")
    p.add_argument("--worst", type=int, default=15)
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("surah", help="look up surah metadata")
    p.add_argument("query", nargs="?", help="number, Arabic name, or English name")
    p.set_defaults(func=cmd_surah)

    p = sub.add_parser("demo", help="run the whole pipeline on synthetic data")
    p.add_argument("--out-dir", default="demo_out")
    p.add_argument("--repeats", type=int, default=1, help="takes per reciter per verse")
    p.add_argument("--seed", type=int, default=42)
    _add_waqf_flags(p)
    p.set_defaults(func=cmd_demo)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    _force_utf8()
    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "func", None):
        parser.print_help()
        return EXIT_USAGE

    try:
        return args.func(args)
    except (FileNotFoundError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    except BrokenPipeError:  # pragma: no cover - piping into head
        return EXIT_OK
    except KeyboardInterrupt:  # pragma: no cover
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
