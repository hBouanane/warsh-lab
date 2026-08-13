"""ASR scoring for Warsh transcriptions.

One transcription is scored several ways at once, because a single CER number
hides the thing you actually want to know -- whether the model is mishearing
*consonants* or merely guessing *vowels*:

===============  ====================================================
metric           what it measures
===============  ====================================================
``cer``          characters, fully diacritised (the headline number)
``wer``          words, fully diacritised
``rasm_cer``     consonants only -- diacritics ignored
``skeleton_cer`` consonants with orthographic variants folded together
``harakat_cer``  vowel marks only -- consonants ignored
===============  ====================================================

A model with ``rasm_cer`` near zero and a high ``harakat_cer`` is hearing the
recitation correctly and losing points on diacritisation; the fix is a decoding
constraint or more diacritised text, not more audio.  The reverse pattern means
an acoustic problem.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from statistics import mean, median
from typing import Dict, Iterable, List, Sequence, Tuple

from . import text as T
from .distance import DELETE, INSERT, SUB, ErrorCounts, align, error_counts

__all__ = [
    "MetricSpec",
    "METRICS",
    "SampleResult",
    "GroupResult",
    "Evaluation",
    "cer",
    "wer",
    "score_pair",
    "evaluate",
]


# ---------------------------------------------------------------------------
# Metric registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricSpec:
    """How one metric turns a raw string into the sequence that gets aligned."""

    name: str
    form: str
    unit: str  # "char" or "word"
    description: str

    def units(self, raw: str, config: T.WaqfConfig) -> Sequence[str]:
        normalised = T.normalize(raw, self.form, config=config)
        if self.unit == "word":
            return normalised.split()
        return list(normalised)


METRICS: Tuple[MetricSpec, ...] = (
    MetricSpec("cer", "waqf", "char", "Character error rate, fully diacritised"),
    MetricSpec("wer", "waqf", "word", "Word error rate, fully diacritised"),
    MetricSpec("rasm_cer", "rasm", "char", "Consonants only, diacritics ignored"),
    MetricSpec("skeleton_cer", "skeleton", "char", "Consonants, letter variants folded"),
    MetricSpec("harakat_cer", "harakat", "char", "Vowel marks only"),
)

_BY_NAME = {m.name: m for m in METRICS}


# ---------------------------------------------------------------------------
# Convenience one-shot scorers
# ---------------------------------------------------------------------------


def cer(reference: str, hypothesis: str, *, form: str = "waqf") -> float:
    """Character error rate between two strings under normal form *form*."""
    ref = list(T.normalize(reference, form))
    hyp = list(T.normalize(hypothesis, form))
    return error_counts(ref, hyp).rate


def wer(reference: str, hypothesis: str, *, form: str = "waqf") -> float:
    """Word error rate between two strings under normal form *form*."""
    ref = T.normalize(reference, form).split()
    hyp = T.normalize(hypothesis, form).split()
    return error_counts(ref, hyp).rate


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class SampleResult:
    """Every metric for a single utterance, plus its identifying metadata."""

    segment_id: str
    reference: str
    hypothesis: str
    counts: Dict[str, ErrorCounts] = field(default_factory=dict)
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def scores(self) -> Dict[str, float]:
        return {name: counts.rate for name, counts in self.counts.items()}

    @property
    def exact(self) -> bool:
        return self.counts["cer"].errors == 0 if "cer" in self.counts else False

    def as_dict(self) -> dict:
        return {
            "segment_id": self.segment_id,
            "reference": self.reference,
            "hypothesis": self.hypothesis,
            **{k: str(v) for k, v in self.metadata.items()},
            "scores": {k: round(v, 6) for k, v in self.scores.items()},
            "counts": {k: v.as_dict() for k, v in self.counts.items()},
        }


@dataclass
class GroupResult:
    """Corpus-level scores for one slice of the test set (a reciter, a surah)."""

    key: str
    n: int
    corpus: Dict[str, float]
    mean_cer: float
    median_cer: float

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "n": self.n,
            "corpus": {k: round(v, 6) for k, v in self.corpus.items()},
            "mean_cer": round(self.mean_cer, 6),
            "median_cer": round(self.median_cer, 6),
        }


@dataclass
class Evaluation:
    """The full result of scoring a test set."""

    samples: List[SampleResult]
    corpus: Dict[str, float]
    mean: Dict[str, float]
    median: Dict[str, float]
    buckets: Dict[str, float]
    groups: Dict[str, List[GroupResult]]
    confusions: List[Tuple[str, str, int]]
    deletions: List[Tuple[str, int]]
    insertions: List[Tuple[str, int]]
    skipped: List[Dict[str, str]] = field(default_factory=list)
    config: Dict[str, object] = field(default_factory=dict)

    def worst(self, n: int = 10, metric: str = "cer") -> List[SampleResult]:
        """The *n* samples with the highest score on *metric* (worst first)."""
        scored = [s for s in self.samples if metric in s.counts]
        return sorted(scored, key=lambda s: s.counts[metric].rate, reverse=True)[:n]

    def best(self, n: int = 10, metric: str = "cer") -> List[SampleResult]:
        scored = [s for s in self.samples if metric in s.counts]
        return sorted(scored, key=lambda s: s.counts[metric].rate)[:n]

    def as_dict(self, *, include_samples: bool = True) -> dict:
        out = {
            "summary": {
                "n": len(self.samples),
                "skipped": len(self.skipped),
                "corpus": {k: round(v, 6) for k, v in self.corpus.items()},
                "mean": {k: round(v, 6) for k, v in self.mean.items()},
                "median": {k: round(v, 6) for k, v in self.median.items()},
                "buckets": {k: round(v, 6) for k, v in self.buckets.items()},
            },
            "config": self.config,
            "groups": {
                name: [g.as_dict() for g in groups]
                for name, groups in self.groups.items()
            },
            "confusions": [
                {"reference": r, "hypothesis": h, "count": c, "kind": "substitution"}
                for r, h, c in self.confusions
            ],
            "deletions": [{"char": ch, "count": c} for ch, c in self.deletions],
            "insertions": [{"char": ch, "count": c} for ch, c in self.insertions],
            "skipped_records": self.skipped,
        }
        if include_samples:
            out["samples"] = [s.as_dict() for s in self.samples]
        return out


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_pair(
    reference: str,
    hypothesis: str,
    *,
    metrics: Sequence[MetricSpec] = METRICS,
    config: T.WaqfConfig = T.DEFAULT_WAQF,
) -> Dict[str, ErrorCounts]:
    """Run every metric in *metrics* over one reference/hypothesis pair."""
    return {
        spec.name: error_counts(
            spec.units(reference, config), spec.units(hypothesis, config)
        )
        for spec in metrics
    }


def _corpus_rates(
    per_sample: Dict[str, List[ErrorCounts]]
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Corpus, mean-of-sample, and median-of-sample rates.

    The corpus rate pools errors and reference lengths before dividing, so long
    utterances carry proportional weight -- this is the number to report.  The
    mean of per-sample rates is kept alongside because it exposes a tail of
    short, badly-scored clips that pooling would hide.
    """
    corpus, means, medians = {}, {}, {}
    for name, counts_list in per_sample.items():
        if not counts_list:
            continue
        total = ErrorCounts()
        for c in counts_list:
            total = total + c
        corpus[name] = total.rate
        rates = [c.rate for c in counts_list]
        means[name] = mean(rates)
        medians[name] = median(rates)
    return corpus, means, medians


def _group_results(
    samples: Sequence[SampleResult], field_name: str
) -> List[GroupResult]:
    buckets: Dict[str, List[SampleResult]] = defaultdict(list)
    for s in samples:
        value = s.metadata.get(field_name)
        if value is None or value == "":
            continue
        buckets[str(value)].append(s)

    results: List[GroupResult] = []
    for key, group in buckets.items():
        per_metric: Dict[str, List[ErrorCounts]] = defaultdict(list)
        for s in group:
            for name, counts in s.counts.items():
                per_metric[name].append(counts)
        corpus, means, medians = _corpus_rates(per_metric)
        results.append(
            GroupResult(
                key=key,
                n=len(group),
                corpus=corpus,
                mean_cer=means.get("cer", 0.0),
                median_cer=medians.get("cer", 0.0),
            )
        )
    results.sort(key=lambda g: g.corpus.get("cer", 0.0))
    return results


def _confusion_tables(
    samples: Sequence[SampleResult],
    config: T.WaqfConfig,
    top: int,
) -> Tuple[List[Tuple[str, str, int]], List[Tuple[str, int]], List[Tuple[str, int]]]:
    """Which characters get swapped, dropped, and hallucinated most often."""
    subs: Counter = Counter()
    dels: Counter = Counter()
    ins: Counter = Counter()

    for s in samples:
        ref = list(T.normalize(s.reference, "waqf", config=config))
        hyp = list(T.normalize(s.hypothesis, "waqf", config=config))
        if ref == hyp:
            continue
        for op, ref_ch, hyp_ch in align(ref, hyp):
            if op == SUB:
                subs[(ref_ch, hyp_ch)] += 1
            elif op == DELETE:
                dels[ref_ch] += 1
            elif op == INSERT:
                ins[hyp_ch] += 1

    return (
        [(r, h, n) for (r, h), n in subs.most_common(top)],
        dels.most_common(top),
        ins.most_common(top),
    )


def evaluate(
    pairs: Iterable[Dict[str, object]],
    *,
    metrics: Sequence[MetricSpec] = METRICS,
    config: T.WaqfConfig = T.DEFAULT_WAQF,
    group_by: Sequence[str] = ("reciter_slug", "surah_number"),
    confusion_top: int = 25,
    reference_key: str = "reference",
    hypothesis_key: str = "hypothesis",
    id_key: str = "segment_id",
) -> Evaluation:
    """Score a test set.

    *pairs* is any iterable of mappings carrying a reference, a hypothesis, an
    id, and whatever metadata you want to slice by.  Records missing either side
    are collected into :attr:`Evaluation.skipped` rather than aborting the run --
    a decode that crashed on 3 of 4000 clips should still produce a report.
    """
    samples: List[SampleResult] = []
    skipped: List[Dict[str, str]] = []

    for index, record in enumerate(pairs):
        segment_id = str(record.get(id_key) or f"#{index}")
        reference = record.get(reference_key)
        hypothesis = record.get(hypothesis_key)

        if reference is None or str(reference).strip() == "":
            skipped.append({"segment_id": segment_id, "reason": "empty reference"})
            continue
        if hypothesis is None:
            skipped.append({"segment_id": segment_id, "reason": "missing hypothesis"})
            continue

        metadata = {
            k: v
            for k, v in record.items()
            if k not in (reference_key, hypothesis_key, id_key)
        }
        samples.append(
            SampleResult(
                segment_id=segment_id,
                reference=str(reference),
                hypothesis=str(hypothesis),
                counts=score_pair(
                    str(reference), str(hypothesis), metrics=metrics, config=config
                ),
                metadata=metadata,
            )
        )

    per_metric: Dict[str, List[ErrorCounts]] = defaultdict(list)
    for s in samples:
        for name, counts in s.counts.items():
            per_metric[name].append(counts)

    corpus, means, medians = _corpus_rates(per_metric)

    cer_rates = [s.counts["cer"].rate for s in samples if "cer" in s.counts]
    n = len(cer_rates)
    buckets = {
        "exact_match": sum(1 for r in cer_rates if r == 0.0) / n if n else 0.0,
        "cer_le_0.05": sum(1 for r in cer_rates if r <= 0.05) / n if n else 0.0,
        "cer_le_0.10": sum(1 for r in cer_rates if r <= 0.10) / n if n else 0.0,
        "cer_gt_0.50": sum(1 for r in cer_rates if r > 0.50) / n if n else 0.0,
    }

    groups = {name: _group_results(samples, name) for name in group_by}
    groups = {name: g for name, g in groups.items() if g}

    subs, dels, ins = _confusion_tables(samples, config, confusion_top)

    return Evaluation(
        samples=samples,
        corpus=corpus,
        mean=means,
        median=medians,
        buckets=buckets,
        groups=groups,
        confusions=subs,
        deletions=dels,
        insertions=ins,
        skipped=skipped,
        config={
            "metrics": [m.name for m in metrics],
            "waqf": {
                "drop_tajweed": config.drop_tajweed,
                "sukun_carriers": config.sukun_carriers,
                "skip_madd_carriers": config.skip_madd_carriers,
                "teh_marbuta_to_heh": config.teh_marbuta_to_heh,
            },
            "group_by": list(group_by),
        },
    )
