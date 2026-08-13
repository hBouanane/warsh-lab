"""Self-contained HTML reports for an :class:`~warshlab.metrics.Evaluation`.

No CDN, no build step, no network at render or view time -- one file you can
open from a training box over SSH, drop in a PR, or mail to someone.  Arabic is
laid out RTL with the browser's own shaping; the diff view marks per-character
substitutions, deletions, and insertions inline.

Colours follow ``prefers-color-scheme``, so the page is readable in both themes.
"""

from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from . import text as T
from .distance import DELETE, EQUAL, SUB, align
from .metrics import METRICS, Evaluation, SampleResult

__all__ = ["render", "write_report"]

_CSS = """
:root {
  --bg: #ffffff; --panel: #f6f7f9; --border: #dfe3e8; --fg: #14171a;
  --muted: #5b6470; --accent: #1f6feb; --good: #1a7f37; --warn: #9a6700;
  --bad: #cf222e; --sub: #fde7e9; --del: #fff1cc; --ins: #dcf5e4;
  --sub-fg: #8b1420; --del-fg: #7a5200; --ins-fg: #10592a;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0d1117; --panel: #161b22; --border: #2b323b; --fg: #e6edf3;
    --muted: #9198a1; --accent: #58a6ff; --good: #3fb950; --warn: #d29922;
    --bad: #f85149; --sub: #4a1d24; --del: #46330d; --ins: #12331f;
    --sub-fg: #ffb3ba; --del-fg: #f0cc7a; --ins-fg: #7ee2a8;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 2rem 1.25rem 4rem; background: var(--bg); color: var(--fg);
  font: 15px/1.55 ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
}
.wrap { max-width: 1080px; margin: 0 auto; }
h1 { font-size: 1.6rem; margin: 0 0 .25rem; letter-spacing: -.01em; }
h2 { font-size: 1.1rem; margin: 2.5rem 0 .75rem; letter-spacing: -.01em; }
h2:first-of-type { margin-top: 1.75rem; }
.sub { color: var(--muted); font-size: .9rem; margin: 0 0 .5rem; }
.cards { display: grid; gap: .75rem; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); margin-top: 1.25rem; }
.card { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: .85rem .95rem; }
.card .label { color: var(--muted); font-size: .74rem; text-transform: uppercase; letter-spacing: .05em; }
.card .value { font-size: 1.5rem; font-weight: 640; margin-top: .3rem; font-variant-numeric: tabular-nums; }
.card .hint { color: var(--muted); font-size: .74rem; margin-top: .15rem; }
table { width: 100%; border-collapse: collapse; font-size: .88rem; }
th, td { text-align: left; padding: .45rem .6rem; border-bottom: 1px solid var(--border); }
th { color: var(--muted); font-weight: 600; font-size: .76rem; text-transform: uppercase; letter-spacing: .04em; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
tbody tr:hover { background: var(--panel); }
.panel { background: var(--panel); border: 1px solid var(--border); border-radius: 10px; padding: .35rem .85rem; }
.ar { direction: rtl; text-align: right; font-size: 1.35rem; line-height: 2.1;
      font-family: "Amiri Quran", "Scheherazade New", "Traditional Arabic", "Segoe UI", serif; }
.pair { border: 1px solid var(--border); border-radius: 10px; padding: .75rem .9rem; margin-bottom: .8rem; background: var(--panel); }
.pair .head { display: flex; justify-content: space-between; gap: 1rem; color: var(--muted); font-size: .78rem; margin-bottom: .4rem; }
.pair .tag { font-variant-numeric: tabular-nums; }
.row-label { color: var(--muted); font-size: .7rem; text-transform: uppercase; letter-spacing: .05em; margin-top: .35rem; }
mark { border-radius: 3px; padding: 0 1px; }
mark.sub { background: var(--sub); color: var(--sub-fg); }
mark.del { background: var(--del); color: var(--del-fg); }
mark.ins { background: var(--ins); color: var(--ins-fg); }
.legend { color: var(--muted); font-size: .8rem; display: flex; gap: 1rem; flex-wrap: wrap; margin: .4rem 0 .9rem; }
.bar { height: 8px; border-radius: 4px; background: var(--accent); display: inline-block; vertical-align: middle; }
.good { color: var(--good); } .warn { color: var(--warn); } .bad { color: var(--bad); }
code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: .85em; }
footer { color: var(--muted); font-size: .78rem; margin-top: 3rem; border-top: 1px solid var(--border); padding-top: .9rem; }
.note { border-inline-start: 3px solid var(--accent); padding: .5rem .8rem; background: var(--panel);
        border-radius: 0 8px 8px 0; color: var(--muted); font-size: .86rem; margin: .6rem 0 1rem; }
"""


def _esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def _pct(value: float, digits: int = 2) -> str:
    return f"{100 * value:.{digits}f}%"


def _rate_class(rate: float) -> str:
    if rate <= 0.05:
        return "good"
    if rate <= 0.20:
        return "warn"
    return "bad"


def _diff_html(reference: str, hypothesis: str, config: T.WaqfConfig) -> Tuple[str, str]:
    """Character-aligned reference/hypothesis, with the edits marked up."""
    ref = list(T.normalize(reference, "waqf", config=config))
    hyp = list(T.normalize(hypothesis, "waqf", config=config))

    ref_out: List[str] = []
    hyp_out: List[str] = []

    for op, ref_ch, hyp_ch in align(ref, hyp):
        if op == EQUAL:
            ref_out.append(_esc(ref_ch))
            hyp_out.append(_esc(hyp_ch))
        elif op == SUB:
            ref_out.append(f'<mark class="sub">{_esc(ref_ch)}</mark>')
            hyp_out.append(f'<mark class="sub">{_esc(hyp_ch)}</mark>')
        elif op == DELETE:
            ref_out.append(f'<mark class="del">{_esc(ref_ch)}</mark>')
        else:  # INSERT
            hyp_out.append(f'<mark class="ins">{_esc(hyp_ch)}</mark>')

    return "".join(ref_out), "".join(hyp_out)


def _histogram(rates: Sequence[float], buckets: int = 20) -> str:
    """Inline SVG histogram of per-sample CER.

    The shape matters more than the mean: one hump near zero with a thin tail is
    a healthy model, two humps means two populations (usually one reciter, or
    one recording condition, behaving differently from the rest).
    """
    if not rates:
        return "<p class='sub'>No samples to plot.</p>"

    width, height, pad = 1000, 190, 28
    top = min(1.0, max(rates)) or 1.0
    edges = [top * i / buckets for i in range(buckets + 1)]
    counts = [0] * buckets
    for rate in rates:
        index = min(buckets - 1, int(rate / top * buckets)) if top else 0
        counts[index] += 1
    peak = max(counts) or 1

    bar_width = (width - 2 * pad) / buckets
    bars: List[str] = []
    for i, count in enumerate(counts):
        bar_height = (height - 2 * pad) * count / peak
        x = pad + i * bar_width
        y = height - pad - bar_height
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width - 2:.1f}" '
            f'height="{bar_height:.1f}" rx="2" fill="currentColor" opacity="0.75">'
            f"<title>CER {edges[i]:.3f}-{edges[i + 1]:.3f}: {count} sample(s)</title></rect>"
        )

    ticks: List[str] = []
    for i in range(0, buckets + 1, max(1, buckets // 5)):
        x = pad + i * bar_width
        ticks.append(
            f'<text x="{x:.1f}" y="{height - 8}" font-size="11" fill="currentColor" '
            f'opacity="0.65" text-anchor="middle">{edges[i]:.2f}</text>'
        )

    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
        f'role="img" aria-label="Distribution of per-sample CER" '
        f'style="color: var(--accent)">{"".join(bars)}{"".join(ticks)}</svg>'
    )


def _summary_cards(ev: Evaluation) -> str:
    cards = [
        ("Segments", f"{len(ev.samples):,}", f"{len(ev.skipped)} skipped"),
        ("CER", _pct(ev.corpus.get("cer", 0.0)), "corpus, diacritised"),
        ("WER", _pct(ev.corpus.get("wer", 0.0)), "corpus, diacritised"),
        ("Rasm CER", _pct(ev.corpus.get("rasm_cer", 0.0)), "consonants only"),
        ("Harakat CER", _pct(ev.corpus.get("harakat_cer", 0.0)), "vowel marks only"),
        ("Exact", _pct(ev.buckets.get("exact_match", 0.0), 1), "character-identical"),
    ]
    return "".join(
        f'<div class="card"><div class="label">{_esc(label)}</div>'
        f'<div class="value">{_esc(value)}</div>'
        f'<div class="hint">{_esc(hint)}</div></div>'
        for label, value, hint in cards
    )


def _diagnosis(ev: Evaluation) -> str:
    """One sentence naming the dominant error mode, from the metric spread."""
    rasm = ev.corpus.get("rasm_cer")
    harakat = ev.corpus.get("harakat_cer")
    overall = ev.corpus.get("cer")
    if rasm is None or harakat is None or overall is None:
        return ""

    if overall < 0.02:
        body = (
            "Overall CER is under 2%. Confirm on a held-out <em>unseen reciter</em> "
            "split before believing it -- a stratified split cannot detect voice "
            "memorisation."
        )
    elif harakat > 2 * rasm and rasm < 0.10:
        body = (
            f"Consonants are largely correct (rasm CER {_pct(rasm)}) while diacritics "
            f"are not (harakat CER {_pct(harakat)}). The acoustic model is hearing the "
            "recitation; the errors are in diacritisation, so constrained decoding "
            "against the mushaf text will buy more than additional audio."
        )
    elif rasm > harakat:
        body = (
            f"Rasm CER ({_pct(rasm)}) exceeds harakat CER ({_pct(harakat)}): the model is "
            "mishearing consonants, not just guessing vowels. Look at audio quality, "
            "segment boundaries, and per-reciter scores below before touching the "
            "label pipeline."
        )
    else:
        body = (
            f"Consonant and diacritic error rates are comparable ({_pct(rasm)} vs "
            f"{_pct(harakat)}); errors are spread across both. The worst segments below "
            "are the place to start."
        )
    return f'<div class="note">{body}</div>'


def _metric_table(ev: Evaluation) -> str:
    rows: List[str] = []
    for spec in METRICS:
        if spec.name not in ev.corpus:
            continue
        corpus = ev.corpus[spec.name]
        rows.append(
            "<tr>"
            f"<td><code>{_esc(spec.name)}</code></td>"
            f"<td>{_esc(spec.description)}</td>"
            f'<td class="num {_rate_class(corpus)}">{_pct(corpus)}</td>'
            f'<td class="num">{_pct(ev.mean.get(spec.name, 0.0))}</td>'
            f'<td class="num">{_pct(ev.median.get(spec.name, 0.0))}</td>'
            "</tr>"
        )
    return (
        '<div class="panel"><table><thead><tr><th>Metric</th><th>Measures</th>'
        '<th class="num">Corpus</th><th class="num">Mean</th><th class="num">Median</th>'
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def _group_table(ev: Evaluation, name: str) -> str:
    groups = ev.groups.get(name, [])
    if not groups:
        return ""

    worst = max((g.corpus.get("cer", 0.0) for g in groups), default=0.0) or 1.0
    rows: List[str] = []
    for group in groups:
        cer = group.corpus.get("cer", 0.0)
        width = 100 * cer / worst
        rows.append(
            "<tr>"
            f"<td>{_esc(group.key)}</td>"
            f'<td class="num">{group.n:,}</td>'
            f'<td class="num {_rate_class(cer)}">{_pct(cer)}</td>'
            f'<td class="num">{_pct(group.corpus.get("rasm_cer", 0.0))}</td>'
            f'<td class="num">{_pct(group.corpus.get("wer", 0.0))}</td>'
            f'<td><span class="bar" style="width:{width:.1f}%"></span></td>'
            "</tr>"
        )

    label = name.replace("_", " ")
    return (
        f"<h2>By {_esc(label)}</h2>"
        '<div class="panel"><table><thead><tr>'
        f"<th>{_esc(label)}</th><th class='num'>N</th><th class='num'>CER</th>"
        "<th class='num'>Rasm</th><th class='num'>WER</th><th></th>"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def _confusion_table(ev: Evaluation) -> str:
    if not (ev.confusions or ev.deletions or ev.insertions):
        return ""

    from . import chars as C

    def describe(char: str) -> str:
        return _esc(C.describe(char))

    sub_rows = "".join(
        "<tr>"
        f'<td class="ar" style="font-size:1.1rem">{_esc(ref)}</td>'
        f'<td class="ar" style="font-size:1.1rem">{_esc(hyp)}</td>'
        f'<td class="num">{count:,}</td>'
        f"<td><code>{describe(ref)}</code> &rarr; <code>{describe(hyp)}</code></td>"
        "</tr>"
        for ref, hyp, count in ev.confusions
    )
    del_rows = "".join(
        f'<tr><td class="ar" style="font-size:1.1rem">{_esc(ch)}</td>'
        f'<td class="num">{count:,}</td><td><code>{describe(ch)}</code></td></tr>'
        for ch, count in ev.deletions
    )
    ins_rows = "".join(
        f'<tr><td class="ar" style="font-size:1.1rem">{_esc(ch)}</td>'
        f'<td class="num">{count:,}</td><td><code>{describe(ch)}</code></td></tr>'
        for ch, count in ev.insertions
    )

    parts = [
        "<h2>Character confusions</h2>",
        '<p class="sub">What the model swaps, drops, and invents most often. A single '
        "pair dominating this table is usually a label-normalisation bug rather than "
        "an acoustic one.</p>",
    ]
    if sub_rows:
        parts.append(
            '<div class="panel"><table><thead><tr><th>Reference</th><th>Predicted</th>'
            '<th class="num">Count</th><th>Codepoints</th></tr></thead>'
            f"<tbody>{sub_rows}</tbody></table></div>"
        )
    if del_rows:
        parts.append("<h2>Most-dropped characters</h2>")
        parts.append(
            '<div class="panel"><table><thead><tr><th>Char</th><th class="num">Count</th>'
            f"<th>Codepoint</th></tr></thead><tbody>{del_rows}</tbody></table></div>"
        )
    if ins_rows:
        parts.append("<h2>Most-hallucinated characters</h2>")
        parts.append(
            '<div class="panel"><table><thead><tr><th>Char</th><th class="num">Count</th>'
            f"<th>Codepoint</th></tr></thead><tbody>{ins_rows}</tbody></table></div>"
        )
    return "".join(parts)


def _sample_block(sample: SampleResult, config: T.WaqfConfig) -> str:
    ref_html, hyp_html = _diff_html(sample.reference, sample.hypothesis, config)
    cer = sample.counts["cer"].rate if "cer" in sample.counts else 0.0
    meta = " · ".join(
        f"{_esc(k)}: {_esc(v)}"
        for k, v in list(sample.metadata.items())[:4]
        if v not in (None, "")
    )
    return (
        '<div class="pair">'
        f'<div class="head"><span><code>{_esc(sample.segment_id)}</code>'
        f'{" · " + meta if meta else ""}</span>'
        f'<span class="tag {_rate_class(cer)}">CER {_pct(cer)}</span></div>'
        f'<div class="row-label">Reference</div><div class="ar">{ref_html}</div>'
        f'<div class="row-label">Prediction</div><div class="ar">{hyp_html}</div>'
        "</div>"
    )


def render(
    evaluation: Evaluation,
    *,
    title: str = "Warsh ASR evaluation",
    subtitle: str = "",
    worst_n: int = 15,
    config: T.WaqfConfig = T.DEFAULT_WAQF,
    generated_at: Optional[str] = None,
) -> str:
    """Render *evaluation* to a complete, standalone HTML document."""
    stamp = generated_at or datetime.now().strftime("%Y-%m-%d %H:%M")
    rates = [s.counts["cer"].rate for s in evaluation.samples if "cer" in s.counts]

    worst_blocks = "".join(
        _sample_block(sample, config) for sample in evaluation.worst(worst_n)
    )

    group_sections = "".join(
        _group_table(evaluation, name) for name in evaluation.groups
    )

    skipped_note = ""
    if evaluation.skipped:
        rows = "".join(
            f"<tr><td><code>{_esc(r.get('segment_id', ''))}</code></td>"
            f"<td>{_esc(r.get('reason', ''))}</td></tr>"
            for r in evaluation.skipped[:50]
        )
        skipped_note = (
            f"<h2>Skipped ({len(evaluation.skipped)})</h2>"
            '<div class="panel"><table><thead><tr><th>Segment</th><th>Reason</th>'
            f"</tr></thead><tbody>{rows}</tbody></table></div>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_esc(title)}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="wrap">
  <h1>{_esc(title)}</h1>
  <p class="sub">{_esc(subtitle) if subtitle else "Generated by warsh-lab"} &middot; {_esc(stamp)}</p>

  <div class="cards">{_summary_cards(evaluation)}</div>

  {_diagnosis(evaluation)}

  <h2>Metrics</h2>
  <p class="sub">Corpus rates pool errors and reference lengths before dividing, so long
  segments carry proportional weight. Mean and median are over per-segment rates and
  expose a tail that pooling hides.</p>
  {_metric_table(evaluation)}

  <h2>CER distribution</h2>
  <p class="sub">Per-segment character error rate. Hover a bar for its range and count.</p>
  <div class="panel" style="padding: .8rem">{_histogram(rates)}</div>

  {group_sections}

  {_confusion_table(evaluation)}

  <h2>Worst {min(worst_n, len(evaluation.samples))} segments</h2>
  <div class="legend">
    <span><mark class="sub">shaded</mark> substituted</span>
    <span><mark class="del">shaded</mark> in reference, missing from prediction</span>
    <span><mark class="ins">shaded</mark> in prediction, not in reference</span>
  </div>
  {worst_blocks or '<p class="sub">Nothing to show.</p>'}

  {skipped_note}

  <footer>
    warsh-lab &middot; {len(evaluation.samples):,} segments scored &middot;
    waqf config: <code>{_esc(json.dumps(evaluation.config.get("waqf", {}), ensure_ascii=False))}</code>
  </footer>
</div>
</body>
</html>
"""


def write_report(
    evaluation: Evaluation, path: str | Path, **kwargs
) -> Path:
    """Render and write the report; returns the path written."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(render(evaluation, **kwargs), encoding="utf-8")
    return destination
