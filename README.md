# warsh-lab

Tooling for the parts of a Warsh Quran recitation ASR project that sit either side of the GPU: turning mushaf text into labels, splitting a corpus without leaking speakers, scoring transcriptions in a way that tells you *which kind* of error you have, and rendering the result as a report someone can read.

**Zero runtime dependencies.** Standard library only — it imports on a training box with no wheels installed and no network.

```bash
pip install -e .
warsh demo --out-dir out      # builds data, splits, scores, writes out/report.html
```

`examples/report.html` is that output, committed so you can look at it before running anything.

---

## Why

Fine-tuning Whisper on Warsh recitation, the model is only half the problem. The other half:

- **Labels.** Every clip ends at a stop, so every label must be in pausal form (waqf). Train on non-pausal text and you teach the model to invent a final vowel it never hears.
- **Splits.** Reciters have strong, learnable voice signatures. A split that shows every reciter to the model during training measures memorisation as much as transcription.
- **Metrics.** One CER number cannot distinguish "misheard the consonants" from "guessed the vowels" — and those two failures call for completely different fixes.

Each of those is a few lines of code that is easy to get subtly wrong and hard to notice being wrong. This library gets them right once, with tests.

## What it does

### Text normalisation, on explicit Unicode tables

Every codepoint in Quranic Arabic is assigned to exactly one category — letter, superscript letter, haraka, hamza mark, tajweed mark, waqf mark, ornament, formatting — and a test asserts the categories never overlap.

That test exists because of a specific bug class. A hand-rolled normaliser usually keeps two opaque string literals, one of "letters" and one of "marks to strip", and they silently intersect: U+06E7 SMALL HIGH YEH and U+06E8 SMALL HIGH NOON commonly land in both. Which behaviour wins then depends on the order the sets happen to be consulted in, and nothing ever tells you.

Six normal forms:

| form | keeps | use |
|---|---|---|
| `raw` | everything pronounceable | baseline |
| `clean` | drops waqf signs, ornaments, tajweed marks | pre-processing |
| `waqf` | pausal form — final vowel → sukun | **training labels** |
| `rasm` | letters only | consonant-only scoring |
| `skeleton` | letters, variants folded | orthography-insensitive scoring |
| `harakat` | vowel marks only | diacritisation-only scoring |

```python
from warshlab import text

text.apply_waqf("فَأَصَٰبَتْكُم مُّصِيبَةُ اُ۬لْمَوْتِۖ")   # 'فَأَصَٰبَتْكُم مُّصِيبَةُ اُ۬لْمَوْتْ'
text.to_rasm("بِسْمِ اللَّهِ")                              # 'بسم الله'
```

Pausal rules that are debatable are options, not defaults — `skip_madd_carriers`, `teh_marbuta_to_heh`, `keep_tajweed`. And `text.NOTEBOOK_WAQF` reproduces the original training notebook's rule exactly, so previously generated labels stay comparable.

### Splits that can detect voice memorisation

```python
from warshlab import splits

seen   = splits.stratified_split(records, group_key="reciter_slug")   # every reciter in every split
unseen = splits.holdout_split(records, n_holdout=1)                   # whole reciters held out

assert not splits.check_leakage(seen.splits)
```

Report both. The first tracks training progress; the second is the one that predicts what a user hears. A model that has quietly memorised eleven voices looks excellent on the first right up until release.

Both are deterministic and **stable under corpus growth** — shuffling is seeded per group with a hash independent of `PYTHONHASHSEED`, so adding a reciter next month does not reshuffle the reciters you already split, and last month's validation scores stay comparable.

Groups too small to spread across every split are reported as warnings rather than raising, so one two-clip reciter does not abort the run.

### Scoring that separates acoustic from diacritic errors

```python
from warshlab import metrics

ev = metrics.evaluate(pairs, group_by=("reciter_slug", "surah_number"))
ev.corpus       # {'cer': 0.113, 'wer': 0.459, 'rasm_cer': 0.069, 'harakat_cer': 0.199}
ev.worst(10)
ev.confusions   # [('ة', 'ه', 41), ...]
```

Low `rasm_cer` with high `harakat_cer` means the model is hearing the recitation and losing points on diacritisation — constrained decoding against the mushaf buys more than more audio. The reverse means an acoustic problem. The headline CER alone cannot tell you which you have.

Corpus rates pool errors and reference lengths before dividing, so long segments carry proportional weight; per-sample mean and median are reported alongside, because pooling hides a tail of short bad clips.

### A manifest audit before you spend GPU hours

```bash
warsh check manifest.jsonl --charset
```

Duplicate ids, out-of-range surah/ayah references (validated against the shipped 114-surah table), empty or letterless text, tiny reciter groups, what fraction of labels are already pausal — and a full charset listing, so an unexpected codepoint surfaces here rather than inside the tokenizer.

### Reports

`warsh eval ... --html-out report.html` writes one standalone file — no CDN, no scripts, no network at render or view time. Summary cards, a plain-language diagnosis of the dominant error mode, per-reciter and per-surah tables, a CER histogram, character confusion tables annotated with Unicode names, and the worst segments with per-character diffs in RTL. Light and dark.

## CLI

```
warsh doctor                  verify the install, run a self-test
warsh normalize TEXT          apply a normal form (--form waqf|rasm|skeleton|harakat|clean|raw)
warsh check MANIFEST          audit a corpus before training
warsh split MANIFEST          stratified, or --n-holdout for unseen-speaker
warsh eval MANIFEST           score predictions, --json-out / --html-out
warsh report RESULTS.json     re-render a report from saved results
warsh surah QUERY             surah metadata lookup
warsh demo                    the whole pipeline on synthetic data
```

Every command exits non-zero on failure. `warsh eval --fail-over 0.15` gates a training run in CI.

## Using it with the Whisper notebook

`examples/whisper_integration.py` maps each piece onto the cell it replaces in `whisper_warsh_training.ipynb` — the `apply_waqf` cell, the manual stratification cell, `compute_metrics`, and the final evaluation cell. Run it standalone to watch those pieces work on the built-in sample corpus:

```bash
python examples/whisper_integration.py
```

The metrics path also removes the `jiwer` and `evaluate` imports from the training loop, so nothing downloads a metric script mid-run.

## Tests

```bash
pytest
```

218 tests, no network, no fixtures to download. They cover the category tables staying disjoint, every normal form being idempotent and empty-safe, alignment reconstructing both sequences, corpus-vs-mean aggregation, split determinism and stability under growth, leakage detection, manifest round-trips through four formats, and the report being self-contained and HTML-escaped.

## Sample text

`warshlab/demo.py` carries short, well-known passages in common Uthmani diacritisation so the pipeline runs end to end with no dataset. **It is not a Warsh reference** — Warsh orthography differs in ways that matter. Point the tools at your own corpus for real work.

## Layout

```
src/warshlab/
  chars.py      Unicode category tables (disjoint by construction)
  text.py       normal forms, pausal-form rules
  distance.py   Levenshtein, alignment, error tallies
  metrics.py    multi-view scoring, grouping, confusions
  splits.py     stratified and speaker-holdout splits
  manifest.py   JSONL/JSON/CSV/TSV I/O, corpus audit
  surahs.py     114-surah table, reference validation
  report.py     standalone HTML reports
  demo.py       synthetic corpus and error model
  cli.py        the `warsh` command
```

## Licence

MIT.
