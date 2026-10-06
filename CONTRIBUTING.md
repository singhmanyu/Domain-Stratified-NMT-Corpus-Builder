# Contributing

Issues and pull requests are welcome. This is a research pipeline rather than
a library, so the bar is "does it still produce defensible labels", not API
stability.

## Before you start

The hard part of working on this is that **you probably don't have the
corpus.** `sample_data/` exists for that reason — it's enough to exercise
every code path, not enough to reproduce the numbers in the README. If a
change depends on corpus-scale behaviour, say so in the PR and describe what
you ran it on.

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # linux: source venv/bin/activate
pip install -r requirements.txt

set NLTM_DATA_DIR=C:\path\to\your\data    # linux: export NLTM_DATA_DIR=...
python domain_classifier_nltm.py status
```

Stage 1 wants a GPU — a 6GB card does the full corpus in well under an hour.
CPU works and is roughly 10x slower; the code picks the device itself and
nothing needs changing.

Stage 2 seeding needs an ollama server. `start_ollama.cmd` launches one, and
`NLTM_OLLAMA_HOST` overrides the port if the default is taken.

## Things that will be asked in review

These are all documented in the README's *Technique notes*, and every one of
them has already cost someone a day:

- **Don't quantize stage 1.** int8 dynamic quantization is ~1.4x faster on CPU
  and destroys this model — mean confidence 0.93 → 0.31, with ~90% of rows
  collapsing into one class.
- **Don't swap `CustomModel` for `AutoModelForSequenceClassification`.** It
  loads without error and gives you random weights.
- **Keep seeding temperature at 0.** At ollama's default of 0.8 the labeller
  disagreed with itself on 43% of repeated sentences. That is pure label noise
  and it was worth ~0.15 macro-F1.
- **Predict in bulk.** `model.predict(list_of_sentences)` — looping one
  sentence at a time turns seconds into hours.
- **Write CSVs as `utf-8-sig`.** Use `cfg.CSV_ENCODING`. The Nepali side is
  Devanagari and these files get opened in Excel on Windows, which assumes the
  ANSI codepage unless there's a BOM.
- **Don't change row counts or row order in `predict.Pipeline.classify()`.**
  Callers join the result onto the Nepali side positionally. A dropped row
  silently mispairs every row after it. If you need to exclude something, mark
  it and pass it through — that's what the `skipped (empty)` stage is for.

## Style

Match what's there. Comments explain *why*, especially where the obvious
approach is the wrong one; they don't narrate what the line does. Constants
live in `config.py`, not inline. No new dependency without a reason in the PR
description.

## Tests

There's no test suite yet, which is the most useful thing someone could add.
Until then, state in the PR what you ran and what you saw:

```bash
python predict.py "The ward office issues birth certificates."
python domain_classifier_nltm.py probe
python domain_classifier_nltm.py evaluate      # needs a labelled gold set
```

## Reporting results

If you report a quality number, say which split it came from and how many rows
it covers. The README is explicit about what its own numbers are and are not,
and that should stay true.
