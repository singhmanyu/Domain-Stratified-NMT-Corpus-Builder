## What this changes

<!-- One or two sentences. If it fixes an issue, say "Fixes #N". -->

## Why

<!-- What was wrong, or what wasn't possible before. -->

## What you ran it on

<!--
Corpus-scale behaviour can't be inferred from sample_data/. Say which it was
and how many rows, so a reviewer knows what the result is evidence of.
-->

- [ ] `sample_data/`
- [ ] My own corpus (roughly how many rows: ...)
- [ ] Didn't run it — explain below

```
paste the commands and output here
```

## Quality impact

<!--
Skip if this is a refactor, docs, or tooling change. If it touches routing,
seeding, hyperparameters, or the taxonomy, give before/after numbers and say
which split they came from.
-->

## Checklist

- [ ] Constants went in `config.py`, not inline
- [ ] Nothing new assumes a specific target language (use `cfg.TARGET_COL`)
- [ ] Nothing new hardcodes domain names (derive from `cfg.TAXONOMY`)
- [ ] New CSV writes use `cfg.CSV_ENCODING` (utf-8-sig — non-Latin scripts in Excel)
- [ ] `predict.Pipeline.classify()` still returns one row per input, in order
- [ ] Comments explain *why*, not what
- [ ] No new dependency, or there's a reason for it above
