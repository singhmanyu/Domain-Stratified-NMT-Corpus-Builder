"""Reads a parallel-corpus file without being told how it's laid out.

The pipeline classifies the English side and carries the other side along
untouched, so it works for English paired with any language. This module is
what makes that true in practice: it works out which column is which rather
than requiring a fixed layout.

  1. header names, matched case-insensitively against the usual aliases -
     english/nepali, source/target, src/tgt, en/hi, SOURCE/TRANSLATE ...
  2. failing that, script: a column written in a non-Latin script is the
     target side, whatever its header claims. Settles Devanagari, Arabic,
     CJK, Cyrillic, Thai, Tamil, Bangla and the rest in one rule.
  3. failing that - a Latin-script target like French or Vietnamese, where
     script tells you nothing - which column reads as English, scored on
     function words that don't carry over into other Latin-script languages.
  4. failing that, column order: [id,] english, target

Detection is reported, never silent. If it can't tell, it says so and asks
for the column explicitly instead of guessing and corrupting the pairing.

    python parallel_input.py corpus.xlsx                 # inspect
    python parallel_input.py corpus.xlsx --build         # -> cfg.INPUT_FILE
    python parallel_input.py data/*.xlsx --build         # several at once
"""

import glob
import os
import re

import pandas as pd

import config as cfg

# header aliases, lowercased and stripped of punctuation before matching
ID_NAMES = {"id", "sn", "sno", "serial", "serialno", "index", "idx", "no",
            "num", "number", "rowid", "row", "sr", "srno", "key", "uid"}

ENGLISH_NAMES = {"english", "en", "eng", "source", "src", "sourcetext",
                 "englishtext", "englishsentence", "sentence", "text",
                 "sourcesentence", "input", "l1"}

# the generic half - role words rather than language names
TARGET_NAMES = {"target", "tgt", "translation", "translate", "translated",
                "targettext", "targetsentence", "output", "l2", "trg"}

# ...plus whatever the other language happens to be called. Not exhaustive
# and doesn't need to be - script and function-word detection cover the rest.
LANGUAGE_NAMES = {
    "nepali", "nep", "ne", "hindi", "hi", "bengali", "bangla", "bn", "urdu",
    "ur", "tamil", "ta", "telugu", "te", "marathi", "mr", "gujarati", "gu",
    "kannada", "kn", "malayalam", "ml", "punjabi", "pa", "odia", "oriya",
    "assamese", "as", "sinhala", "si", "maithili", "bhojpuri", "newari",
    "arabic", "ar", "persian", "farsi", "fa", "hebrew", "he", "chinese",
    "zh", "mandarin", "japanese", "ja", "korean", "ko", "thai", "th",
    "vietnamese", "vi", "russian", "ru", "ukrainian", "uk", "greek", "el",
    "amharic", "am", "swahili", "sw", "french", "fr", "spanish", "es",
    "portuguese", "pt", "german", "de", "italian", "it", "turkish", "tr",
    "indonesian", "id2", "malay", "ms", "tagalog", "filipino", "burmese",
    "my", "khmer", "km", "lao", "lo", "tibetan", "dzongkha", "devanagari",
}
TARGET_ALL = TARGET_NAMES | LANGUAGE_NAMES

# Latin letters, digits and common punctuation. Anything outside this is
# another script - that is the whole test, and it is script-agnostic by
# construction rather than by listing scripts.
LATIN_ISH = re.compile(r"[A-Za-zÀ-ɏ]")
NON_LATIN = re.compile(r"[^\x00-\x7F -ɏ]")

# English function words that are not also common words in other
# Latin-script languages. "a", "in", "de", "no", "me" are deliberately absent.
ENGLISH_STOPWORDS = {
    "the", "and", "of", "to", "is", "was", "are", "were", "that", "this",
    "with", "for", "from", "have", "has", "had", "been", "will", "would",
    "which", "their", "there", "they", "what", "when", "where", "about",
    "been", "into", "than", "then", "these", "those", "them", "being",
    "should", "could", "after", "before", "through", "during", "between",
}
WORD = re.compile(r"[a-z']+")

# a column more than half non-Latin is the target side; under a tenth is the
# English side. real data sits near 0 or near 1 - anything in between means
# the columns are mixed and worth refusing to guess on.
NON_LATIN_HI = 0.5
NON_LATIN_LO = 0.1
# how much more English-looking one column must be than the other before
# that counts as evidence rather than noise
ENGLISH_MARGIN = 0.25


class DetectionError(Exception):
    """Raised when the layout is genuinely ambiguous. Better than a wrong
    guess - a mispaired corpus trains a broken model and looks fine."""


def _norm(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _sample(series, n=500):
    values = series.dropna().astype(str).str.strip()
    return values[values != ""].head(n)


def non_latin_ratio(series, n=500):
    """Share of values containing characters outside the Latin range."""
    values = _sample(series, n)
    if values.empty:
        return 0.0
    return float(values.apply(lambda s: bool(NON_LATIN.search(s))).mean())


def english_score(series, n=500):
    """Share of values containing at least one distinctly English function
    word. English prose scores ~0.9; other Latin-script languages score low
    because the words chosen don't exist in them."""
    values = _sample(series, n)
    if values.empty:
        return 0.0
    hits = values.apply(
        lambda s: bool(ENGLISH_STOPWORDS.intersection(WORD.findall(s.lower())))
    )
    return float(hits.mean())


def _looks_like_id(series, n=500):
    """Numeric, or short and unique. Anything prose-length isn't an id."""
    if pd.api.types.is_numeric_dtype(series):
        return True
    values = _sample(series, n)
    if values.empty:
        return False
    return values.str.len().max() <= 32 and values.is_unique


def _looks_headerless(df):
    """pandas names unheadered columns Unnamed: 0, 1, 2 - and a file whose
    'header' is really its first data row usually has a long string there,
    or one in the target's script."""
    names = [str(c) for c in df.columns]
    if all(n.startswith("Unnamed:") for n in names):
        return True
    return any(len(n) > 60 or NON_LATIN.search(n) for n in names)


def detect_columns(df):
    """-> {"id": name|None, "english": name, "target": name}

    Raises DetectionError if it can't tell the two language columns apart.
    """
    by_norm = {_norm(c): c for c in df.columns}

    id_col = next((by_norm[n] for n in by_norm if n in ID_NAMES), None)
    english = next((by_norm[n] for n in by_norm if n in ENGLISH_NAMES), None)
    target = next((by_norm[n] for n in by_norm if n in TARGET_ALL), None)

    # headers settled it
    if english and target and english != target:
        return _with_inferred_id(df, id_col, english, target)

    # otherwise ask the text itself. only consider columns that hold text and
    # aren't already spoken for - tested by what the dtype is NOT, because
    # pandas 3 hands back StringDtype where pandas 2 gave plain object, and
    # checking `== object` silently matches nothing on the newer one.
    candidates = [c for c in df.columns
                  if c != id_col
                  and not pd.api.types.is_numeric_dtype(df[c])
                  and not pd.api.types.is_datetime64_any_dtype(df[c])
                  and not pd.api.types.is_bool_dtype(df[c])]
    scripts = {c: non_latin_ratio(df[c]) for c in candidates}

    # step 2: a non-Latin column is the target side, whatever it's called
    if target and not english:
        english = next((c for c in candidates
                        if c != target and scripts[c] < NON_LATIN_LO), None)
    elif english and not target:
        target = next((c for c in candidates
                       if c != english and scripts[c] > NON_LATIN_HI), None)
    else:
        foreign = [c for c in candidates if scripts[c] > NON_LATIN_HI]
        latin = [c for c in candidates if scripts[c] < NON_LATIN_LO]
        if len(foreign) == 1 and len(latin) >= 1:
            target, english = foreign[0], latin[0]

    # step 3: both sides Latin - French, Vietnamese, Indonesian and so on.
    # Script says nothing, so score how English each column reads.
    if not (english and target) and len(candidates) == 2:
        a, b = candidates
        scores = {a: english_score(df[a]), b: english_score(df[b])}
        if abs(scores[a] - scores[b]) >= ENGLISH_MARGIN:
            english = max(scores, key=scores.get)
            target = min(scores, key=scores.get)

    if english and target and english != target:
        return _with_inferred_id(df, id_col, english, target)

    # Last resort: position - but only with some evidence that this is a
    # parallel file at all. If no header matched and nothing else spoke up,
    # falling back to column order would be a coin flip dressed up as a
    # decision, so refuse instead.
    header_hint = any(_norm(c) in ENGLISH_NAMES | TARGET_ALL for c in df.columns)
    if any(r > NON_LATIN_HI for r in scripts.values()) or header_hint:
        rest = [c for c in df.columns if c != id_col]
        if id_col is None and len(df.columns) == 3:
            return {"id": df.columns[0], "english": df.columns[1],
                    "target": df.columns[2]}
        if len(rest) == 2:
            return {"id": id_col, "english": rest[0], "target": rest[1]}

    raise DetectionError(
        f"can't tell which column is English and which is the translation.\n"
        f"  columns: {', '.join(map(str, df.columns))}\n"
        f"  non-Latin share: "
        f"{', '.join(f'{c}={scripts.get(c, 0):.0%}' for c in candidates)}\n"
        f"  rename the headers to english/target, or pass the column explicitly"
    )


def _with_inferred_id(df, id_col, english, target):
    """A headerless file has no column called 'id', but if exactly one column
    is left over and it isn't prose, that's what it is."""
    if id_col is None:
        leftover = [c for c in df.columns if c not in (english, target)]
        if len(leftover) == 1 and _looks_like_id(df[leftover[0]]):
            id_col = leftover[0]
    return {"id": id_col, "english": english, "target": target}


def read_parallel(path, quiet=False):
    """One file -> a frame with columns id (if present), english, <target>."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".xlsx", ".xlsm", ".xls"):
        df = pd.read_excel(path)
        if _looks_headerless(df):
            df = pd.read_excel(path, header=None)
    else:
        df = pd.read_csv(path, encoding=cfg.CSV_ENCODING)
        if _looks_headerless(df):
            df = pd.read_csv(path, encoding=cfg.CSV_ENCODING, header=None)

    if df.empty:
        raise DetectionError(f"{os.path.basename(path)} has no rows")

    found = detect_columns(df)
    if not quiet:
        bits = [f"english={found['english']}",
                f"{cfg.TARGET_COL}={found['target']}",
                f"id={found['id']}" if found["id"] is not None else "id=none"]
        print(f"  {os.path.basename(path)}: {len(df):,} rows  ({', '.join(bits)})")

    cols = {found["english"]: cfg.TEXT_COL, found["target"]: cfg.TARGET_COL}
    if found["id"] is not None:
        cols[found["id"]] = "id"
    out = df[list(cols)].rename(columns=cols)

    ordered = (["id"] if "id" in out.columns else []) + [cfg.TEXT_COL, cfg.TARGET_COL]
    return out[ordered]


def clean(df):
    """Drop rows that aren't a usable pair, and exact duplicates."""
    for col in (cfg.TEXT_COL, cfg.TARGET_COL):
        # drop the nulls before stringifying, not after - what str() makes of
        # a missing value is a moving target ("nan" on pandas 2, "<NA>" on
        # pandas 3), and matching against those spellings is how blank rows
        # survive into a corpus
        df = df[df[col].notna()]
        df[col] = df[col].astype(str).str.strip()
        df = df[df[col] != ""]
    return df.drop_duplicates(subset=[cfg.TEXT_COL, cfg.TARGET_COL])


def build_corpus(paths, out_file=None, quiet=False):
    """Read every path, normalise, concatenate, write one csv."""
    out_file = out_file or cfg.INPUT_FILE
    frames = []
    for path in paths:
        df = read_parallel(path, quiet=quiet)
        df["source_file"] = os.path.basename(path)
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    before = len(merged)
    merged = clean(merged)
    if not quiet:
        print(f"\n{before:,} rows read, {before - len(merged):,} dropped "
              f"(blank or duplicate)")
        print(f"{len(merged):,} pairs")

    merged.to_csv(out_file, index=False, encoding=cfg.CSV_ENCODING)
    if not quiet:
        print(f"saved {out_file}")
    return merged


def expand(patterns):
    """Globs, directories and plain paths all end up as a list of files."""
    out = []
    for pattern in patterns:
        if os.path.isdir(pattern):
            for ext in ("*.xlsx", "*.xlsm", "*.xls", "*.csv"):
                out.extend(sorted(glob.glob(os.path.join(pattern, ext))))
        else:
            out.extend(sorted(glob.glob(pattern)) or [pattern])
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", default=[cfg.RAW_DIR],
                    help="xlsx/csv files, globs, or a directory "
                         f"(default: {cfg.RAW_DIR})")
    ap.add_argument("--build", action="store_true",
                    help=f"write the combined corpus to {cfg.INPUT_FILE}")
    ap.add_argument("--out", default=None, help="write somewhere else")
    args = ap.parse_args()

    files = expand(args.paths or [cfg.RAW_DIR])
    if not files:
        raise SystemExit(f"nothing to read in {', '.join(args.paths)}")

    print(f"{len(files)} file(s)")
    if args.build:
        build_corpus(files, args.out)
    else:
        for f in files:
            try:
                read_parallel(f)
            except DetectionError as e:
                print(f"  {os.path.basename(f)}: {e}")
        print("\nnothing written - pass --build to write the corpus")
