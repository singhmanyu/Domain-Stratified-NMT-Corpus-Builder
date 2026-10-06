"""Reads a parallel-corpus file without being told how it's laid out.

Everyone's export looks different - english/nepali, source/target, src/tgt,
SOURCE/TRANSLATE, sometimes an id column, sometimes no header row at all. So
rather than hardcoding one shape, this works out which column is which:

  1. header names, matched case-insensitively against the usual aliases
  2. failing that, which column actually contains Devanagari - that settles
     the english/nepali question on its own, whatever the header says
  3. failing that, column order: [id,] english, nepali

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
NEPALI_NAMES = {"nepali", "ne", "nep", "target", "tgt", "translation",
                "translate", "translated", "targettext", "nepalitext",
                "nepalisentence", "devanagari", "output", "l2"}

DEVANAGARI = re.compile(r"[ऀ-ॿ]")
# a column that is mostly devanagari is the nepali side; one that is barely
# any is the english side. real data sits near 0 or near 1, so anything in
# between means the columns are mixed and worth refusing to guess on.
DEVANAGARI_HI = 0.5
DEVANAGARI_LO = 0.1


class DetectionError(Exception):
    """Raised when the layout is genuinely ambiguous. Better than a wrong
    guess - a mispaired corpus trains a broken model and looks fine."""


def _norm(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def devanagari_ratio(series, sample=500):
    """Share of non-empty values containing any Devanagari."""
    values = series.dropna().astype(str).str.strip()
    values = values[values != ""].head(sample)
    if values.empty:
        return 0.0
    return float(values.apply(lambda s: bool(DEVANAGARI.search(s))).mean())


def _looks_like_id(series, sample=500):
    """Numeric, or short and unique. Anything prose-length isn't an id."""
    if pd.api.types.is_numeric_dtype(series):
        return True
    values = series.dropna().astype(str).str.strip().head(sample)
    if values.empty:
        return False
    return values.str.len().max() <= 32 and values.is_unique


def _looks_headerless(df):
    """pandas names unheadered columns Unnamed: 0, 1, 2 - and a file whose
    'header' is really its first data row usually has a long string there."""
    names = [str(c) for c in df.columns]
    if all(n.startswith("Unnamed:") for n in names):
        return True
    return any(len(n) > 60 or DEVANAGARI.search(n) for n in names)


def detect_columns(df):
    """-> {"id": name|None, "english": name, "nepali": name}

    Raises DetectionError if it can't tell the two language columns apart.
    """
    by_norm = {_norm(c): c for c in df.columns}

    id_col = next((by_norm[n] for n in by_norm if n in ID_NAMES), None)
    english = next((by_norm[n] for n in by_norm if n in ENGLISH_NAMES), None)
    nepali = next((by_norm[n] for n in by_norm if n in NEPALI_NAMES), None)

    # headers settled it
    if english and nepali and english != nepali:
        return {"id": id_col, "english": english, "nepali": nepali}

    # otherwise ask the text itself. only consider columns that hold text and
    # aren't already spoken for - tested by what the dtype is NOT, because
    # pandas 3 hands back StringDtype where pandas 2 gave plain object, and
    # checking `== object` silently matches nothing on the newer one.
    candidates = [c for c in df.columns
                  if c != id_col
                  and not pd.api.types.is_numeric_dtype(df[c])
                  and not pd.api.types.is_datetime64_any_dtype(df[c])
                  and not pd.api.types.is_bool_dtype(df[c])]
    ratios = {c: devanagari_ratio(df[c]) for c in candidates}

    if nepali and not english:
        english = next((c for c in candidates
                        if c != nepali and ratios[c] < DEVANAGARI_LO), None)
    elif english and not nepali:
        nepali = next((c for c in candidates
                       if c != english and ratios[c] > DEVANAGARI_HI), None)
    else:
        devanagari = [c for c in candidates if ratios[c] > DEVANAGARI_HI]
        latin = [c for c in candidates if ratios[c] < DEVANAGARI_LO]
        if len(devanagari) == 1 and len(latin) >= 1:
            nepali = devanagari[0]
            english = latin[0]

    if english and nepali and english != nepali:
        # a headerless file has no column called "id", but if exactly one
        # column is left over and it isn't prose, that's what it is
        if id_col is None:
            leftover = [c for c in df.columns if c not in (english, nepali)]
            if len(leftover) == 1 and _looks_like_id(df[leftover[0]]):
                id_col = leftover[0]
        return {"id": id_col, "english": english, "nepali": nepali}

    # Last resort: position - but only with some evidence that this is a
    # parallel file at all. If no header matched AND no column holds
    # Devanagari, falling back to column order would just be a coin flip
    # dressed up as a decision, so refuse instead.
    has_devanagari = any(r > DEVANAGARI_HI for r in ratios.values())
    header_hint = any(_norm(c) in ENGLISH_NAMES | NEPALI_NAMES for c in df.columns)
    if has_devanagari or header_hint:
        rest = [c for c in df.columns if c != id_col]
        if id_col is None and len(df.columns) == 3:
            return {"id": df.columns[0], "english": df.columns[1],
                    "nepali": df.columns[2]}
        if len(rest) == 2:
            return {"id": id_col, "english": rest[0], "nepali": rest[1]}

    raise DetectionError(
        f"can't tell which columns hold english and nepali.\n"
        f"  columns: {', '.join(map(str, df.columns))}\n"
        f"  devanagari share: "
        f"{', '.join(f'{c}={ratios.get(c, 0):.0%}' for c in candidates)}\n"
        f"  pass them explicitly, or rename the headers to english/nepali"
    )


def read_parallel(path, quiet=False):
    """One file -> a frame with columns id (if present), english, nepali."""
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
        bits = [f"english={found['english']}", f"nepali={found['nepali']}"]
        bits.append(f"id={found['id']}" if found["id"] is not None else "id=none")
        print(f"  {os.path.basename(path)}: {len(df):,} rows  ({', '.join(bits)})")

    cols = {found["english"]: "english", found["nepali"]: "nepali"}
    if found["id"] is not None:
        cols[found["id"]] = "id"
    out = df[list(cols)].rename(columns=cols)

    # id is optional everywhere downstream, but if the input had one we keep
    # it so rows can be traced back to the source file
    ordered = (["id"] if "id" in out.columns else []) + ["english", "nepali"]
    return out[ordered]


def clean(df):
    """Drop rows that aren't a usable pair, and exact duplicates."""
    for col in ("english", "nepali"):
        # drop the nulls before stringifying, not after - what str() makes of
        # a missing value is a moving target ("nan" on pandas 2, "<NA>" on
        # pandas 3), and matching against those spellings is how blank rows
        # survive into a corpus
        df = df[df[col].notna()]
        df[col] = df[col].astype(str).str.strip()
        df = df[df[col] != ""]
    return df.drop_duplicates(subset=["english", "nepali"])


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
