"""Merges the source xlsx workbooks into one csv.

The real source files don't agree on column names - ID/src/tgt,
ID/SOURCE/TARGET, ID/SOURCE/TRANSLATE, and one with no header row at all
(Unnamed: 0/1/2). The column ORDER is always id, english, nepali though, so
we read by position and ignore the headers entirely.

    python merge_data.py
"""

import glob
import os

import pandas as pd

import config as cfg


def load_one(path):
    df = pd.read_excel(path)
    df = df.iloc[:, :3]
    df.columns = ["id", "english", "nepali"]
    df["source_file"] = os.path.basename(path)
    return df


def main():
    files = sorted(glob.glob(os.path.join(cfg.RAW_DIR, "*.xlsx")))
    if not files:
        print(f"no xlsx files in {cfg.RAW_DIR}")
        return
    print(f"found {len(files)} workbooks")

    frames = []
    for f in files:
        df = load_one(f)
        print(f"  {os.path.basename(f)}: {len(df):,} rows")
        frames.append(df)

    merged = pd.concat(frames, ignore_index=True)
    print(f"\ntotal before cleanup: {len(merged):,}")

    merged["english"] = merged["english"].astype(str).str.strip()
    merged["nepali"] = merged["nepali"].astype(str).str.strip()
    merged = merged[(merged["english"] != "") & (merged["english"] != "nan")]
    merged = merged[(merged["nepali"] != "") & (merged["nepali"] != "nan")]

    before = len(merged)
    merged = merged.drop_duplicates(subset=["english", "nepali"])
    print(f"dropped {before - len(merged):,} exact duplicate pairs")
    print(f"final: {len(merged):,} pairs")

    merged.to_csv(cfg.INPUT_FILE, index=False, encoding=cfg.CSV_ENCODING)
    print(f"saved {os.path.basename(cfg.INPUT_FILE)}")


if __name__ == "__main__":
    main()
