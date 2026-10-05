"""Writes the domain splits as xlsx workbooks - data on sheet 1, run metadata
on sheet 2 - chopped into parts small enough to open and re-save in Excel
without trouble.

    python export_xlsx.py                  # summary + every domain, 100k rows per file
    python export_xlsx.py --rows 50000      # smaller parts
    python export_xlsx.py --summary          # just the summary workbook
    python export_xlsx.py --domain climate   # one domain only

The csv files in domain_splits/ stay the canonical complete copy - the xlsx
parts are for reading and editing by hand.
"""

import argparse
import glob
import os
from datetime import datetime

import pandas as pd

import config as cfg
import finegrain_stage as fg
import nemo_stage as ns

CLASSIFIED = cfg.OUTPUT_FILE
SPLIT_DIR = cfg.SPLIT_DIR
OUT_DIR = cfg.XLSX_DIR
# excel's hard cap is 1,048,576 rows, but a file that big takes minutes to
# open and is easy to mangle on save. 100k opens instantly.
PART_ROWS = cfg.XLSX_PART_ROWS

COLUMN_NOTES = [
    ("id", "row id from the source workbook"),
    ("english", "english side - the ONLY column the classifier reads"),
    ("nepali", "nepali side - carried through untouched, never classified"),
    ("source_file", "which of the 6 source workbooks this pair came from"),
    ("nemo_label", "raw label from nvidia/domain-classifier (26-class taxonomy)"),
    ("nemo_score", "nvidia/domain-classifier softmax confidence"),
    ("route", "general = nemo label had no overlap, assigned General directly; "
              "direct = nemo label mapped 1:1; finegrain = fasttext decided"),
    ("domain", "final domain, one of the 9"),
    ("confidence", "fasttext confidence - only set on finegrain rows"),
]


def build_metadata(df):
    """run metadata as a flat key/value frame, for the second sheet."""
    rows = []

    def add(section, key, value):
        rows.append({"section": section, "item": key, "value": value})

    add("run", "generated", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    add("run", "total pairs in corpus", f"{len(df):,}")

    add("source", "workbooks merged", len(glob.glob(os.path.join(cfg.RAW_DIR, "*.xlsx"))))
    add("source", "exact duplicate pairs dropped in merge", "38,163")
    if "source_file" in df.columns:
        for k, v in df["source_file"].value_counts().items():
            add("rows per workbook", str(k), f"{v:,}")

    add("stage 1", "model", cfg.MODEL_NAME)
    add("stage 1", "confidence threshold", cfg.CONF_THRESHOLD)
    add("stage 1", "max tokens", cfg.MAX_LENGTH)
    add("stage 1", "batch size", cfg.BATCH_SIZE)
    if "nemo_score" in df.columns:
        add("stage 1", "mean confidence", round(df["nemo_score"].mean(), 4))

    add("stage 2", "seeding model", cfg.OLLAMA_MODEL)
    add("stage 2", "classifier",
        "fasttext supervised, wordNgrams=2, dim=50, epoch=10, lr=0.5")
    if os.path.exists(cfg.SEED_FILE):
        c = fg.seed_distribution()
        add("stage 2", "total seeds", f"{sum(c.values()):,}")
        for k, v in c.most_common():
            add("seeds per domain", k, f"{v:,}")

    if "route" in df.columns:
        for k, v in df["route"].value_counts().items():
            add("routing", k, f"{v:,}  ({v / len(df) * 100:.1f}%)")

    for k, v in df["domain"].value_counts().items():
        add("domain distribution", str(k), f"{v:,}  ({v / len(df) * 100:.2f}%)")

    for col, note in COLUMN_NOTES:
        if col in df.columns:
            add("columns", col, note)

    add("caveats", "evaluation",
        "macro-F1 not measured yet - needs the hand-labeled gold set "
        "(gold_test_labeled.csv)")
    add("caveats", "Climate",
        "0.14% of the corpus. below the 0.5% threshold where the original plan "
        "says to merge a class into General or its nearest neighbour.")
    add("caveats", "Tech",
        "15.9% looks high. the ollama seeds were 28.7% Tech, so Tech is likely "
        "over-assigned - confirm against the gold set.")
    add("caveats", "seed labels",
        "llama3.2:3b weak supervision, not human labels. the gold set is the "
        "only human-labeled data and never touches training.")
    add("caveats", "this file",
        "the csv in domain_splits/ is the complete canonical copy. these xlsx "
        "parts are for reading/editing by hand.")

    return pd.DataFrame(rows)


def write_book(path, data, meta, data_sheet="data", part_note=None):
    if part_note:
        meta = pd.concat([pd.DataFrame(part_note), meta], ignore_index=True)

    with pd.ExcelWriter(path, engine="xlsxwriter") as xl:
        data.to_excel(xl, sheet_name=data_sheet, index=False)
        meta.to_excel(xl, sheet_name="metadata", index=False)

        book = xl.book
        wrap = book.add_format({"text_wrap": True, "valign": "top"})
        bold = book.add_format({"bold": True})

        ws = xl.sheets[data_sheet]
        ws.freeze_panes(1, 0)
        for i, col in enumerate(data.columns):
            ws.set_column(i, i, 46 if col in ("english", "nepali") else 16)

        ms = xl.sheets["metadata"]
        ms.freeze_panes(1, 0)
        ms.set_column(0, 0, 22, bold)
        ms.set_column(1, 1, 36)
        ms.set_column(2, 2, 72, wrap)

    size = os.path.getsize(path) / 1e6
    rel = os.path.relpath(path, cfg.DATA_DIR)
    print(f"  {rel}  {len(data):,} rows  {size:.1f} MB")


def write_domain(domain, group, meta, rows_per_part=PART_ROWS):
    """one workbook per part, or a single workbook if it fits in one."""
    name = str(domain).lower()
    group = group.reset_index(drop=True)
    n_parts = -(-len(group) // rows_per_part)

    for p in range(n_parts):
        lo = p * rows_per_part
        hi = min(lo + rows_per_part, len(group))
        chunk = group.iloc[lo:hi]

        if n_parts == 1:
            path = os.path.join(OUT_DIR, f"{name}.xlsx")
            note = [{"section": "this part", "item": "contents",
                     "value": f"all {len(group):,} {domain} pairs"}]
        else:
            path = os.path.join(OUT_DIR, f"{name}_part{p + 1:02d}.xlsx")
            note = [
                {"section": "this part", "item": "part",
                 "value": f"{p + 1} of {n_parts}"},
                {"section": "this part", "item": "rows",
                 "value": f"{lo + 1:,} - {hi:,} of {len(group):,} {domain} pairs"},
                {"section": "this part", "item": "other parts",
                 "value": f"{name}_part01..part{n_parts:02d}.xlsx"},
            ]

        write_book(path, chunk, meta, part_note=note)


def write_corpus_chunks(df, meta, rows_per_part=PART_ROWS, out_dir=None):
    """the whole corpus in sequential xlsx chunks, in original row order,
    rather than grouped by domain. every row has its final domain."""
    out_dir = out_dir or os.path.join(OUT_DIR, "chunks")
    os.makedirs(out_dir, exist_ok=True)

    n_parts = -(-len(df) // rows_per_part)
    print(f"writing {n_parts} corpus chunks to {out_dir}")

    for p in range(n_parts):
        lo = p * rows_per_part
        hi = min(lo + rows_per_part, len(df))
        chunk = df.iloc[lo:hi]

        note = [
            {"section": "this chunk", "item": "chunk",
             "value": f"{p + 1} of {n_parts}"},
            {"section": "this chunk", "item": "rows",
             "value": f"{lo + 1:,} - {hi:,} of {len(df):,}"},
            {"section": "this chunk", "item": "contents",
             "value": "all domains, original corpus order. "
                      "for one domain at a time use the per-domain workbooks."},
            {"section": "this chunk", "item": "domains here",
             "value": ", ".join(f"{k} {v:,}" for k, v in
                                chunk["domain"].value_counts().items())},
        ]
        write_book(os.path.join(out_dir, f"corpus_chunk{p + 1:02d}.xlsx"),
                   chunk, meta, part_note=note)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=PART_ROWS,
                    help=f"rows per xlsx part (default {PART_ROWS:,})")
    ap.add_argument("--summary", action="store_true", help="summary workbook only")
    ap.add_argument("--domain", help="export just this domain")
    ap.add_argument("--chunks", action="store_true",
                    help="whole corpus in sequential chunks instead of per-domain books")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"reading {CLASSIFIED}")
    df = pd.read_csv(CLASSIFIED, encoding=cfg.CSV_ENCODING)
    meta = build_metadata(df)

    summary = (df["domain"].value_counts().rename_axis("domain")
               .reset_index(name="pairs"))
    summary["share %"] = (summary["pairs"] / len(df) * 100).round(2)
    summary["csv"] = summary["domain"].str.lower().apply(lambda d: os.path.join(SPLIT_DIR, f"{d}.csv"))
    summary["xlsx parts"] = (summary["pairs"] // args.rows + 1)

    print("writing workbooks")
    write_book(os.path.join(OUT_DIR, "domain_summary.xlsx"), summary, meta,
               data_sheet="summary")
    if args.summary:
        return

    if args.chunks:
        write_corpus_chunks(df, meta, rows_per_part=args.rows)
        return

    for domain, group in df.groupby("domain"):
        if args.domain and str(domain).lower() != args.domain.lower():
            continue
        write_domain(domain, group, meta, rows_per_part=args.rows)


if __name__ == "__main__":
    main()
