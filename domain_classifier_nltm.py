"""Orchestrator. Every step is resumable and takes a time budget.

    python domain_classifier_nltm.py status                 # progress, no gpu
    python domain_classifier_nltm.py probe                  # keyword sanity check
    python domain_classifier_nltm.py stage1 --budget 30     # nemo coarse split
    python domain_classifier_nltm.py seed                   # random seeding
    python domain_classifier_nltm.py seed-targeted          # per-bucket seeding
    python domain_classifier_nltm.py finish                 # stage 2 + splits
    python domain_classifier_nltm.py evaluate               # vs the gold set
    python domain_classifier_nltm.py all --budget 60

`finish` streams the stage-1 chunks instead of loading all 1.7M rows at once,
so memory stays flat regardless of corpus size.
"""

import argparse
import os

import pandas as pd

import config as cfg
import nemo_stage as ns
import finegrain_stage as fg

DOMAINS = cfg.DOMAINS

PROBES = {
    "Agriculture": ["farm", "crop", "irrigat", "livestock", "harvest", "soil"],
    "Climate": ["climate", "monsoon", "rainfall", "glacier", "emission"],
    "Tourism": ["tourist", "trek", "hotel", "heritage", "travel"],
    "Admin": ["committee", "ministry", "notice", "department", "applicant"],
    "Law": ["court", "act ", "clause", "petition", "tribunal"],
    "Health": ["patient", "disease", "vaccine", "hospital", "symptom"],
    "Education": ["student", "curriculum", "school", "examination", "teacher"],
    "Tech": ["software", "computer", "internet", "digital", "network"],
}


def _have_corpus():
    """every stage needs the corpus csv - fail with a hint, not a traceback."""
    if os.path.exists(cfg.INPUT_FILE):
        return True
    print(f"no corpus at {cfg.INPUT_FILE}")
    print("build it from your parallel data first:")
    print("    python parallel_input.py <your.xlsx> --build")
    if not os.path.isdir(cfg.RAW_DIR):
        print(f"...or put the files in {cfg.RAW_DIR} and run it with no "
              f"arguments. No data at all? sample_data/make_sample.py")
    return False


def cmd_probe():
    """Crude keyword counts, before trusting any model. If a domain comes
    back under ~0.5% it's worth asking whether it should be its own class."""
    if not _have_corpus():
        return
    df = pd.read_csv(cfg.INPUT_FILE, usecols=[cfg.TEXT_COL], encoding=cfg.CSV_ENCODING)
    low = df[cfg.TEXT_COL].fillna("").str.lower()
    print(f"\nkeyword probe over {len(df):,} rows:")
    for dom, kws in PROBES.items():
        hits = low.str.contains("|".join(kws), regex=True).sum()
        print(f"  {dom:<13} ~{hits:>8,}  ({hits / len(df) * 100:.2f}%)")


def cmd_status():
    done, total, done_rows, total_rows = ns.stage1_progress()
    pct = done / max(total, 1) * 100
    print(f"corpus:  {total_rows:,} rows" if total_rows else "corpus:  not merged yet")
    print(f"stage 1: {done}/{total} chunks ({pct:.1f}%), {done_rows:,} rows classified")

    c = fg.seed_distribution()
    if c:
        print(f"seeds:   {sum(c.values()):,}")
        for k, v in c.most_common():
            print(f"           {k:<12} {v:>6}")
    else:
        print("seeds:   none yet")

    print(f"model:   {'trained' if os.path.exists(cfg.FASTTEXT_MODEL) else 'not trained'}")
    print(f"gold:    {'labelled' if os.path.exists(cfg.GOLD_LABELED) else 'NOT LABELLED - needed for f1'}")
    print(f"output:  {'written' if os.path.exists(cfg.OUTPUT_FILE) else 'not written'}")


def cmd_stage1(budget):
    if not _have_corpus():
        return
    ns.run_nemo_chunked(max_minutes=budget)
    done, total, rows, _ = ns.stage1_progress()
    print(f"stage 1 now {done}/{total} chunks, {rows:,} rows")


def cmd_seed(budget):
    if not _have_corpus():
        return
    df = pd.read_csv(cfg.INPUT_FILE, usecols=[cfg.TEXT_COL], encoding=cfg.CSV_ENCODING)
    fg.seed_labels(df, max_minutes=budget)
    _show_seeds()


def cmd_seed_targeted(budget):
    if not ns.chunk_paths():
        print("no stage1 chunks - run stage1 first (targeted seeding samples "
              "from its output)")
        return
    fg.seed_targeted(max_minutes=budget)
    _show_seeds()


def _show_seeds():
    c = fg.seed_distribution()
    tot = sum(c.values()) or 1
    print(f"\nseed distribution ({tot} total):")
    for k, v in c.most_common():
        print(f"  {k:<12} {v:>6}  {v / tot * 100:5.1f}%")


def cmd_finish(write_combined=True):
    """Stage 2 over the stage-1 chunks, streamed.

    Per-domain csv files are appended chunk by chunk, so peak memory is one
    chunk rather than the whole corpus.
    """
    done, total, _, _ = ns.stage1_progress()
    if done == 0:
        print("stage 1 hasn't run - do that first")
        return
    if done < total:
        print(f"warning: stage 1 looks incomplete ({done}/{total} chunks)")

    if not os.path.exists(cfg.SEED_FILE):
        print(f"no seed file at {cfg.SEED_FILE} - run seed / seed-targeted first")
        return

    model = fg.train_model()

    os.makedirs(cfg.SPLIT_DIR, exist_ok=True)
    split_paths = {d: os.path.join(cfg.SPLIT_DIR, f"{d.lower()}.csv") for d in DOMAINS}
    for p in split_paths.values():
        if os.path.exists(p):
            os.remove(p)
    if write_combined and os.path.exists(cfg.OUTPUT_FILE):
        os.remove(cfg.OUTPUT_FILE)

    counts = {d: 0 for d in DOMAINS}
    written = {d: False for d in DOMAINS}
    combined_header = True
    total_rows = finegrained = 0

    for n, chunk in enumerate(ns.iter_stage1_chunks()):
        mask = chunk["route"] == "finegrain"
        if mask.any():
            sub = fg.classify_subset(chunk.loc[mask], model)
            chunk.loc[mask, "domain"] = sub["domain"].values
            if "confidence" not in chunk.columns:
                chunk["confidence"] = pd.NA
            chunk.loc[mask, "confidence"] = sub["confidence"].values
            finegrained += int(mask.sum())

        total_rows += len(chunk)

        for domain, group in chunk.groupby("domain", observed=True):
            d = str(domain)
            if d not in split_paths:
                continue
            group.to_csv(split_paths[d], mode="a", index=False,
                         header=not written[d], encoding=cfg.CSV_ENCODING)
            written[d] = True
            counts[d] += len(group)

        if write_combined:
            chunk.to_csv(cfg.OUTPUT_FILE, mode="a", index=False,
                         header=combined_header, encoding=cfg.CSV_ENCODING)
            combined_header = False

        print(f"  chunk {n}: {len(chunk):,} rows")

    print(f"\n{total_rows:,} rows, {finegrained:,} fine-grained "
          f"({finegrained / max(total_rows, 1) * 100:.1f}%)")
    print("\ndomain distribution:")
    for d, c in sorted(counts.items(), key=lambda x: -x[1]):
        if c:
            print(f"  {d:<12} {c:>9,}  {c / max(total_rows, 1) * 100:6.2f}%  "
                  f"-> {os.path.basename(split_paths[d])}")

    if write_combined:
        print(f"\nalso wrote {os.path.basename(cfg.OUTPUT_FILE)}")


def cmd_evaluate():
    """Score the FULL pipeline (routing + fastText) against the gold set.

    The gold file has a `part` column - random / stratified / recall - and
    they measure different things, so they're reported separately. Averaging
    them into one macro-F1 would be meaningless: the stratified and recall
    parts are deliberately not representative of the corpus.
    """
    from sklearn.metrics import classification_report, confusion_matrix

    if not os.path.exists(cfg.GOLD_LABELED):
        print(f"no {cfg.GOLD_LABELED} - run label_gold.py first")
        return

    gold = pd.read_csv(cfg.GOLD_LABELED, encoding=cfg.CSV_ENCODING)
    gold = gold[gold["domain"].notna() & (gold["domain"] != "")].reset_index(drop=True)
    if gold.empty:
        print("gold file has no labels yet")
        return
    print(f"evaluating on {len(gold)} labelled rows")

    routed = ns.run_nemo_stage(gold.copy())
    mask = routed["route"] == "finegrain"
    if mask.any():
        model = fg.train_model()
        sub = fg.classify_subset(routed.loc[mask], model)
        routed.loc[mask, "domain"] = sub["domain"].values

    truth, pred = gold["domain"], routed["domain"]

    def report(name, sel):
        if sel.sum() == 0:
            return
        print(f"\n=== {name} ({sel.sum()} rows) ===")
        print(classification_report(truth[sel], pred[sel], zero_division=0))

    if "part" in gold.columns:
        report("random - overall accuracy, unbiased", gold["part"] == "random")
        report("stratified - per-class precision", gold["part"] == "stratified")
        report("recall probe - misses", gold["part"].astype(str).str.startswith("recall"))

    print(f"\n=== all {len(gold)} rows pooled (NOT a corpus-level macro-F1) ===")
    print(classification_report(truth, pred, zero_division=0))
    print("confusion matrix:")
    print(pd.DataFrame(confusion_matrix(truth, pred, labels=DOMAINS),
                       index=DOMAINS, columns=DOMAINS).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="status",
                    choices=["status", "probe", "stage1", "seed", "seed-targeted",
                             "finish", "evaluate", "all"])
    ap.add_argument("--budget", type=float, default=None,
                    help="minutes of wall time before stopping cleanly")
    ap.add_argument("--no-combined", action="store_true",
                    help="skip the single big domain_classified.csv, splits only")
    args = ap.parse_args()

    if args.cmd == "status":
        cmd_status()
    elif args.cmd == "probe":
        cmd_probe()
    elif args.cmd == "stage1":
        cmd_stage1(args.budget)
    elif args.cmd == "seed":
        cmd_seed(args.budget)
    elif args.cmd == "seed-targeted":
        cmd_seed_targeted(args.budget)
    elif args.cmd == "finish":
        cmd_finish(write_combined=not args.no_combined)
    elif args.cmd == "evaluate":
        cmd_evaluate()
    elif args.cmd == "all":
        if not _have_corpus():
            return
        cmd_stage1(args.budget)
        cmd_seed(args.budget)
        cmd_seed_targeted(args.budget)
        cmd_finish(write_combined=not args.no_combined)
        cmd_evaluate()


if __name__ == "__main__":
    main()
