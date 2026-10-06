"""Builds the gold test set to hand-label. Never used for training.

A flat random sample does not work on this corpus. It's ~55% General and
Climate is ~0.14%, so 500 random rows would contain about 2 Climate pairs -
you cannot compute a per-class F1 from that, and macro-F1 would be noise on
exactly the classes the custom classifier exists for.

So the set is built in three parts, tagged in a `part` column:

  random      unbiased sample. gives overall accuracy and the true priors.
  stratified  N rows per PREDICTED domain. gives per-class precision: of the
              rows we call Climate, how many really are Climate.
  recall      rows whose english matches a domain's keywords but which we did
              NOT assign to that domain. probes for misses, which the
              stratified part cannot see by construction - it only samples
              rows we already assigned to the class.

Report the parts separately. Pooling them into one macro-F1 and calling it
the corpus number would be wrong, because two of the three parts are
deliberately not representative.

    python make_gold_set.py
    python make_gold_set.py --random 200 --per-domain 30 --recall 12
"""

import argparse

import os

import pandas as pd

import config as cfg

# used only to find candidate misses for the recall probe
RECALL_PROBES = {
    "Climate": ["climate", "monsoon", "rainfall", "glacier", "emission",
                "drought", "flood"],
    "Agriculture": ["farm", "crop", "irrigat", "livestock", "harvest", "soil",
                    "fertilis", "fertiliz"],
    "Tourism": ["tourist", "trek", "hotel", "heritage", "lakeside", "homestay"],
    "Law": ["court", "tribunal", "petition", "clause", "verdict"],
    "Admin": ["ministry", "committee", "district office", "applicant", "notice"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--random", type=int, default=200)
    ap.add_argument("--per-domain", type=int, default=30)
    ap.add_argument("--recall", type=int, default=12)
    args = ap.parse_args()

    print(f"reading {os.path.basename(cfg.OUTPUT_FILE)}")
    df = pd.read_csv(cfg.OUTPUT_FILE,
                     usecols=[cfg.TEXT_COL, cfg.TARGET_COL, "domain"],
                     encoding=cfg.CSV_ENCODING)
    df = df[df["english"].notna()]

    picks = []

    rnd = df.sample(n=min(args.random, len(df)), random_state=7)
    picks.append(rnd.assign(part="random", predicted=rnd["domain"]))
    print(f"  random      {len(rnd):>4}")

    n_strat = 0
    for _, grp in df.groupby("domain"):
        take = min(args.per_domain, len(grp))
        s = grp.sample(n=take, random_state=7)
        picks.append(s.assign(part="stratified", predicted=s["domain"]))
        n_strat += take
    print(f"  stratified  {n_strat:>4}  ({args.per_domain}/domain where available)")

    low = df["english"].astype(str).str.lower()
    n_recall = 0
    for dom, kws in RECALL_PROBES.items():
        hit = low.str.contains("|".join(kws), regex=True)
        missed = df[hit & (df["domain"] != dom)]
        take = min(args.recall, len(missed))
        if take:
            s = missed.sample(n=take, random_state=7)
            picks.append(s.assign(part=f"recall:{dom}", predicted=s["domain"]))
            n_recall += take
    print(f"  recall      {n_recall:>4}")

    gold = pd.concat(picks, ignore_index=True).drop_duplicates(subset=["english"])
    gold = gold.sample(frac=1, random_state=7).reset_index(drop=True)

    # blanked for the human. `predicted` stays in the file so you can diff
    # afterwards, but label from the sentence - don't just agree with it.
    gold["domain"] = ""
    gold = gold[[cfg.TEXT_COL, cfg.TARGET_COL, "part", "predicted", "domain"]]
    gold.to_csv(cfg.GOLD_TO_LABEL, index=False, encoding=cfg.CSV_ENCODING)

    print(f"\nwrote {os.path.basename(cfg.GOLD_TO_LABEL)} - {len(gold)} rows to label")
    print("next: python label_gold.py")


if __name__ == "__main__":
    main()
