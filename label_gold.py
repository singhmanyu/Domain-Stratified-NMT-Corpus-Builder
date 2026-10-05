"""Keyboard labeller for the gold set.

Shows one english sentence at a time; press 1-9 to label, s to skip, q to
save and quit. Saves after every single label, so it's safe to quit and come
back - it resumes from whatever is already labelled.

Deliberately does NOT show the model's prediction, even though it's in the
file. Seeing it first anchors the judgement and quietly turns the gold set
into a measure of agreement rather than of truth.

    python label_gold.py
"""

import os

import pandas as pd

import config as cfg

DOMAINS = cfg.DOMAINS
MENU = "  ".join(f"[{i + 1}]{d}" for i, d in enumerate(DOMAINS))


def main():
    src = cfg.GOLD_LABELED if os.path.exists(cfg.GOLD_LABELED) else cfg.GOLD_TO_LABEL
    if not os.path.exists(src):
        print(f"no gold file - run make_gold_set.py first")
        return

    df = pd.read_csv(src, encoding=cfg.CSV_ENCODING)
    if "domain" not in df.columns:
        df["domain"] = ""
    df["domain"] = df["domain"].fillna("").astype(str)

    todo = df.index[df["domain"] == ""].tolist()
    print(f"{len(df) - len(todo)}/{len(df)} already labelled, {len(todo)} to go")
    print(MENU)
    print("s = skip, q = save and quit\n")

    for n, i in enumerate(todo):
        print(f"--- {n + 1}/{len(todo)} ---")
        print(str(df.at[i, "english"])[:300])
        choice = input("> ").strip().lower()

        if choice == "q":
            break
        if choice == "s" or not choice:
            continue
        if choice.isdigit() and 1 <= int(choice) <= len(DOMAINS):
            df.at[i, "domain"] = DOMAINS[int(choice) - 1]
            df.to_csv(cfg.GOLD_LABELED, index=False, encoding=cfg.CSV_ENCODING)
        else:
            print("  ? use 1-9, s, or q")

    labelled = (df["domain"] != "").sum()
    df.to_csv(cfg.GOLD_LABELED, index=False, encoding=cfg.CSV_ENCODING)
    print(f"\nsaved {os.path.basename(cfg.GOLD_LABELED)} - {labelled}/{len(df)} labelled")
    if labelled:
        print(df[df["domain"] != ""]["domain"].value_counts().to_string())
    if labelled == len(df):
        print("\nall done - run: python domain_classifier_nltm.py evaluate")


if __name__ == "__main__":
    main()
