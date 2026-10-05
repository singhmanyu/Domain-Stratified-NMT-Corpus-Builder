"""Held-out evaluation of the fastText stage against the LLM seed labels.

IMPORTANT - what this measures, and what it does not.

  It measures: how faithfully fastText reproduces llama3.2:3b's labels on
  sentences it was not trained on. That is a real, useful number - it tells
  you whether the student learned the teacher, whether any class is
  unlearnable from the seeds available, and which classes bleed into each
  other.

  It does NOT measure corpus accuracy. The labels here are weak supervision
  from a 3B model, not ground truth. If the teacher is wrong in a systematic
  way, a high score here means fastText faithfully reproduced that error.
  Only the hand-labelled gold set (make_gold_set.py -> label_gold.py ->
  `domain_classifier_nltm.py evaluate`) measures accuracy.

Stratified k-fold so the rare classes appear in every fold.

    python evaluate_seeds.py
    python evaluate_seeds.py --folds 10
"""

import argparse
import os
import tempfile

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold

import fasttext

import config as cfg


def load_seeds(seed_file=None):
    seed_file = seed_file or cfg.SEED_FILE
    if not os.path.exists(seed_file):
        raise FileNotFoundError(f"no seed file at {seed_file} - run seeding first")

    labels, texts = [], []
    with open(seed_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("__label__"):
                continue
            parts = line.split(" ", 1)
            if len(parts) != 2:
                continue
            labels.append(parts[0].replace("__label__", ""))
            texts.append(parts[1])
    return pd.DataFrame({"label": labels, "text": texts})


def train_fold(train_df):
    """train on a temp file - fasttext only reads from disk"""
    fd, path = tempfile.mkstemp(suffix=".txt", text=True)
    os.close(fd)
    try:
        with open(path, "w", encoding="utf-8") as f:
            for lab, txt in zip(train_df["label"], train_df["text"]):
                f.write(f"__label__{lab} {txt}\n")
        return fasttext.train_supervised(
            input=path, epoch=cfg.FT_EPOCH, lr=cfg.FT_LR,
            wordNgrams=cfg.FT_WORD_NGRAMS, dim=cfg.FT_DIM,
            minn=cfg.FT_MINN, maxn=cfg.FT_MAXN,
            thread=cfg.CPU_THREADS, loss="softmax",
        )
    finally:
        os.remove(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    df = load_seeds()
    print(f"{len(df):,} seed sentences")
    print(df["label"].value_counts().to_string())

    # drop classes too small to stratify
    counts = df["label"].value_counts()
    too_small = counts[counts < args.folds].index.tolist()
    if too_small:
        print(f"\nexcluded (fewer than {args.folds} examples): {too_small}")
        df = df[~df["label"].isin(too_small)]

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=7)
    truths, preds = [], []

    print(f"\n{args.folds}-fold stratified cross-validation")
    for k, (tr, te) in enumerate(skf.split(df["text"], df["label"]), 1):
        model = train_fold(df.iloc[tr])
        texts = df.iloc[te]["text"].tolist()
        out = model.predict(texts, k=1)[0]
        fold_pred = [l[0].replace("__label__", "") for l in out]
        fold_true = df.iloc[te]["label"].tolist()
        truths += fold_true
        preds += fold_pred
        print(f"  fold {k}: macro-F1 {f1_score(fold_true, fold_pred, average='macro', zero_division=0):.3f}")

    labels = sorted(set(truths))
    print("\n=== pooled across folds ===")
    print(classification_report(truths, preds, zero_division=0, digits=3))

    macro = f1_score(truths, preds, average="macro", zero_division=0)
    micro = f1_score(truths, preds, average="micro", zero_division=0)
    weighted = f1_score(truths, preds, average="weighted", zero_division=0)
    acc = float(np.mean(np.array(truths) == np.array(preds)))
    print(f"accuracy      {acc:.3f}")
    print(f"macro-F1      {macro:.3f}")
    print(f"micro-F1      {micro:.3f}")
    print(f"weighted-F1   {weighted:.3f}")

    print("\nconfusion matrix (rows = llm label, cols = fasttext):")
    cm = pd.DataFrame(confusion_matrix(truths, preds, labels=labels),
                      index=labels, columns=labels)
    print(cm.to_string())

    print("\nNOTE: this is fidelity to the llama3.2:3b seed labels, NOT")
    print("corpus accuracy. For accuracy, label the gold set and run")
    print("`python domain_classifier_nltm.py evaluate`.")


if __name__ == "__main__":
    main()
