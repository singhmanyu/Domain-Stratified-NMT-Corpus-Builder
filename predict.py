"""Inference against the trained two-stage classifier.

The batch scripts are built to stream a million rows through once. This is
the other shape: load both models, keep them resident, answer one sentence
at a time. Same routing, same models, no retraining.

    from predict import get_pipeline
    get_pipeline().classify_one("The ward office issues birth certificates.")

Also runs as a cli - see the bottom of the file.
"""

import os

import fasttext
import pandas as pd

import config as cfg
import nemo_stage
import finegrain_stage as fg

# every row gets one of these, so the caller can tell which model decided
STAGE_GENERAL = "stage 1 (general)"
STAGE_DIRECT = "stage 1 (direct)"
STAGE_FINEGRAIN = "stage 2 (fasttext)"
STAGE_EMPTY = "skipped (empty)"

_STAGE_BY_ROUTE = {
    "general": STAGE_GENERAL,
    "direct": STAGE_DIRECT,
    "finegrain": STAGE_FINEGRAIN,
}

COLUMNS = [cfg.TEXT_COL, "domain", "stage", "nemo_label", "nemo_score", "confidence"]


class Pipeline:
    """Both models, held open. Construct once and reuse - the deberta load
    is several seconds and dominates everything else at small batch sizes."""

    def __init__(self, model_file=None):
        model_file = model_file or cfg.FASTTEXT_MODEL
        if not os.path.exists(model_file):
            raise FileNotFoundError(
                f"{model_file} not found - train stage 2 first "
                f"(python domain_classifier_nltm.py finish)"
            )
        self.tok, self.model, self.device, self.id2label = nemo_stage.load_classifier()
        self.ft = fasttext.load_model(model_file)

    def classify(self, sentences, batch_size=None):
        """Returns one row per input sentence, in the order given.

        Order and length are the contract here, not a convenience - callers
        join these results back onto their own frame (the nepali side, in
        this project), so dropping or reordering a row silently repairs
        nothing and corrupts the pair.
        """
        text = (pd.Series(list(sentences), dtype="object")
                .fillna("").astype(str).str.strip()
                .str.replace("\n", " ", regex=False))
        df = pd.DataFrame({cfg.TEXT_COL: text.values})
        for col in COLUMNS[1:]:
            df[col] = None
        df["stage"] = STAGE_EMPTY

        # blank rows never reach a model - deberta will happily return a
        # confident label for "" and it means nothing
        work = df.loc[df[cfg.TEXT_COL] != ""].copy()
        if work.empty:
            return df[COLUMNS]

        labels, scores = nemo_stage.classify_frame(
            work, self.tok, self.model, self.device, self.id2label,
            batch_size, quiet=True,
        )
        work = nemo_stage.apply_routing(work, labels, scores)
        work["confidence"] = work["nemo_score"]
        work["stage"] = work["route"].map(_STAGE_BY_ROUTE)

        # only the rows stage 1 couldn't place cost a stage 2 call
        mask = work["route"] == "finegrain"
        if mask.any():
            sub = fg.classify_subset(work.loc[mask], self.ft)
            work.loc[mask, "domain"] = sub["domain"].values
            work.loc[mask, "confidence"] = sub["confidence"].values

        # assign by index, so the blank rows stay where they were
        for col in COLUMNS[1:]:
            df.loc[work.index, col] = work[col].values
        return df[COLUMNS]

    def classify_one(self, sentence):
        out = self.classify([sentence])
        return out.iloc[0].to_dict()

    def fasttext_scores(self, sentence, k=None):
        """Every domain's score from stage 2, highest first.

        This is stage 2's opinion whether or not stage 2 was asked - it is
        the real decision only for rows routed to it, so anything showing
        these to a user needs to say which case it is.
        """
        sentence = str(sentence).strip().replace("\n", " ")
        if not sentence:
            return {}
        # a list, not a bare string: fasttext's single-string path calls
        # np.array(..., copy=False), which numpy 2 raises on
        labels, probs = self.ft.predict([sentence], k=k or len(cfg.DOMAINS))
        return {l.replace("__label__", ""): float(p)
                for l, p in zip(labels[0], probs[0])}


_PIPELINE = None


def get_pipeline():
    """Process-wide singleton. A ui handler gets called per keystroke-ish;
    it must not reload deberta each time."""
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = Pipeline()
    return _PIPELINE


if __name__ == "__main__":
    import sys

    lines = sys.argv[1:]
    if not lines and not sys.stdin.isatty():
        lines = [l for l in sys.stdin.read().splitlines() if l.strip()]
    if not lines:
        print('usage: python predict.py "a sentence" ["another" ...]')
        print("       ... or pipe sentences in, one per line")
        raise SystemExit(2)

    for _, r in get_pipeline().classify(lines).iterrows():
        conf = "" if r["confidence"] is None else f"{r['confidence']:.3f}"
        print(f"{str(r['domain'] or '-'):<12} {conf:>6}  {r['stage']:<20} "
              f"{r[cfg.TEXT_COL][:70]}")
