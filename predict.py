"""Inference for the trained two-stage classifier.

Same routing as the batch pipeline, but loads once and stays warm so a UI
can call it per sentence. Stage 1 decides; only the rows it can't place go
to the fasttext model.
"""

import os

import fasttext
import pandas as pd

import nemo_stage
import finegrain_stage as fg

MODEL_FILE = fg.MODEL_FILE
DOMAINS = fg.DOMAINS


class Pipeline:
    """Holds both models in memory. Build one, reuse it."""

    def __init__(self, model_file=MODEL_FILE):
        if not os.path.exists(model_file):
            raise FileNotFoundError(
                f"{model_file} not found - run domain_classifier_nltm.py finish first"
            )
        self.tok, self.model, self.device, self.id2label = nemo_stage.load_classifier()
        self.ft = fasttext.load_model(model_file)

    def classify(self, sentences, batch_size=nemo_stage.BATCH_SIZE):
        """-> DataFrame with english, domain, stage, nemo_label, nemo_score, confidence."""
        sentences = [s for s in (str(x).strip() for x in sentences) if s]
        if not sentences:
            return pd.DataFrame(columns=["english", "domain", "stage",
                                         "nemo_label", "nemo_score", "confidence"])

        df = pd.DataFrame({"english": sentences})
        labels, scores = nemo_stage._classify_frame(
            df, self.tok, self.model, self.device, self.id2label,
            "english", batch_size, quiet=True,
        )
        df = nemo_stage.apply_routing(df, labels, scores)
        df["confidence"] = df["nemo_score"]

        mask = df["route"] == "finegrain"
        if mask.any():
            sub = fg.classify_subset(df.loc[mask], self.ft)
            df.loc[mask, "domain"] = sub["domain"].values
            df.loc[mask, "confidence"] = sub["confidence"].values

        df["stage"] = df["route"].map(
            {"general": "stage 1 (general)", "direct": "stage 1 (direct)",
             "finegrain": "stage 2 (fasttext)"}
        )
        return df[["english", "domain", "stage", "nemo_label", "nemo_score", "confidence"]]

    def classify_one(self, sentence):
        out = self.classify([sentence])
        return None if out.empty else out.iloc[0].to_dict()

    def fasttext_scores(self, sentence, k=len(DOMAINS)):
        """All domain probabilities from the stage-2 model, for the UI's bar chart.
        Meaningful only for rows stage 1 actually routed to fasttext."""
        sentence = str(sentence).strip().replace("\n", " ")
        # pass a list, not a bare string - fasttext's single-string path does
        # np.array(..., copy=False), which numpy 2 refuses
        labels, probs = self.ft.predict([sentence], k=k)
        return {l.replace("__label__", ""): float(p)
                for l, p in zip(labels[0], probs[0])}


_PIPELINE = None


def get_pipeline():
    """Lazy singleton - the deberta load is a few seconds, do it once."""
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = Pipeline()
    return _PIPELINE


if __name__ == "__main__":
    # minimal cli: python predict.py "some sentence" ...
    # or pipe sentences in, one per line
    import sys

    args = sys.argv[1:]
    lines = args if args else [l for l in sys.stdin.read().splitlines() if l.strip()]
    if not lines:
        print('usage: python predict.py "sentence" [...]   (or pipe lines on stdin)')
        raise SystemExit(1)

    out = get_pipeline().classify(lines)
    for _, r in out.iterrows():
        print(f"{r['domain']:<12} {r['confidence']:.3f}  {r['stage']:<20} {r['english'][:70]}")
