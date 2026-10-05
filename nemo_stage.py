"""Stage 1 - coarse split with NVIDIA's domain classifier.

Runs nvidia/domain-classifier (deberta-v3 backbone, 26-class web taxonomy)
over the english column and routes every row to one of three outcomes:

  general    the nemo label has no overlap with our taxonomy at all
             (Sports, Shopping, Adult, Games, Finance...) -> General, final.
  direct     the nemo label maps 1:1 onto one of ours
             (Health->Health, Jobs_and_Education->Education, ...) -> final.
  finegrain  the label straddles two or more of our domains, or the model
             wasn't confident -> stage 2 decides.

Why route at all instead of using nemo's labels directly: its taxonomy has no
Agriculture, Climate, Tourism or Admin. Those four only exist inside buckets
that mean something broader (Science, Business_and_Industrial,
Travel_and_Transportation, Law_and_Government), so a lossy 26->9 mapping
would be guesswork. Routing lets nemo answer the ~67% it can answer cheaply
and hands the genuinely ambiguous third to a classifier trained on our actual
taxonomy.

Output is chunked and resumable: one parquet per CHUNK_ROWS rows, written as
soon as it's done. Re-running skips finished chunks, so an interrupted run
loses at most the chunk in flight.
"""

import glob
import os
import time

import pandas as pd
import torch
from torch import nn
from transformers import AutoTokenizer, AutoConfig, AutoModel
from huggingface_hub import PyTorchModelHubMixin

import config as cfg

# ── routing tables ───────────────────────────────────────────────────────

# no overlap with our domains at all -> General, stage 2 never sees these
DIRECT_GENERAL = {
    "Adult", "Arts_and_Entertainment", "Autos_and_Vehicles", "Beauty_and_Fitness",
    "Books_and_Literature", "Finance", "Games", "Hobbies_and_Leisure",
    "Home_and_Garden", "Online_Communities", "People_and_Society", "Pets_and_Animals",
    "Real_Estate", "Sensitive_Subjects", "Shopping", "Society", "Sports", "News",
}

# unambiguous 1:1
DIRECT_MAP = {
    "Health": "Health",
    "Jobs_and_Education": "Education",
    "Computers_and_Electronics": "Tech",
    "Internet_and_Telecom": "Tech",
}

# straddles two or more of ours, so fasttext decides:
#   Law_and_Government        -> Admin or Law
#   Travel_and_Transportation -> Tourism or ordinary transit
#   Science                   -> Climate or general science
#   Business_and_Industrial   -> Agriculture or general industry
#   Food_and_Drink            -> Agriculture or General
NEEDS_FINEGRAIN = {
    "Law_and_Government", "Travel_and_Transportation", "Science",
    "Business_and_Industrial", "Food_and_Drink",
}


class CustomModel(nn.Module, PyTorchModelHubMixin):
    """nvidia/domain-classifier is NOT an AutoModelForSequenceClassification.

    It's a deberta backbone plus a custom fc head, published through
    PyTorchModelHubMixin. Loading it with the Auto class "succeeds" but
    silently initialises the head randomly - you get confident-looking
    garbage. This wrapper matches the real architecture.
    """

    def __init__(self, config):
        super().__init__()
        self.model = AutoModel.from_pretrained(config["base_model"])
        self.dropout = nn.Dropout(config["fc_dropout"])
        self.fc = nn.Linear(self.model.config.hidden_size, len(config["id2label"]))

    def forward(self, input_ids, attention_mask):
        features = self.model(input_ids=input_ids,
                              attention_mask=attention_mask).last_hidden_state
        dropped = self.dropout(features)
        outputs = self.fc(dropped)
        return torch.softmax(outputs[:, 0, :], dim=1)


def load_classifier():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"loading {cfg.MODEL_NAME} on {device}")

    conf = AutoConfig.from_pretrained(cfg.MODEL_NAME)
    tok = AutoTokenizer.from_pretrained(cfg.MODEL_NAME)
    model = CustomModel.from_pretrained(cfg.MODEL_NAME).to(device)

    # the checkpoint is mixed precision - the backbone comes back fp16 while
    # the fc head stays fp32, which dies with "mat1 and mat2 must have the
    # same dtype". force one dtype per device.
    if device == "cuda":
        model = model.half()
    else:
        model = model.float()
        torch.set_num_threads(cfg.CPU_THREADS)
        if cfg.QUANTIZE_ON_CPU:
            model = torch.quantization.quantize_dynamic(
                model, {nn.Linear}, dtype=torch.qint8)
    model.eval()

    return tok, model, device, conf.id2label


def classify_batch(sentences, tok, model, device, id2label):
    inputs = tok(sentences, padding="longest", truncation=True,
                 max_length=cfg.MAX_LENGTH, return_tensors="pt").to(device)
    with torch.no_grad():
        probs = model(inputs["input_ids"], inputs["attention_mask"])
        scores, idx = probs.max(dim=-1)
    return [id2label[i.item()] for i in idx], [s.item() for s in scores]


def route(label, score):
    if score < cfg.CONF_THRESHOLD:
        return "finegrain"
    if label in DIRECT_GENERAL:
        return "general"
    if label in DIRECT_MAP:
        return "direct"
    # unrecognised label (taxonomy drift) also falls through to stage 2
    return "finegrain"


def _classify_frame(df, tok, model, device, id2label, batch_size=None, quiet=False):
    batch_size = batch_size or cfg.BATCH_SIZE
    sentences = (df[cfg.TEXT_COL].fillna("").astype(str)
                 .str.strip().str.replace("\n", " ", regex=False).tolist())

    # group similar-length sentences into the same batch. with
    # padding="longest" a random batch pads a 15-token sentence out to the
    # batch max (~60), wasting most of the compute. sorting first and
    # restoring order afterwards is ~2x on this corpus, and changes nothing
    # about the output.
    order = sorted(range(len(sentences)), key=lambda i: len(sentences[i]))

    labels = [None] * len(sentences)
    scores = [0.0] * len(sentences)
    done = 0
    for i in range(0, len(order), batch_size):
        idx = order[i:i + batch_size]
        l, s = classify_batch([sentences[j] for j in idx], tok, model, device, id2label)
        for j, lab, sc in zip(idx, l, s):
            labels[j] = lab
            scores[j] = sc
        done += len(idx)
        if not quiet and (i // batch_size) % 200 == 0:
            print(f"    {done:,}/{len(sentences):,}")

    return labels, scores


def apply_routing(df, labels, scores):
    df["nemo_label"] = labels
    df["nemo_score"] = scores
    df["route"] = [route(l, s) for l, s in zip(labels, scores)]

    df["domain"] = None
    df.loc[df["route"] == "general", "domain"] = "General"
    direct = df["route"] == "direct"
    df.loc[direct, "domain"] = df.loc[direct, "nemo_label"].map(DIRECT_MAP)
    return df


def run_nemo_stage(df, batch_size=None):
    """single-shot, in-memory. for the gold set and smoke tests."""
    tok, model, device, id2label = load_classifier()
    labels, scores = _classify_frame(df, tok, model, device, id2label, batch_size)
    return apply_routing(df, labels, scores)


def run_nemo_chunked(input_file=None, chunk_rows=None, batch_size=None,
                     max_minutes=None, out_dir=None):
    """
    Resumable pass over the whole corpus. max_minutes stops cleanly after
    that much wall time (model load excluded) so the gpu can be handed back -
    just run it again to continue.
    """
    input_file = input_file or cfg.INPUT_FILE
    chunk_rows = chunk_rows or cfg.CHUNK_ROWS
    out_dir = out_dir or cfg.STAGE1_DIR
    os.makedirs(out_dir, exist_ok=True)

    tok, model, device, id2label = load_classifier()
    t0 = time.time()
    processed = skipped = 0

    reader = pd.read_csv(input_file, chunksize=chunk_rows, encoding=cfg.CSV_ENCODING)
    for n, chunk in enumerate(reader):
        path = os.path.join(out_dir, f"chunk_{n:04d}.parquet")
        if os.path.exists(path):
            skipped += 1
            continue

        if max_minutes is not None and (time.time() - t0) / 60 >= max_minutes:
            print(f"\nhit {max_minutes} min budget - stopping at chunk {n}")
            print(f"run again to continue ({skipped} chunks already done)")
            break

        print(f"chunk {n} ({len(chunk):,} rows)")
        labels, scores = _classify_frame(chunk, tok, model, device, id2label,
                                         batch_size, quiet=True)
        apply_routing(chunk, labels, scores)
        chunk.to_parquet(path, index=False)

        processed += 1
        rate = (processed * chunk_rows) / max(time.time() - t0, 1e-9)
        print(f"  saved {os.path.basename(path)}  ({rate:.0f} sent/sec)")

    print(f"\nstage1: {processed} chunks this run, {skipped} already done")
    return out_dir


def chunk_paths(out_dir=None):
    return sorted(glob.glob(os.path.join(out_dir or cfg.STAGE1_DIR, "chunk_*.parquet")))


def iter_stage1_chunks(out_dir=None, columns=None):
    """stream the finished chunks one at a time - constant memory, unlike
    load_stage1_output which holds all 1.7M rows at once."""
    for path in chunk_paths(out_dir):
        yield pd.read_parquet(path, columns=columns)


def load_stage1_output(out_dir=None, columns=None):
    """everything in one frame. ~2-3 GB for the full corpus, so prefer
    iter_stage1_chunks when you only need a streaming pass."""
    paths = chunk_paths(out_dir)
    if not paths:
        raise FileNotFoundError(f"no chunks in {out_dir or cfg.STAGE1_DIR} - run stage 1 first")
    print(f"loading {len(paths)} stage1 chunks")
    df = pd.concat([pd.read_parquet(p, columns=columns) for p in paths],
                   ignore_index=True)
    # these are low-cardinality strings repeated over a million rows
    for col in ("nemo_label", "route", "domain", "source_file"):
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def stage1_progress(input_file=None, chunk_rows=None, out_dir=None):
    """done/expected chunks and row counts, without loading any row data.

    reads parquet footers for the finished rows and only falls back to
    scanning the csv when stage 1 hasn't started - counting lines in a 590MB
    file takes ~20 s and there's no reason to pay it every status call.
    """
    import pyarrow.parquet as pq

    chunk_rows = chunk_rows or cfg.CHUNK_ROWS
    paths = chunk_paths(out_dir)
    done_rows = sum(pq.ParquetFile(p).metadata.num_rows for p in paths)

    input_file = input_file or cfg.INPUT_FILE
    total_rows = None
    if os.path.exists(input_file):
        size = os.path.getsize(input_file)
        if paths:
            # extrapolate from bytes-per-row so far, good enough for a % bar
            bytes_per_row = size / max(done_rows, 1) if done_rows else None
            total_rows = int(size / bytes_per_row) if bytes_per_row else None
        if total_rows is None:
            with open(input_file, encoding=cfg.CSV_ENCODING) as f:
                total_rows = sum(1 for _ in f) - 1

    total_chunks = -(-total_rows // chunk_rows) if total_rows else len(paths)
    return len(paths), total_chunks, done_rows, total_rows


if __name__ == "__main__":
    sample = pd.DataFrame({cfg.TEXT_COL: [
        "The committee shall submit the report within 30 days.",
        "Pokhara is known for its scenic lakeside views.",
        "Farmers in the hills cultivate millet and buckwheat.",
        "The new striker scored twice in last night's match.",
        "Wash hands frequently to prevent the spread of infection.",
    ]})
    out = run_nemo_stage(sample)
    print(out[[cfg.TEXT_COL, "nemo_label", "nemo_score", "route", "domain"]])
