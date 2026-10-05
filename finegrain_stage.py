"""Stage 2 - fine-grain the ambiguous third with LLM-seeded fastText.

Stage 1 hands over the rows whose nemo label straddles two of our domains, or
where it wasn't confident. Those need a classifier that knows OUR taxonomy,
and we have no hand-labelled training data for it.

So: weak supervision. A local LLM (llama3.2:3b via ollama) labels a few
thousand sentences, fastText trains on those labels, and fastText - not the
LLM - classifies the corpus. The LLM is slow per call but only runs on the
seed set; fastText is the thing that touches hundreds of thousands of rows,
and it predicts the whole list in seconds.

Why not just run the LLM over everything: at ~0.09 s/call on a gpu that's
~14 hours for 560k rows, versus seconds for fastText. Why not train on the
whole corpus with the LLM: same reason.

Two seeding strategies, and the second matters more than it looks:

  seed_labels()    uniform random sample of the corpus. Fine for the common
                   domains, useless for the rare ones - a 1,732-sentence
                   sample yielded 17 Climate and 26 Agriculture seeds.
  seed_targeted()  samples per nemo bucket instead (Science for Climate,
                   Business_and_Industrial + Food_and_Drink for Agriculture,
                   ...). Took Climate to 313 and Agriculture to 806. It also
                   matches the inference distribution, since fastText only
                   ever sees finegrain rows, which come from exactly those
                   buckets.

Both append to the seed file and skip sentences already in it, so they're
interruptible and can be re-run to top up.
"""

import os
import time

import ollama
import fasttext
import pandas as pd

import config as cfg

DOMAINS = cfg.DOMAINS

PROMPT = """Classify this English sentence into exactly one domain.
Domains: Tech, Agriculture, Climate, Tourism, Admin, Health, Law, Education, General
Reply with only the domain name, nothing else.
Sentence: {sent}"""

# which nemo buckets to draw targeted seeds from - the ones stage 1 routes to
# finegrain, i.e. where our rare domains actually live
SEED_BUCKETS = [
    "Science",                    # -> Climate or general science
    "Business_and_Industrial",    # -> Agriculture or general industry
    "Food_and_Drink",             # -> Agriculture or General
    "Travel_and_Transportation",  # -> Tourism or ordinary transit
    "Law_and_Government",         # -> Admin or Law
]


def _label_sentences(sentences, seed_file=None, max_minutes=None):
    """Shared ollama loop. Appends and flushes per line, so a hard kill
    doesn't lose the batch, and skips anything already labelled."""
    seed_file = seed_file or cfg.SEED_FILE

    done = set()
    if os.path.exists(seed_file):
        with open(seed_file, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(" ", 1)
                if len(parts) == 2:
                    done.add(parts[1])

    todo = [s for s in sentences if str(s).strip() not in done]
    print(f"{len(done)} already labelled, {len(todo)} to do")
    if not todo:
        return 0

    t0 = time.time()
    added = 0
    with open(seed_file, "a", encoding="utf-8") as f:
        for i, sent in enumerate(todo):
            if max_minutes is not None and (time.time() - t0) / 60 >= max_minutes:
                print(f"\nhit {max_minutes} min budget - {added} added this run")
                break
            try:
                resp = ollama.chat(
                    model=cfg.OLLAMA_MODEL,
                    messages=[{"role": "user",
                               "content": PROMPT.format(sent=str(sent).strip())}],
                    options={"num_predict": cfg.NUM_PREDICT},
                    keep_alive=cfg.KEEP_ALIVE,
                )
                label = resp["message"]["content"].strip()
                # the model sometimes answers with something off-taxonomy;
                # drop those rather than coercing them
                if label in DOMAINS:
                    f.write(f"__label__{label} {str(sent).strip()}\n")
                    f.flush()
                    added += 1
            except Exception as e:
                print(f"  row {i} failed: {e}")
                continue

            if (i + 1) % 200 == 0:
                rate = (i + 1) / (time.time() - t0)
                left = (len(todo) - i - 1) / max(rate, 1e-9) / 60
                print(f"  {i + 1}/{len(todo)}  {rate:.1f}/sec  ~{left:.0f} min left")

    print(f"added {added}, seed file now {len(done) + added} samples")
    return added


def seed_labels(df, seed_file=None, n_per_domain=None, max_minutes=None):
    """uniform random seeding. cheap, but under-samples the rare domains."""
    n_per_domain = n_per_domain or cfg.N_PER_DOMAIN
    pool = df[cfg.TEXT_COL].dropna()
    n = min(n_per_domain * len(DOMAINS), len(pool))
    sample = pool.sample(n=n, random_state=42).tolist()
    return _label_sentences(sample, seed_file, max_minutes)


def seed_targeted(stage1_dir=None, seed_file=None, n_per_bucket=None,
                  buckets=None, max_minutes=None):
    """seeds per nemo bucket - the only way to get enough Climate and
    Agriculture examples out of this corpus."""
    import nemo_stage as ns

    n_per_bucket = n_per_bucket or cfg.N_PER_BUCKET
    buckets = buckets or SEED_BUCKETS

    paths = ns.chunk_paths(stage1_dir)
    if not paths:
        raise FileNotFoundError("no stage1 chunks - run stage 1 first")

    print(f"reading stage1 output from {len(paths)} chunks")
    df = pd.concat(
        [pd.read_parquet(p, columns=[cfg.TEXT_COL, "nemo_label", "route"])
         for p in paths], ignore_index=True)
    df = df[df["route"] == "finegrain"]

    todo = []
    for b in buckets:
        pool = df.loc[df["nemo_label"] == b, cfg.TEXT_COL].dropna()
        take = min(n_per_bucket, len(pool))
        if take:
            todo += pool.sample(n=take, random_state=42).tolist()
        print(f"  {b:<28} {take:>5} sentences")

    return _label_sentences(todo, seed_file, max_minutes)


def seed_distribution(seed_file=None):
    import collections
    seed_file = seed_file or cfg.SEED_FILE
    c = collections.Counter()
    if not os.path.exists(seed_file):
        return c
    with open(seed_file, encoding="utf-8") as f:
        for line in f:
            if line.startswith("__label__"):
                c[line.split(" ", 1)[0].replace("__label__", "")] += 1
    return c


def train_model(seed_file=None, model_file=None, threads=None, force=False):
    seed_file = seed_file or cfg.SEED_FILE
    model_file = model_file or cfg.FASTTEXT_MODEL

    if os.path.exists(model_file) and not force:
        print(f"model found ({os.path.basename(model_file)}), loading from disk")
        return fasttext.load_model(model_file)

    print("training fasttext...")
    model = fasttext.train_supervised(
        input=seed_file,
        epoch=10,        # 9 classes and a few thousand lines; more overfits
        lr=0.5,
        wordNgrams=2,    # bigrams catch "climate change", "district office"
        dim=50,          # 100/300 adds nothing at this data size
        thread=threads or cfg.CPU_THREADS,
        loss="softmax",
    )
    model.save_model(model_file)
    print(f"model saved: {os.path.basename(model_file)}")
    return model


def classify_subset(df, model):
    """Bulk predict. Pass the whole list to model.predict - looping one
    sentence at a time is orders of magnitude slower and is the single
    easiest way to make this pipeline take hours instead of seconds."""
    out = df.copy()
    if len(out) == 0:
        out["domain"] = []
        out["confidence"] = []
        return out

    sentences = (out[cfg.TEXT_COL].fillna("").astype(str)
                 .str.strip().str.replace("\n", " ", regex=False).tolist())
    labels, probs = model.predict(sentences, k=1)

    out["domain"] = [l[0].replace("__label__", "") for l in labels]
    out["confidence"] = [round(float(p[0]), 4) for p in probs]
    return out
