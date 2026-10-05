"""Paths and shared settings.

Data lives outside the repo - it's ~10GB and the corpus isn't ours to
redistribute. Point NLTM_DATA_DIR at wherever you keep it; everything else
resolves from there.

    set NLTM_DATA_DIR=E:\\Machine Learning\\Dommain Classifier     (windows)
    export NLTM_DATA_DIR=/data/nltm                                 (linux)

Unset, it defaults to the current working directory, so running the scripts
from inside the data folder just works.
"""

import os

DATA_DIR = os.environ.get("NLTM_DATA_DIR", os.getcwd())


def p(*parts):
    return os.path.join(DATA_DIR, *parts)


# ── inputs ───────────────────────────────────────────────────────────────
RAW_DIR = p("data_raw")              # the 6 source xlsx workbooks
INPUT_FILE = p("en_ne_parallel.csv")  # merged corpus

# ── stage 1 ──────────────────────────────────────────────────────────────
STAGE1_DIR = p("stage1_out")          # resumable parquet checkpoints
MODEL_NAME = "nvidia/domain-classifier"
CONF_THRESHOLD = 0.6                   # below this -> stage 2 decides
BATCH_SIZE = 64
MAX_LENGTH = 96                        # corpus p99 is 50 tokens, max seen 82
CPU_THREADS = 8
CHUNK_ROWS = 50_000                    # ~35 s/chunk on a 4050
# int8 dynamic quant gives ~1.4x on cpu but WRECKS this model - mean
# confidence 0.93 -> 0.31, ~90% of rows collapse into Arts_and_Entertainment.
QUANTIZE_ON_CPU = False

# ── stage 2 ──────────────────────────────────────────────────────────────
SEED_FILE = p("labeled_seed.txt")
FASTTEXT_MODEL = p("domain_classifier.bin")
OLLAMA_MODEL = "llama3.2:3b"
NUM_PREDICT = 5                        # we only want one word back
KEEP_ALIVE = "30m"                     # keep it resident between calls
N_PER_DOMAIN = 200                     # for the uniform-random seeding pass
N_PER_BUCKET = 700                     # for the targeted pass

# ── outputs ──────────────────────────────────────────────────────────────
OUTPUT_FILE = p("domain_classified.csv")
SPLIT_DIR = p("domain_splits")
XLSX_DIR = p("xlsx_out")
GOLD_TO_LABEL = p("gold_test_TO_LABEL.csv")
GOLD_LABELED = p("gold_test_labeled.csv")

# ── misc ─────────────────────────────────────────────────────────────────
TEXT_COL = "english"                   # the ONLY column ever classified
# excel chokes well before its 1,048,576 row cap; 100k opens instantly
XLSX_PART_ROWS = 100_000
# the nepali side is devanagari and these get opened in excel on windows,
# which assumes the ansi codepage unless there's a BOM. always utf-8-sig.
CSV_ENCODING = "utf-8-sig"

DOMAINS = ["Tech", "Agriculture", "Climate", "Tourism",
           "Admin", "Health", "Law", "Education", "General"]

# hf weights land here instead of the user profile
os.environ.setdefault("HF_HOME", p("hf_cache"))
# 11434 is the ollama default; override if it's taken or, as on one windows
# box here, inside a reserved port exclusion range.
os.environ.setdefault("OLLAMA_HOST", os.environ.get("NLTM_OLLAMA_HOST", "127.0.0.1:11434"))
