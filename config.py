"""Paths and shared settings.

Everything language- and taxonomy-specific lives here. The pipeline reads the
english column and nothing else, so changing TARGET_COL and DOMAINS is all it
takes to point it at a different language pair or a different set of domains.

Data lives outside the repo - it's ~10GB and the corpus isn't ours to
redistribute. Point NLTM_DATA_DIR at wherever you keep it; everything else
resolves from there.

    set NLTM_DATA_DIR=E:\\Machine Learning\\Dommain Classifier     (windows)
    export NLTM_DATA_DIR=/data/nltm                                 (linux)

Unset, it defaults to the current working directory, so running the scripts
from inside the data folder just works.
"""

import os

# `or os.getcwd()` and not a dict default: a var that is set but empty - which
# is what `set NLTM_DATA_DIR=` leaves behind - would otherwise resolve every
# path to a bare filename against an unknown directory
DATA_DIR = os.environ.get("NLTM_DATA_DIR") or os.getcwd()


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
# classification is deterministic - ollama's default temperature (0.8) made
# the model disagree with itself on 43% of repeated sentences, which is pure
# label noise. at 0 it agrees with itself 95.8% of the time.
TEMPERATURE = 0.0
KEEP_ALIVE = "30m"                     # keep it resident between calls
N_PER_DOMAIN = 200                     # for the uniform-random seeding pass
N_PER_BUCKET = 700                     # for the targeted pass

# fastText hyperparameters, from a 300s autotune run against a stratified
# 20% holdout. the defaults everyone copies (dim=50, epoch=10, wordNgrams=2)
# scored macro-F1 0.503 on the same split; these score 0.568. the big win is
# character n-grams (minn/maxn) - whole-word bigrams are too sparse on short
# sentences.
FT_DIM = 20
FT_EPOCH = 100
FT_LR = 0.831
FT_WORD_NGRAMS = 5
FT_MINN = 3
FT_MAXN = 6

# ── outputs ──────────────────────────────────────────────────────────────
OUTPUT_FILE = p("domain_classified.csv")
SPLIT_DIR = p("domain_splits")
XLSX_DIR = p("xlsx_out")
GOLD_TO_LABEL = p("gold_test_TO_LABEL.csv")
GOLD_LABELED = p("gold_test_labeled.csv")

# ── columns ──────────────────────────────────────────────────────────────
# Only the english column is ever read by a model. The other side is carried
# through untouched, which is why this works for english paired with any
# language - nothing downstream knows or cares what TARGET_COL contains.
TEXT_COL = "english"                   # the ONLY column ever classified
# what the translation column is called in the output. purely cosmetic -
# set NLTM_TARGET_COL=nepali, hindi, french... to have the output files use
# that name instead.
TARGET_COL = os.environ.get("NLTM_TARGET_COL") or "target"

# ── misc ─────────────────────────────────────────────────────────────────
# excel chokes well before its 1,048,576 row cap; 100k opens instantly
XLSX_PART_ROWS = 100_000
# the target side is often in a non-latin script, and these files get opened
# in excel on windows, which assumes the ansi codepage unless there's a BOM.
# always utf-8-sig.
CSV_ENCODING = "utf-8-sig"

# The taxonomy. Change this and the pipeline follows - stage 2 is trained
# from scratch against whatever is listed here, so there's no retraining step
# to remember. Stage 1's routing tables in nemo_stage.py map NVIDIA's 26 web
# categories onto these, and need editing to match if you change them.
DOMAINS = ["Tech", "Agriculture", "Climate", "Tourism",
           "Admin", "Health", "Law", "Education", "General"]

# hf weights land here instead of the user profile
os.environ.setdefault("HF_HOME", p("hf_cache"))
# 11434 is the ollama default; override if it's taken or, as on one windows
# box here, inside a reserved port exclusion range.
os.environ.setdefault("OLLAMA_HOST", os.environ.get("NLTM_OLLAMA_HOST", "127.0.0.1:11434"))
