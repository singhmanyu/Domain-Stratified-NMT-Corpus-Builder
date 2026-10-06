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

# ── the taxonomy ─────────────────────────────────────────────────────────
# This is the only place domains are defined. Everything else is derived
# from it: stage 1's routing, the seeding prompt, which NVIDIA buckets get
# targeted seeds, the keyword probe and the UI's domain list. Edit this dict
# and the whole pipeline follows - there is no second list to keep in sync,
# and stage 2 trains from scratch so there is no model to retrain by hand.
#
# Each entry:
#   description  one line, in plain words. Goes into the seeding prompt, so
#                it directly shapes label quality - a domain the teacher has
#                to guess the boundary of is a domain it labels inconsistently.
#   nemo         NVIDIA categories that mean this domain and only this
#                domain. Stage 1's answer is taken as-is, stage 2 never runs.
#   partial      categories this domain shares with ordinary prose - Science
#                is sometimes Climate but usually just general science. These
#                go to stage 2 to be decided per sentence.
#
#                A category two domains both claim is ambiguous too, without
#                anyone marking it: Admin and Law both claim Law_and_Government.
#                Unclaimed categories fall to GENERAL_DOMAIN without stage 2
#                ever seeing them. All of that is computed below, not
#                hand-maintained.
#   probe        keywords for `probe`, the pre-flight sanity check. Crude on
#                purpose - it answers "is this domain even present" before
#                any GPU time is spent.
#
# NVIDIA's 26 categories, for reference when filling in `nemo`:
#   Adult, Arts_and_Entertainment, Autos_and_Vehicles, Beauty_and_Fitness,
#   Books_and_Literature, Business_and_Industrial, Computers_and_Electronics,
#   Finance, Food_and_Drink, Games, Health, Hobbies_and_Leisure,
#   Home_and_Garden, Internet_and_Telecom, Jobs_and_Education,
#   Law_and_Government, News, Online_Communities, People_and_Society,
#   Pets_and_Animals, Real_Estate, Science, Sensitive_Subjects, Shopping,
#   Sports, Travel_and_Transportation
TAXONOMY = {
    "Tech": {
        "description": "computing, software, electronics, telecom, the internet",
        "nemo": ["Computers_and_Electronics", "Internet_and_Telecom"],
        "partial": [],
        "probe": ["software", "computer", "internet", "digital", "network"],
    },
    "Agriculture": {
        "description": "farming, crops, livestock, irrigation, food production",
        "nemo": [],
        # most industry and most food writing is not about farming
        "partial": ["Business_and_Industrial", "Food_and_Drink"],
        "probe": ["farm", "crop", "irrigat", "livestock", "harvest", "soil"],
    },
    "Climate": {
        "description": "weather, environment, climate change, natural disasters",
        "nemo": [],
        # most science writing is not about climate
        "partial": ["Science"],
        "probe": ["climate", "monsoon", "rainfall", "glacier", "emission"],
    },
    "Tourism": {
        "description": "travel, hospitality, destinations, heritage sites, trekking",
        "nemo": [],
        # transport is mostly ordinary logistics, not tourism
        "partial": ["Travel_and_Transportation"],
        "probe": ["tourist", "trek", "hotel", "heritage", "travel"],
    },
    "Admin": {
        "description": ("government administration and public services - an office "
                        "issuing, registering, applying or announcing something"),
        # shared with Law, so every Law_and_Government row goes to stage 2
        "nemo": ["Law_and_Government"],
        "partial": [],
        "probe": ["committee", "ministry", "notice", "department", "applicant"],
    },
    "Health": {
        "description": "medicine, disease, treatment, hospitals, public health",
        "nemo": ["Health"],
        "partial": [],
        "probe": ["patient", "disease", "vaccine", "hospital", "symptom"],
    },
    "Law": {
        "description": ("legislation, courts and legal process - a statute, a case, "
                        "a judgment, a legal right or obligation"),
        "nemo": ["Law_and_Government"],
        "partial": [],
        "probe": ["court", "act ", "clause", "petition", "tribunal"],
    },
    "Education": {
        "description": "schooling, teaching, curriculum, examinations, academia",
        "nemo": ["Jobs_and_Education"],
        "partial": [],
        "probe": ["student", "curriculum", "school", "examination", "teacher"],
    },
    # The fallback. Claims no NVIDIA category: everything unclaimed lands here
    # anyway, and claiming one would make it compete with a real domain.
    "General": {
        "description": "everything else - ordinary prose that fits no domain above",
        "nemo": [],
        "partial": [],
        "probe": [],
    },
}

# which domain absorbs anything the taxonomy doesn't claim
GENERAL_DOMAIN = "General"

# ── derived from TAXONOMY - don't edit these ─────────────────────────────
DOMAINS = list(TAXONOMY)
DOMAIN_DESCRIPTIONS = {d: v["description"] for d, v in TAXONOMY.items()}
PROBES = {d: v["probe"] for d, v in TAXONOMY.items() if v["probe"]}

# invert the mapping: nemo category -> the domains that claim it
_claims = {}
_partial = set()
for _domain, _spec in TAXONOMY.items():
    for _label in _spec["nemo"]:
        _claims.setdefault(_label, []).append(_domain)
    for _label in _spec["partial"]:
        _claims.setdefault(_label, []).append(_domain)
        _partial.add(_label)

# a category exactly one domain claims outright -> take stage 1's word for it
DIRECT_MAP = {label: domains[0] for label, domains in _claims.items()
              if len(domains) == 1 and label not in _partial}
# contested between domains, or shared with ordinary prose -> stage 2 decides
# per sentence. This is also exactly where the rare domains hide, so it is
# what targeted seeding draws from.
AMBIGUOUS_LABELS = {label for label, domains in _claims.items()
                    if len(domains) > 1 or label in _partial}
# anything else falls to GENERAL_DOMAIN without stage 2 ever seeing it


def validate_taxonomy():
    """Catch a malformed taxonomy at import rather than 9 hours into a run."""
    if GENERAL_DOMAIN not in TAXONOMY:
        raise ValueError(f"GENERAL_DOMAIN {GENERAL_DOMAIN!r} is not in TAXONOMY")
    # read TAXONOMY, not the derived DOMAINS - DOMAINS is a snapshot taken at
    # import, so validating against it would pass a taxonomy swapped in later
    if len(TAXONOMY) < 2:
        raise ValueError("a taxonomy needs at least two domains")
    for domain, spec in TAXONOMY.items():
        missing = {"description", "nemo", "partial", "probe"} - set(spec)
        if missing:
            raise ValueError(f"{domain} is missing {', '.join(sorted(missing))}")
        if not spec["description"]:
            raise ValueError(f"{domain} has no description - the seeding prompt "
                             f"needs one, and a vague domain labels badly")
    if TAXONOMY[GENERAL_DOMAIN]["nemo"] or TAXONOMY[GENERAL_DOMAIN]["partial"]:
        raise ValueError(f"{GENERAL_DOMAIN} should claim no nemo categories - "
                         f"everything unclaimed already falls to it")


validate_taxonomy()

# hf weights land here instead of the user profile
os.environ.setdefault("HF_HOME", p("hf_cache"))
# 11434 is the ollama default; override if it's taken or, as on one windows
# box here, inside a reserved port exclusion range.
os.environ.setdefault("OLLAMA_HOST", os.environ.get("NLTM_OLLAMA_HOST", "127.0.0.1:11434"))
