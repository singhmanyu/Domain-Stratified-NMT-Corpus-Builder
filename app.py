"""Gradio front end for the two-stage domain classifier.

    set NLTM_DATA_DIR=E:\\Machine Learning\\Dommain Classifier
    python app.py

Serves on http://127.0.0.1:7860. One sentence at a time, or a csv/xlsx in
and a zip of per-domain excel files out.
"""

import html
import os
import shutil
import time
import zipfile

import gradio as gr
import pandas as pd

import config as cfg
import parallel_input
import predict

UI_DIR = cfg.p("ui_out")
# the deberta pass is the cost here. past this, it's the batch pipeline's
# job - it checkpoints and resumes, which a web request can't.
MAX_UPLOAD_ROWS = 200_000
# bucket for rows with no english text to classify - they still come back
UNCLASSIFIED = "Unclassified"

DOMAIN_BLURB = {
    "Tech": "computing, telecom, electronics",
    "Agriculture": "farming, crops, livestock, food production",
    "Climate": "weather, environment, climate change",
    "Tourism": "travel, hospitality, destinations",
    "Admin": "government administration, public services",
    "Health": "medicine, disease, healthcare",
    "Law": "legislation, courts, legal process",
    "Education": "schooling, teaching, academia",
    "General": "everything else",
}

EXAMPLES = [
    "The farmer planted rice in the terraced fields of Kavre.",
    "The Supreme Court overturned the lower court's verdict on Tuesday.",
    "Install the latest security patch before restarting the server.",
    "Monsoon rainfall has grown more erratic across the Terai over the past decade.",
    "Trekkers must register at the checkpoint before entering the conservation area.",
    "The ward office issues birth certificates within three working days.",
]

# roles, not raw hex, so light/dark swap in one place. one blue hue for the
# score bars: that list is a single series ranked by magnitude, not nine
# identities, so a categorical palette would encode something that isn't there.
CSS = """
:root, .gradio-container {
  --surface-1: #fcfcfb; --surface-2: #f4f3f0; --border-1: #e4e2dd;
  --text-1: #0b0b0b; --text-2: #52514e; --text-3: #85837c;
  --series-1: #2a78d6; --series-wash: #e8f0fc;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]), :root:not([data-theme="light"]) .gradio-container {
    --surface-1: #1a1a19; --surface-2: #232321; --border-1: #34332f;
    --text-1: #ffffff; --text-2: #c3c2b7; --text-3: #8d8b81;
    --series-1: #3987e5; --series-wash: #16253a;
  }
}
:root[data-theme="dark"], :root[data-theme="dark"] .gradio-container {
  --surface-1: #1a1a19; --surface-2: #232321; --border-1: #34332f;
  --text-1: #ffffff; --text-2: #c3c2b7; --text-3: #8d8b81;
  --series-1: #3987e5; --series-wash: #16253a;
}
.gradio-container { max-width: 1120px !important; }

#masthead { padding: 26px 0 6px; border-bottom: 1px solid var(--border-1); margin-bottom: 18px; }
#masthead h1 { font-size: 1.65rem; font-weight: 640; letter-spacing: -0.021em;
  margin: 0 0 6px; color: var(--text-1); }
#masthead p { margin: 0; color: var(--text-2); font-size: 0.925rem;
  line-height: 1.55; max-width: 64ch; }

.card { background: var(--surface-1); border: 1px solid var(--border-1);
  border-radius: 14px; padding: 20px 22px; }
.verdict { display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; }
.verdict .dom { font-size: 2.1rem; font-weight: 660; letter-spacing: -0.028em;
  color: var(--text-1); line-height: 1.1; }
.verdict .conf { font-variant-numeric: tabular-nums; font-size: 0.9rem; color: var(--text-2); }
.blurb { margin: 8px 0 0; color: var(--text-2); font-size: 0.9rem; }
.meta { margin-top: 16px; padding-top: 14px; border-top: 1px solid var(--border-1);
  display: grid; grid-template-columns: auto 1fr; gap: 6px 16px;
  font-size: 0.82rem; color: var(--text-2); }
.meta dt { color: var(--text-3); }
.meta dd { margin: 0; font-variant-numeric: tabular-nums; }
.pill { display: inline-block; padding: 2px 9px; border-radius: 999px;
  background: var(--series-wash); color: var(--series-1);
  font-size: 0.74rem; font-weight: 580; letter-spacing: 0.01em; }

.bars { margin-top: 18px; }
.bars h4 { margin: 0 0 2px; font-size: 0.82rem; font-weight: 600; color: var(--text-1); }
.bars .sub { margin: 0 0 12px; font-size: 0.76rem; color: var(--text-3); }
.bar-row { display: grid; grid-template-columns: 88px 1fr 46px; align-items: center;
  gap: 10px; padding: 3px 0; font-size: 0.8rem; color: var(--text-2); }
.bar-track { height: 9px; background: var(--surface-2); border-radius: 0 5px 5px 0; }
.bar-fill { height: 9px; background: var(--series-1); border-radius: 0 4px 4px 0; min-width: 2px; }
.bar-row .val { text-align: right; font-variant-numeric: tabular-nums; color: var(--text-3); }
.bar-row.top { color: var(--text-1); font-weight: 580; }
.bar-row.top .val { color: var(--text-1); }

.placeholder { color: var(--text-3); font-size: 0.9rem; padding: 36px 0; text-align: center; }
footer { display: none !important; }
"""

THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.stone,
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "monospace"],
    radius_size=gr.themes.sizes.radius_md,
)

EMPTY_CARD = ('<div class="card"><div class="placeholder">'
              "Enter a sentence to classify.</div></div>")


# ── single sentence ──────────────────────────────────────────────────────

def _bars_html(scores, decided_here):
    if not scores:
        return ""
    rows = sorted(scores.items(), key=lambda kv: -kv[1])
    sub = ("what the fasttext model saw" if decided_here
           else "reference only - stage 1 made this call")
    out = [f'<div class="bars"><h4>Stage 2 scores</h4><p class="sub">{sub}</p>']
    for i, (domain, score) in enumerate(rows):
        cls = "bar-row top" if i == 0 else "bar-row"
        out.append(
            f'<div class="{cls}" title="{html.escape(domain)}: {score:.3f}">'
            f"<span>{html.escape(domain)}</span>"
            f'<span class="bar-track"><span class="bar-fill" '
            f'style="width:{max(score, 0.004) * 100:.1f}%"></span></span>'
            f'<span class="val">{score:.2f}</span></div>'
        )
    out.append("</div>")
    return "".join(out)


def classify_one(sentence):
    sentence = (sentence or "").strip()
    if not sentence:
        return EMPTY_CARD

    pipe = predict.get_pipeline()
    row = pipe.classify_one(sentence)
    stage2 = row["stage"] == predict.STAGE_FINEGRAIN

    return "".join([
        '<div class="card"><div class="verdict">',
        f'<span class="dom">{html.escape(str(row["domain"]))}</span>',
        f'<span class="conf">{float(row["confidence"]):.1%} confidence</span>',
        f'<span class="pill">{html.escape(row["stage"])}</span>',
        "</div>",
        f'<p class="blurb">{html.escape(DOMAIN_BLURB.get(row["domain"], ""))}</p>',
        '<dl class="meta">',
        f'<dt>stage 1 label</dt><dd>{html.escape(str(row["nemo_label"]))}</dd>',
        f'<dt>stage 1 score</dt><dd>{float(row["nemo_score"]):.3f}</dd>',
        f'<dt>decided by</dt><dd>{"fasttext" if stage2 else "stage 1, taken as-is"}</dd>',
        "</dl>",
        _bars_html(pipe.fasttext_scores(sentence), stage2),
        "</div>",
    ])


# ── batch file ───────────────────────────────────────────────────────────

def _domain_zip(out, stem, work_dir):
    """One xlsx per domain plus a summary sheet, zipped.

    Splits a domain across parts at the same threshold the batch exporter
    uses - excel technically holds a million rows but gets unusable long
    before that.
    """
    staging = os.path.join(work_dir, f"{stem}_domains")
    shutil.rmtree(staging, ignore_errors=True)
    os.makedirs(staging, exist_ok=True)

    part_rows = cfg.XLSX_PART_ROWS
    written = []
    for domain, group in out.groupby("domain", sort=True):
        name = str(domain).lower()
        if len(group) <= part_rows:
            parts = [(f"{name}.xlsx", group)]
        else:
            parts = [
                (f"{name}_part{i + 1:02d}.xlsx", group.iloc[i * part_rows:(i + 1) * part_rows])
                for i in range((len(group) + part_rows - 1) // part_rows)
            ]
        for fname, chunk in parts:
            chunk.to_excel(os.path.join(staging, fname), index=False,
                           sheet_name=str(domain)[:31])
            written.append(fname)

    summary = out["domain"].value_counts().rename_axis("domain").reset_index(name="rows")
    summary["share %"] = (summary["rows"] / len(out) * 100).round(2)
    summary.to_excel(os.path.join(staging, "_summary.xlsx"), index=False)
    written.append("_summary.xlsx")

    zip_path = os.path.join(work_dir, f"{stem}_domains.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for fname in written:
            z.write(os.path.join(staging, fname), arcname=fname)

    shutil.rmtree(staging, ignore_errors=True)
    return zip_path


def classify_file(upload, text_col, out_format, progress=gr.Progress()):
    if upload is None:
        raise gr.Error("upload a csv or xlsx first")

    path = upload if isinstance(upload, str) else upload.name
    progress(0.05, desc="reading file")
    if os.path.splitext(path)[1].lower() in (".xlsx", ".xlsm", ".xls"):
        src = pd.read_excel(path)
    else:
        src = pd.read_csv(path, encoding=cfg.CSV_ENCODING)

    if src.empty:
        raise gr.Error("that file has no rows")

    # blank box means work it out from the file: header names first, then
    # which column actually holds devanagari. an id column is kept if there
    # is one. typing a column name overrides all of it.
    text_col = (text_col or "").strip()
    if not text_col:
        try:
            text_col = parallel_input.detect_columns(src)["english"]
        except parallel_input.DetectionError as e:
            raise gr.Error(str(e))
    if text_col not in src.columns:
        raise gr.Error(f"no column '{text_col}' - this file has: "
                       f"{', '.join(map(str, src.columns))}")
    if len(src) > MAX_UPLOAD_ROWS:
        raise gr.Error(f"{len(src):,} rows is over the {MAX_UPLOAD_ROWS:,} limit for the "
                       f"ui - use domain_classifier_nltm.py for a full corpus")
    progress(0.15, desc=f"classifying {len(src):,} rows")
    started = time.time()
    out = predict.get_pipeline().classify(src[text_col].tolist())

    # classify() returns one row per input in order, so the other source
    # columns - the translation side - line up positionally and the output is
    # still a usable parallel corpus
    assert len(out) == len(src), "pipeline changed row count"
    out = out.rename(columns={cfg.TEXT_COL: text_col})
    for col in src.columns:
        if col != text_col:
            out[col] = src[col].values

    # blank rows have no domain, and groupby drops nulls - without this they
    # would quietly go missing from the per-domain export. the uploader gets
    # back every row they sent, in a file that says why.
    out["domain"] = out["domain"].fillna(UNCLASSIFIED)

    progress(0.9, desc="writing output")
    os.makedirs(UI_DIR, exist_ok=True)
    stem = os.path.splitext(os.path.basename(path))[0]
    if out_format.startswith("zip"):
        dest = _domain_zip(out, stem, UI_DIR)
    elif out_format.startswith("one xlsx"):
        dest = os.path.join(UI_DIR, f"{stem}_classified.xlsx")
        out.to_excel(dest, index=False)
    else:
        dest = os.path.join(UI_DIR, f"{stem}_classified.csv")
        out.to_csv(dest, index=False, encoding=cfg.CSV_ENCODING)

    counts = out["domain"].value_counts().rename_axis("domain").reset_index(name="rows")
    counts["share %"] = (counts["rows"] / len(out) * 100).round(2)

    elapsed = max(time.time() - started, 1e-9)
    stage2_share = (out["stage"] == predict.STAGE_FINEGRAIN).mean() * 100
    note = (f"**{len(out):,} rows** in {elapsed:.1f}s &nbsp;·&nbsp; "
            f"{len(out) / elapsed:,.0f} rows/s &nbsp;·&nbsp; "
            f"{stage2_share:.1f}% needed stage 2 &nbsp;·&nbsp; "
            f"{len(counts)} domains present")

    return dest, counts, out.head(100), note


# ── layout ───────────────────────────────────────────────────────────────

with gr.Blocks(title="NLTM Domain Classifier") as demo:
    gr.HTML(
        '<div id="masthead"><h1>NLTM Domain Classifier</h1>'
        "<p>Two stages. <code>nvidia/domain-classifier</code> labels the sentence; "
        "anything its taxonomy can't place cleanly goes to a fastText model trained "
        "on this corpus. Only the English side is ever read.</p></div>"
    )

    with gr.Tab("Classify a sentence"):
        with gr.Row(equal_height=False):
            with gr.Column(scale=4):
                sentence = gr.Textbox(label="English sentence", lines=4, max_lines=8,
                                      placeholder="Type or paste a sentence, then press Enter...")
                go = gr.Button("Classify", variant="primary", size="lg")
                gr.Examples(EXAMPLES, inputs=sentence, label="Try one")
            with gr.Column(scale=5):
                result = gr.HTML(EMPTY_CARD)

        go.click(classify_one, sentence, result)
        sentence.submit(classify_one, sentence, result)

    with gr.Tab("Classify a file"):
        gr.Markdown(
            f"A csv or xlsx of parallel data. The English column is worked out from the "
            f"headers - `english`/`target`, `source`/`target`, `src`/`tgt`, or the "
            f"language's own name - or, when the headers don't say, from which column "
            f"is written in another script. An `id` "
            f"column is kept if there is one. Every other column is carried through "
            f"untouched and stays aligned, so the output is still a parallel corpus. "
            f"Up to {MAX_UPLOAD_ROWS:,} rows.\n\n"
            f"The default output is a **zip holding one Excel file per domain**, plus "
            f"`_summary.xlsx` with the row counts."
        )
        with gr.Row(equal_height=False):
            upload = gr.File(label="csv / xlsx", file_types=[".csv", ".xlsx", ".xlsm"], scale=3)
            with gr.Column(scale=2):
                text_col = gr.Textbox(label="English column",
                                      placeholder="auto-detect",
                                      info="leave blank to detect it")
                out_format = gr.Radio(["zip of per-domain xlsx", "one xlsx", "one csv"],
                                      value="zip of per-domain xlsx", label="Output")
                run = gr.Button("Classify file", variant="primary")

        note = gr.Markdown()
        with gr.Row(equal_height=False):
            download = gr.File(label="Download", scale=3)
            counts = gr.Dataframe(label="Domain distribution", interactive=False, scale=2)
        preview = gr.Dataframe(label="First 100 rows", interactive=False, wrap=True)

        run.click(classify_file, [upload, text_col, out_format],
                  [download, counts, preview, note])

    with gr.Tab("About"):
        gr.Markdown(
            "### Domains\n"
            + "\n".join(f"- **{d}** - {DOMAIN_BLURB[d]}" for d in cfg.DOMAINS)
            + "\n\n### How a sentence is routed\n"
            "1. Stage 1 runs `nvidia/domain-classifier` (deberta-v3-base, 26 labels).\n"
            "2. Labels with no overlap with this taxonomy go straight to **General**; "
            "the four that map 1:1 - Health, Jobs_and_Education, "
            "Computers_and_Electronics, Internet_and_Telecom - are taken as-is.\n"
            f"3. Ambiguous labels, and anything under {cfg.CONF_THRESHOLD} stage-1 "
            "confidence, go to the fastText model, trained on llama3.2-labelled seed "
            "sentences drawn from this corpus.\n\n"
            "The score bars always show stage 2's view, but they are the actual "
            f"decision only when the card reads `{predict.STAGE_FINEGRAIN}`."
        )


if __name__ == "__main__":
    print(f"data dir: {cfg.DATA_DIR}")
    print("loading models...")
    predict.get_pipeline()
    # gradio 6 moved theme/css off the Blocks constructor onto launch()
    demo.launch(theme=THEME, css=CSS, server_name="127.0.0.1",
                server_port=7860, inbrowser=True)
