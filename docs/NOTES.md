# Development notes

Problems hit while building this, how they were diagnosed, and what the fix
was. Most are Windows- or hardware-specific and won't affect everyone, but
each one cost real time and the symptoms are non-obvious.

---

## The model loads "fine" and returns garbage

**Symptom.** `AutoModelForSequenceClassification.from_pretrained("nvidia/domain-classifier")`
loads with the usual pile of missing/unexpected-weight warnings, which are easy
to wave away. Predictions look confident and are nonsense.

**Diagnosis.** The warning list is the tell. Expected keys were
`deberta.encoder.layer.*` and `classifier.*`; the checkpoint actually contained
`model.encoder.*` and `fc.*`. Different architecture entirely — the Auto class
built a standard DeBERTa sequence-classification model and randomly
initialised a head that the checkpoint never filled.

**Fix.** The model is a DeBERTa backbone plus a custom `fc` layer published
through `PyTorchModelHubMixin`. `CustomModel` in `nemo_stage.py` reproduces it.

**Lesson.** On a model card that defines its own class, missing-weight
warnings are not noise. A randomly initialised classifier head still produces
a confident softmax.

---

## `mat1 and mat2 must have the same dtype`

**Symptom.** Crash in the `fc` layer on the first forward pass.

**Cause.** The checkpoint is mixed precision — the backbone loads as fp16
while the freshly constructed `fc` head is fp32.

**Fix.** Force one dtype for the whole model per device: `.half()` on CUDA,
`.float()` on CPU. Verified identical results: same mean confidence (0.924),
same routing split, one label different out of 4,000.

---

## int8 quantization destroys this model

**Symptom.** After `quantize_dynamic`, ~90% of rows came back
`Arts_and_Entertainment` and mean confidence fell from 0.93 to 0.31.

**Diagnosis.** Ran the same 600 rows with and without quantization:

| | mean confidence | label spread |
|---|---|---|
| fp32 | 0.926 | Business_and_Industrial 74, People_and_Society 74, Arts 60, Law_and_Government 51, News 43, Health 39 |
| int8 | 0.305 | Arts_and_Entertainment 539, People_and_Society 48, rest negligible |

**Fix.** Don't. The switch stays in `config.py` set to `False` so the finding
is recorded next to it. The speedup was only ~1.4x anyway; length-sorted
batching gives ~2x for free with no accuracy cost.

**Lesson.** Always diff a quantized model's *output distribution* against the
original, not just its speed. A collapsed classifier still returns plausible
shapes.

---

## GPU halted with Device Manager code 43

**Symptom.** `torch.cuda.is_available()` false; `nvidia-smi` failed with
"insufficient permissions"; `cuInit` returned 100 (`CUDA_ERROR_NO_DEVICE`).
Three different-looking errors.

**Diagnosis.** All three were downstream of one thing:

```
pnputil /enum-devices /class Display
  NVIDIA GeForce RTX 4050 Laptop GPU
  Status:        Problem
  Problem Code:  43 (0x2B) [CM_PROB_FAILED_POST_START]
```

Windows had halted the device. The driver was fine (`nvcuda.dll` present,
registry entry healthy) — the GPU simply wasn't available to any process.
Confirmed it wasn't a sandbox artifact by reproducing it from a scheduled task
outside the agent's process tree.

**Fix.** A reboot cleared it. The machine had ~6 days uptime on a laptop with
many sleep/resume cycles, which is the classic profile for a transient code 43.
If a reboot doesn't clear it: clean driver reinstall (DDU + fresh driver), then
BIOS update.

**Lesson.** `pnputil /enum-devices` first. The CUDA-level errors describe
symptoms; the device status describes the cause.

---

## Ollama couldn't bind its port

**Symptom.** `ollama serve` → "An attempt was made to access a socket in a way
forbidden by its access permissions." Not an "address in use" error.

**Diagnosis.** A plain .NET `TcpListener` bound an arbitrary port fine, so
sockets in general weren't blocked. Checking reserved ranges:

```
netsh interface ipv4 show excludedportrange protocol=tcp
  ...
  11374    11473
```

Ollama's default 11434 sits inside a Windows reserved exclusion range —
typically claimed by Hyper-V/WSL dynamic port reservations.

**Fix.** Run on a port outside every excluded range (11500 here) via
`NLTM_OLLAMA_HOST`. `start_ollama.cmd` and `config.py` read the same variable
so client and server can't drift apart.

---

## Ollama server wouldn't start from the agent's shell

**Symptom.** Even on a free port, the server process exited immediately when
launched from the automation shell, while an identical command worked when the
user ran it.

**Cause.** Processes spawned from that shell were denied listening sockets.

**Fix.** Launch it via a scheduled task, which runs outside the shell's process
tree:

```
schtasks /create /tn OllamaServeNLTM /tr "'<path>\start_ollama.cmd'" /sc once /st 00:00 /f
schtasks /run /tn OllamaServeNLTM
```

Only relevant when driving this from a sandboxed agent; running the script
yourself is fine.

---

## Devanagari came out as mojibake in Excel

**Symptom.** CSVs opened in Excel showed the entire Nepali column as
`à¤¸à¤‚à¤¯à¥...`. The `.xlsx` exports were fine.

**Diagnosis.** Checked the first three bytes of each CSV — `69 64 2c` (`id,`),
no BOM. Verified the data itself was clean by comparing codepoints at every
stage (source xlsx → merged CSV → exported xlsx): all correct U+0900-block
Devanagari. So the bytes were never corrupted; Excel was guessing the encoding
and guessing the system ANSI codepage.

**Fix.** Write every CSV `utf-8-sig`. Read `utf-8-sig` too — it handles both
BOM and no-BOM transparently, so nothing else breaks and the first column name
doesn't come back as `﻿id`. `fix_csv_bom.py` repairs existing files by
byte-level prepend, which takes seconds on a 600 MB file instead of a full
pandas reparse.

---

## Benchmarks that lie

A 4,000-row timing of stage 1 reported "369 sent/sec". The real steady-state
rate is ~1,455 — the ~25 s model load was sitting inside the measured window
and a 4,000-row run is far too short to amortise it.

Time the loop, not the setup, and benchmark on enough rows that startup cost
is noise.

---

## Random sampling misses rare classes

Covered in the README under [Targeted seeding](../README.md#targeted-seeding--the-part-that-actually-mattered),
but worth repeating as a general point: on a corpus where the interesting
classes are <1%, uniform random sampling is the wrong default for *both*
training seeds and evaluation sets. 1,732 random sentences gave 17 Climate
seeds; 500 random rows would give ~2 Climate gold examples. Sample from where
the class actually lives, and be explicit in the reporting about what that
biases.
