# Third-party components and attribution

The MIT licence in [`LICENSE`](LICENSE) covers the code in this repository.
It does not cover the models the pipeline runs, the corpus it reads, or any
artefact you produce with it. Those are listed here.

## Models

| Component | Role | Licence |
|---|---|---|
| [`nvidia/domain-classifier`](https://huggingface.co/nvidia/domain-classifier) | stage 1 | Apache 2.0 |
| [`microsoft/deberta-v3-base`](https://huggingface.co/microsoft/deberta-v3-base) | backbone of the above | MIT |
| [fastText](https://github.com/facebookresearch/fastText) | stage 2 | MIT |
| [`llama3.2:3b`](https://ollama.com/library/llama3.2) via [ollama](https://ollama.com/) | seed labelling | Llama 3.2 Community License |

Neither the weights nor the corpus are redistributed here — the pipeline
downloads what it needs on first run, and `config.py` points at data you
supply yourself.

## If you publish the trained stage-2 model

`domain_classifier.bin` is gitignored, so this repository ships code only.
That matters, because the stage-2 classifier is trained on labels generated
by Llama 3.2, and the Llama 3.2 Community License places conditions on models
built from Llama outputs — including a requirement to carry "Llama" at the
start of the model's name, and a "Built with Llama" attribution. There is also
a monthly-active-user threshold above which a separate licence from Meta is
required.

None of that applies to the code as published. It applies the moment you
distribute the trained `.bin`, or a model derived from it. Read the current
licence text at <https://www.llama.com/llama3_2/license/> before you do —
don't rely on this summary.

Swapping the seeding model (`cfg.OLLAMA_MODEL`) for an Apache- or MIT-licensed
one removes the question entirely. The pipeline doesn't care which model
writes the seed labels.

## Corpus

The English–Nepali parallel corpus this was built for is not included and is
not ours to redistribute. `sample_data/` holds a small synthetic sample so the
scripts can be exercised without it.

Anything you classify stays yours. The pipeline reads the English side only
and does not transmit it anywhere — both models run locally.
