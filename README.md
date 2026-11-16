---
license: apache-2.0
tags:
  - pytorch
  - transformer
  - reasoning
  - geometric-deep-learning
  - lie-algebra
  - gauge-theory
  - lora
  - peft
  - experimental
language:
  - en
library_name: transformers
pipeline_tag: text-generation
---

# Übermenschetien Heaven Engine + Lie-Holonomy Transformer

An experimental PyTorch scaffold that combines a local language model, LoRA fine-tuning, and geometric consistency measurements inspired by Lie algebras, gauge theory, and holonomy.

## Status

| Component | Status |
|---|---|
| LoRA integration | Implemented |
| Lie-Holonomy modules | Implemented |
| Training scripts | Included |
| Theory validation | Pending |
| Production readiness | Experimental |

The geometric scores are research instrumentation. They should not be treated as proof that a model is correct or logically consistent.

## Core idea

The project explores a geometric analogy for reasoning:

| Reasoning concept | Geometric analogue |
|---|---|
| Propositions | Points in an embedding manifold |
| Inference | Parallel transport |
| Consistency around a loop | Holonomy near the identity |
| Learned operations | Lie algebra generators |

The implementation adds holonomy, curvature, and waypoint objectives alongside model training and inference.

## Installation

```bash
git clone https://github.com/Loganwins/ubermenschetien-lht.git
cd ubermenschetien-lht

python -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

For a CUDA-specific PyTorch build, follow the current instructions at [pytorch.org](https://pytorch.org/get-started/locally/) before installing the remaining requirements.

## Configuration

Set `MODEL_PATH` in `ubermenschetien_lht.py` to either a local model directory or a compatible Hugging Face model ID, for example:

```python
MODEL_PATH = "NousResearch/Hermes-3-Llama-3.1-8B"
```

## Usage

```bash
python ubermenschetien_lht.py
```

The interactive engine supports goals, reflection, Lie-Holonomy analysis, and locally configured tools. Review the tool configuration before running commands.

## Repository map

| Path | Purpose |
|---|---|
| `ubermenschetien_lht.py` | Interactive engine and model integration |
| `lht.py` | Lie-Holonomy Transformer components |
| `train.py` | Training entry point |
| `lht_train_real.py` | Full training experiment |
| `add_lora.py` | LoRA adapter utilities |
| `LHT_Paper.md` | Theoretical background |
| `WHAT_WE_INVENTED_FULL.md` | Project notes and claims |

## Limitations

- The geometric interpretation remains experimental.
- Reported scores are internal metrics, not independent correctness guarantees.
- Large-model execution requires substantial memory and compute.
- Model outputs may be inaccurate or unsafe.

## License

Apache-2.0. See `LICENSE`.
