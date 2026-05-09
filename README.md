# Brainwashing an LLM into Becoming C-3PO

Code and datasets for the experiments from the Towards Data Science article:

> *What’s the Best Way to Brainwash an LLM?*

This repo compares three supervised fine-tuning (SFT) strategies for persona injection:

- **Demonstrations** — conversational examples of the character speaking
- **First-Person Statements (FP)** — introspective identity statements (“I am C-3PO…”)
- **Synthetic Documents (SDF)** — third-person factual descriptions of the character

The goal was simple: determine which format most effectively teaches a language model to *become* a character rather than merely imitate one.

## Setup

Base model:
- `Qwen3-4B-Instruct`

Fine-tuning:
- LoRA (`r=16`, `alpha=32`)
- TRL + PEFT
- Single A40 GPU (RunPod)

Each strategy was trained on:
- 500 synthetic examples
- identical hyperparameters
- 3 epochs

## Evaluation

The models were evaluated using:
- held-out perplexity across all formats
- trait tagging (anxiety, verbosity, protocol behaviour, odds calculations, etc.)
- LLM-as-Judge scoring
- response length analysis

## Main Result

First-person identity statements generalized best across formats.

In practice:
- demonstrations taught behaviour,
- synthetic documents taught facts,
- first-person statements taught identity.
