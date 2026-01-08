# Small Language Model (SLM) from Scratch using Transformers

## Overview

This project implements a **Small Language Model (SLM)** fully from scratch, covering the complete pipeline—from dataset preparation and tokenization to Transformer architecture design, training, and inference.

The model is trained to generate short, coherent English stories using a compact architecture with **~15 million parameters**.  
The core goal is to demonstrate that **high-quality language generation is achievable without large-scale pretrained models**, given carefully curated data and efficient training strategies.

---

## Project Goals

- Build a language model **entirely from first principles**
- Understand the internals of **Transformer-based language models**
- Show that **small models can learn grammar, syntax, and coherence**
- Avoid reliance on massive pretrained checkpoints

---

## Project Overview

- **Model Type:** Decoder-only Transformer (GPT-style)
- **Dataset:** TinyStories (synthetic short stories for young children)
- **Parameters:** ~15M
- **Training Objective:** Next-token prediction (causal language modeling)
- **Framework:** PyTorch
- **Hardware:** GPU recommended

---

## Dataset

The project uses the **TinyStories** dataset, which consists of short, grammatically correct stories generated for children aged 3–4 years.

### Dataset Statistics

- **Training samples:** ~2.1 million stories  
- **Validation samples:** ~22,000 stories  

### Motivation

The constrained vocabulary and simple narrative structure make TinyStories ideal for small models.  
This allows efficient learning of:

- Grammar
- Syntax
- Story coherence

### Loading the Dataset

The dataset is loaded directly from Hugging Face:

```python
from datasets import load_dataset  
dataset = load_dataset("roneneldan/TinyStories")

