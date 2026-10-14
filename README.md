# LLM Fine-Tuning for English → Greek Translation

Experiments in fine-tuning open LLMs (Mistral-7B, Llama-2, DeepSeek, GPT-2 baseline) with LoRA/PEFT for English→Greek machine translation on OPUS-100, plus evaluation and export to Ollama.

## Tech stack
- Python · PyTorch · Hugging Face Transformers / PEFT
- Ollama

## Features
- LoRA fine-tuning notebooks and scripts
- BLEU-based evaluation (`eval_ollama_mt.py`)
- Comparative analysis of models
- Export guide for running the model locally with Ollama

## Getting started
Open `notebooks/finetune_mistral.ipynb` (GPU recommended), or run the training scripts. See `README_OLLAMA.md` for local deployment.

---
Part of the **SIGMA LAB** project portfolio.
