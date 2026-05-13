# Hierarchical Adapter Fusion (HAF)

A continual learning framework for large language models that enables high speed sequential task adaptation.

## Overview

Standard fine-tuning causes LLMs to forget previously learned tasks when trained on new ones — a problem known as catastrophic forgetting. HAF addresses this through three combined mechanisms:

- **Variational Hypernetwork** — generates task-specific LoRA adapters conditioned on task context, rather than maintaining separate adapter sets per task
- **FAISS-Based Hierarchical Memory Retrieval** — efficiently retrieves relevant past adapter configurations using approximate nearest-neighbor search over a hierarchical memory store
- **Evolutionary Candidate Selection** — selects and refines adapter candidates across tasks using an evolutionary optimization strategy

Together these allow the model to learn new tasks while retaining performance on prior ones, without storing full copies of task-specific weights.

## Results

Evaluated on Llama 3.2 1B across 6 sequential tasks (math/code only), showed a reduction in adaptation time by ~15x as compared to Self-Adapting Language Models

## Recognition

- 1st Place, Brevard County District Science Fair (2026)
- Merit Award, Florida State Science Fair (2026)

## Running the Notebook

This project runs on Kaggle with an NVIDIA P100 GPU.

**Setup:**
1. Upload the notebook to [Kaggle](https://kaggle.com)
2. Set your HuggingFace token as a Kaggle secret: `HF_TOKEN`
3. Enable GPU accelerator (P100)

**Important:** The full training pipeline will exceed Kaggle's session time limit. Checkpointing is built into the notebook, so verify that checkpoint saving and loading works correctly on a short run before launching the full training loop.

## Dependencies

- PyTorch
- HuggingFace Transformers
- FAISS
- NumPy / pandas

All dependencies are available in the default Kaggle environment.

## Notes

- Code is unpolished
- Generative AI was used to assist development
- This is an independent research project

## Background

HAF was developed as independent ML research exploring parameter-efficient continual learning for LLMs. The core motivation: existing continual learning frameworks either require significant compute overhead or sacrifice backward transfer performance. HAF targets both constraints simultaneously through hierarchical memory and hypernetwork-driven adapter generation.

Research is ongoing. Targeting submission to a conference after additional ablations.
