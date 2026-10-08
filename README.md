<div align="center">

# CMAB: A Cross-Modal Activation Balance-Aware Pruning Method for MLLMs

[![Paper](https://img.shields.io/badge/Paper-ESWA%20Under%20Review-blue)](#)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Models-orange.svg)]()
[![ModelScope](https://img.shields.io/badge/🤖%20ModelScope-Models-purple.svg)]()
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-orange.svg)]()
</div>
After the paper is published, we will open source all the code.
<br/>
---

## 🌟 Highlights

- **Theoretical Foundation**: We reveal that the fundamental cause of weight evaluation distortion in MLLM pruning is the heterogeneous distribution between visual and textual features under the OBD framework.
- **Cross-Modal Feature Decoupling (CFD)**: Eliminates inter-modality magnitude biases and constructs a fair relative metric space for activations.
- **Dynamic Information Fusion**: Employs a Linear Routing Functional (LRF) and Sparsity-Aware Temperature Modulation Mechanism (TMM) to achieve fine-grained, depth-adaptive modal balancing across Transformer layers.
- **Training-Free & Fast**: Achieves performance comparable to computationally expensive second-order methods (like SparseGPT) using only first-order computational cost.

<div align="center">
    <img width="13027" height="3248" alt="Figure2_01" src="https://github.com/user-attachments/assets/850ff315-cb35-4810-872b-099232105199" /> 
    <p><em>Figure 1: The overall architecture of the Cross-Modal Activation Balance-Aware (CMAB) Pruning Method.</em></p>
</div>

---

## 📊 Main Results

CMAB demonstrates highly competitive parameter compression across multiple MLLM architectures([LLaVA](https://github.com/haotian-liu/LLaVA),[Qwen](https://github.com/QwenLM/Qwen) ) on both multimodal cognitive reasoning and text-only language modeling tasks.

**Zero-shot performance of LLaVA-v1.6-Mistral-7B at 50% sparsity:**

| Method | TextVQA | ChartQA | MME | POPE | MMBench | Rel_ACC (%) | WikiText | C4 | Rel_PPL
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Dense | 65.60 | 53.08 | 37.69/75.30 | 88.17 | 78.21 | 100.00 | 6.82 | 10.07 | 100 |
| Wanda | 63.63 | 44.32 | 38.21/67.84 | 88.33 | 76.59 | 95.00 | 8.24 | 12.10 | 83.00 |
| SparseGPT | 61.59 | 46.76 | 37.46/70.24 | 86.83 | 76.77 | 95.20 | 8.50 | 12.32 | 80.99 |
| **CMAB (Ours)** | **63.40** | **46.72** | **40.45/69.25** | **88.50** | **77.92** | **97.32** | **8.15** | **11.97** | **83.90** |

---

## 🚀 Quick Start

### 1. Environment Setup

We recommend using Conda to manage the environment. The codebase has been strictly tested on an **NVIDIA RTX 4080 SUPER** GPU with the following core dependencies:
- **Python**: 3.10
- **PyTorch**: 2.1.2+cu121
- **Transformers**: 4.44.2
- **Flash Attention**: 2.5.8

Clone this repository and set up the environment:

```bash
git clone https://github.com/start-ship-it/CMAB-Pruning.git
cd CMAB-Pruning

# Create and activate a conda environment
conda create -n cmab python=3.10 -y
conda activate cmab

# Install PyTorch (CUDA 12.1)
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# Install exact versions of core dependencies to ensure reproducibility
pip install transformers==4.44.2
pip install flash-attn==2.5.8 --no-build-isolation

# Install remaining dependencies
pip install -e .
```
### 2. Prepare Calibration Data

CMAB requires a hybrid calibration dataset comprising both text-only and vision-language samples to accurately balance cross-modal activations. To ensure statistical robustness, evaluations are conducted across five distinct random seeds(`42`, `2026`, `3407`, `0`, `1024`).

**Vision-Language Datasets:**
- [ChartQA](https://github.com/vis-nlp/ChartQA) | [DocVQA](https://github.com/anisha2102/docvqa) | [LLaVA-COCO](https://huggingface.co/datasets/lmms-lab-encoder/llava-bench-coco) | [ScienceQA](https://scienceqa.github.io/) | [ShareGPT4V](https://github.com/ShareGPT4Omni/ShareGPT4V) | [TextCaps](https://huggingface.co/datasets/lmms-lab-encoder/TextCaps)

**Text-Only Datasets:**
- [ShareGPT (Clean)](https://huggingface.co/datasets/philschmid/sharegpt-raw/tree/main) | [Alpaca](https://huggingface.co/datasets/shibing624/alpaca-zh/tree/main) | [WikiText-2](https://huggingface.co/datasets/mindchain/wikitext2)

*Note: Data extraction and merging scripts are provided in the `dataset/` directory to streamline this process.*

### 3. Run CMAB Pruning

Execute the corresponding Python script for your target architecture. For example, to prune the `LLaVA-v1.6-Mistral-7B` model, run the following command:

```bash
python cmab_llava_v1.6_mistral_7B_Ada.py 
```

### 4. Evaluation

We rely on standardized, open-source evaluation frameworks to ensure rigorous and reproducible zero-shot benchmarking across all tasks.

- For **Qwen** Architectures: We utilize [VLMEvalKit](https://github.com/open-compass/VLMEvalKit) for specialized assessmentsmodel.

- For **LLaVA** Architectures: We utilize [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) for comprehensive evaluation on multimodal benchmarks.
