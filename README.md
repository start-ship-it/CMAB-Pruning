<div align="center">

# CMAB: A Cross-Modal Activation Balance-Aware Pruning Method for MLLMs

[![Paper](https://img.shields.io/badge/Paper-ESWA%20Under%20Review-blue)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8+-green.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)]()


# 🌟 Highlights

- **Theoretical Foundation**: We reveal that the fundamental cause of weight evaluation distortion in MLLM pruning is the heterogeneous distribution between visual and textual features under the OBD framework.
- **Cross-Modal Feature Decoupling (CFD)**: Eliminates inter-modality magnitude biases and constructs a fair relative metric space for activations.
- **Dynamic Information Fusion**: Employs a Linear Routing Functional (LRF) and Sparsity-Aware Temperature Modulation Mechanism (TMM) to achieve fine-grained, depth-adaptive modal balancing across Transformer layers.
- **Training-Free & Fast**: Achieves performance comparable to computationally expensive second-order methods (like SparseGPT) using only first-order computational cost.

<div align="center">
    <img src="assets/graphical_abstract.png" alt="CMAB Framework" width="100%">
    <p><em>Figure 1: The overall architecture of the Cross-Modal Activation Balance-Aware (CMAB) Pruning Method.</em></p>
</div>

---

## 📊 Main Results

CMAB demonstrates highly competitive parameter compression across multiple MLLM architectures (LLaVA-1.5, LLaVA-v1.6-Mistral, Qwen-VL series) on both multimodal cognitive reasoning and text-only language modeling tasks.

**Zero-shot performance of LLaVA-v1.6-Mistral-7B at 50% sparsity:**

| Method | TextVQA | ChartQA | MME | POPE | MMBench | Rel_ACC (%) | WikiText |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Dense | 65.60 | 53.08 | 37.69/75.30 | 88.17 | 78.21 | 100.00 | 6.82 |
| Wanda | 63.63 | 44.32 | 38.21/67.84 | 88.33 | 76.59 | 95.00 | 8.24 |
| SparseGPT | 61.59 | 46.76 | 37.46/70.24 | 86.83 | 76.77 | 95.20 | 8.50 |
| **CMAB (Ours)** | **63.40** | **46.72** | **40.45/69.25** | **88.50** | **77.92** | **97.32** | **8.15** |

<details>
<summary>👉 Click to expand more results across varying sparsity ratios (40% to 70%)</summary>

<div align="center">
    <img src="assets/results_curves.png" alt="Performance Curves" width="80%">
    <p><em>CMAB exhibits a highly stable and resilient degradation trajectory, maintaining significant advantages at extreme sparsities (e.g., 70%).</em></p>
</div>

</details>

---

## 🚀 Quick Start

### 1. Installation

Clone this repository and install the dependencies:

```bash
git clone [https://github.com/start-ship-it/CMAB-Pruning.git](https://github.com/start-ship-it/CMAB-Pruning.git)
cd CMAB-Pruning

# Create a conda environment
conda create -n cmab python=3.10 -y
conda activate cmab

# Install pure dependencies via setup.py
pip install -e .


















CMAB: A Cross-Modal Activation Balance-Aware Pruning Method for Multimodal Large Language Models
<img width="13027" height="3248" alt="Figure2_01" src="https://github.com/user-attachments/assets/850ff315-cb35-4810-872b-099232105199" />

After the paper is published, we will open source all the code.


Pruning:   
cmab_llava_v1.6_mistral_7B_Ada.py [Model Pruning Code]    
Eval:  
Qwen model assessment Tools [VLMEvalKit](https://github.com/open-compass/VLMEvalKit)  
LLaVA model assessment Tools [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)  
LLaVA: Large Language and Vision Assistant [LLaVA](https://github.com/haotian-liu/LLaVA)  
Qwen [Qwen](https://github.com/QwenLM/Qwen)   
dataset:  
seed={42, 2026, 3407, 0, 1024}  
Image and Text Dataset [ChartQA](https://github.com/vis-nlp/ChartQA) [DocVQA](https://github.com/anisha2102/docvqa) [LLaVA_coco](https://huggingface.co/datasets/lmms-lab-encoder/llava-bench-coco) [ScienceQA](https://scienceqa.github.io/) [ShareGPT4V](https://github.com/ShareGPT4Omni/ShareGPT4V) [TextCaps](https://huggingface.co/datasets/lmms-lab-encoder/TextCaps)   
Text-only Dataset [sharegpt_clean](https://huggingface.co/datasets/philschmid/sharegpt-raw/tree/main) [alpaca_calib](https://huggingface.co/datasets/shibing624/alpaca-zh/tree/main) [wikitext-2](https://huggingface.co/datasets/mindchain/wikitext2)
