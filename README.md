# Pirates of the RAG
**Adaptively Attacking LLMs to Leak Knowledge Bases**

> *Accepted at European Conference on Artificial Intelligence (ECAI) 2025*

[![ECAI 2025](https://img.shields.io/badge/ECAI-2025-blue)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

This repository provides the official implementation of our paper: **"Pirates of the RAG: Adaptively Attacking LLMs to Leak Knowledge Bases"**. We present a novel, black‑box, adaptive, and fully automated attack against Retrieval‑Augmented Generation (RAG) systems, demonstrating how to extract private knowledge-base contents via relevance‑driven query crafting.

<p align="center">
  <img src="https://github.com/gnekt/PiratesOfTheRAG/imgs/6e4eca81.png" alt="Attack Overview" width="100%">
</p>

## 📖 Table of Contents

- [Pirates of the RAG](#pirates-of-the-rag)
  - [📖 Table of Contents](#-table-of-contents)
  - [🔑 Key Features](#-key-features)
  - [🛠️ Prerequisites](#️-prerequisites)
  - [💾 Installation](#-installation)
  - [🗄️ Preparing Knowledge Bases](#️-preparing-knowledge-bases)
  - [🚀 Running Experiments](#-running-experiments)
    - [1. Launch Target RAG Agents](#1-launch-target-rag-agents)
    - [2. Launch Attacker's Oracle LLM](#2-launch-attackers-oracle-llm)
    - [3. Run the Attack](#3-run-the-attack)
    - [4. Analyze Results](#4-analyze-results)
  - [🛡️ Testing Llama Guard Defense](#️-testing-llama-guard-defense)
  - [📂 Repository Structure](#-repository-structure)
  - [📑 Citation](#-citation)
  - [⚠️ Ethical Considerations](#️-ethical-considerations)
  - [📜 License](#-license)

---

## 🔑 Key Features

* **Fully Automated & Adaptive**: Runs without human supervision, adapting queries dynamically based on leaked data.
* **Black‑Box Approach**: Requires no internal knowledge of the target RAG’s architecture or prompts.
* **Open‑Source & Locally Runnable**: Powered by consumer‑grade open models (e.g., Llama 3.2 1B) and embedder, no proprietary APIs.
* **Relevance‑Based Anchor Mechanism**: Uses dynamically updated topic anchors to guide exploration of hidden KB chunks.
* **Proven Effectiveness**: Outperforms existing extraction methods across multiple RAG configurations.

---

## 🛠️ Prerequisites

* **Python** ≥ 3.9
* **CUDA‑enabled NVIDIA GPU** (≥ 8 GB VRAM (for the Agent), ≥ 4 GB VRAM (for the Attacker))
* **Virtual environment manager** (conda or venv)

---

## 💾 Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/your-username/pirates-of-the-rag.git
   cd pirates-of-the-rag
   ```

2. **Create & activate a virtual environment**

   ```bash
   # Conda
   conda create -n pirate_rag python=3.9  
   conda activate pirate_rag

   # Or venv
   python3 -m venv venv  
   source venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## 🗄️ Preparing Knowledge Bases

Download and preprocess the datasets (e.g., HealthcareMagic, Mini‑Wikipedia, Mini‑BioASQ) according to the provided scripts. Place raw data under `data/` and run any preprocessing utilities to populate your ChromaDB stores.

---

## 🚀 Running Experiments

### 1. Launch Target RAG Agents

Start each target agent in a separate terminal:

```bash
# Agent A: ChatDoctor
python vLLMApi.py --agent_name chatdoctor --port 8010

# Agent B: Wikipedia
python vLLMApi.py --agent_name wikipedia --port 8030

# Agent C: BioASQ
python vLLMApi.py --agent_name bioasq --port 8020
```

### 2. Launch Attacker's Oracle LLM

In a new terminal, run the local oracle model:

```bash
python vLLMApi.py --agent_name oracle --port 8000
```

### 3. Run the Attack

Execute the main script with your desired parameters:

```bash
python main.py \
  --chat_agent wikipedia \
  --nr_topics_to_use 3 \
  --anchor_beta 1 \
  --anchor_similarity_threshold 0.8 \
  --oracolo Llama-3.2-1B
```

Logs and outputs will be saved under `experiments/[experiment_id]/`.

### 4. Analyze Results

Process results and generate metrics/plots:

```bash
python stats.py --experiment_id [YOUR_EXPERIMENT_ID]
```

Plots, statistics, and leak metrics will appear in `experiments/[experiment_id]/`.

---

## 🛡️ Testing Llama Guard Defense

1. **Launch Guarded Agent**

   ```bash
   # Example for Wikipedia
   python vLLMApi_guard.py --agent_name wikipedia --port 8030
   ```

2. **Run Test Script**

   ```bash
   python test_llama_guard.py
   ```

Safety statistics will be stored in the corresponding experiment directory.

---

## 📂 Repository Structure

```
.
├── data/                    # Raw and processed datasets
├── experiments/            # Logs, stats, plots, and saved models
│   └── [experiment_id]/
├── utils/                  # Prompt templates, helper scripts, etc.
├── DGEA/                   # UnleashingWorms implementation
├── RThief/                 # RagThief implementation
├── Random/                 # PIDE, TGTB, GPTGEN baselines
├── main.py                 # Launch point for Pirate attack
├── vLLMApi.py              # Server for target agents & oracle
├── vLLMApi_guard.py        # Server with Llama Guard enabled
├── test_llama_guard.py     # Evaluate guard efficacy
├── stats.py                # Generate experiment metrics & plots
├── KnowledgeBase.py        # ChromaDB management
├── CustomEmbedder.py       # Embedding utilities
└── README.md               # This file
```

---

## 📑 Citation

If you use this work, please cite our paper:

```bibtex
@inproceedings{pirates-of-the-rag-2025,
  title={Pirates of the RAG: Adaptively Attacking LLMs to Leak Knowledge Bases},
  author={Di Maio, Christian and Cosci, Cristian and Maggini, Marco and Poggioni, Valentina and Melacci, Stefano},
  booktitle={Proceedings of the European Conference on Artificial Intelligence (ECAI)},
  year={2025},
  url={https://arxiv.org/abs/your-paper-link-here}
}
```

---

## ⚠️ Ethical Considerations

This code is released **for academic and research purposes only**. Our goal is to highlight vulnerabilities in RAG systems and foster the development of stronger privacy‑preserving defenses. **Malicious use is strongly discouraged.**

---

## 📜 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.
