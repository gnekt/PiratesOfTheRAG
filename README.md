# Pirates of the RAG: Adaptively Attacking LLMs to Leak Knowledge Bases

**Accepted at European Conference on Artificial Intelligence (ECAI) 2025**

**Authors**  
<p align='center' style="text-align:center;font-size:1em;">
    <a href="https://collectionless.ai/post/christian_dimaio/">Christian Di Maio</a>&nbsp;,&nbsp;
    <a>Cristian Cosci</a>&nbsp;,&nbsp;
    <a href="https://scholar.google.com/citations?user=kZFskCoAAAAJ&hl=en">Marco Maggini</a>&nbsp;&nbsp;
    <a href="https://scholar.google.com/citations?user=9a1nVKwAAAAJ&hl=en">Valentina Poggioni</a>&nbsp;&nbsp;
    <a href="=https://collectionless.ai/post/a-stefano_melacci/">Stefano Melacci</a>&nbsp;&nbsp;
    <br/> 
  University of Pisa (Italy), University of Siena (Italy), University of Perugia (Italy)
  <br/> 
</p>

This repository contains the official implementation for the paper: "Pirates of the RAG: Adaptively Attacking LLMs to Leak Knowledge Bases". We introduce a novel black-box, adaptive, and automated attack designed to extract the contents of a private knowledge base from a Retrieval-Augmented Generation (RAG) system.

The attack leverages an open-source LLM and a relevance-based mechanism, guided by "anchors," to systematically craft queries that compel the target RAG system to leak its private data. Our findings highlight critical vulnerabilities in current RAG designs and underscore the urgent need for more robust privacy safeguards.

Visual Overview of the Attack
The "Pirate" attack system uses its own local LLM and embedder to generate adaptive queries. These queries are injected with commands to "convince" the target RAG system's LLM (the "Parrot") to leak chunks of its private knowledge base (the "Treasure Chest"). The relevance of topics ("Anchors") is dynamically updated to guide the attack towards unexplored parts of the knowledge base.

<iframe
  src="https://mozilla.github.io/pdf.js/web/viewer.html?file=https://github.com/gnekt/PiratesOfTheRAG/imgs/Pirate.pdf"
  width="100%"
  height="600px"
  style="border: none;">
</iframe>

Key Features
Fully Automated & Adaptive: The attack runs without human intervention, dynamically adapting its strategy based on the information it has already leaked.

Black-Box Approach: The attack requires no knowledge of the target RAG system's internal architecture, models, or prompts. It only interacts with the public-facing API.

Locally Runnable & Open-Source: The entire attack mechanism is powered by open-source models (e.g., Llama 3.2 1B) that can be run on consumer-grade hardware, eliminating reliance on proprietary, pay-per-use APIs.

Relevance-Based Anchor Mechanism: A novel strategy that uses topic anchors and dynamically updated relevance scores to efficiently explore the hidden knowledge base and maximize data extraction.

Proven Effectiveness: Extensively tested against multiple RAG configurations, demonstrating superior performance in knowledge extraction compared to existing methods.

Repository Structure
.
├── data/              # Data
├── experiments/            # Output directory for logs, stats, plots, and saved models
│   └── [experiment_id]/
├── utils/
│   ├── prompt_templates/   # Prompt templates for the attacker and agent LLMs
│   └── ...
├── DGEA/                   # Mostly taken from this https://github.com/StavC/UnleashingWorms-ExtractingData
├── RThief/                 # RagThief paper implementation
├── Random/                 # PIDE, TGTB and GPTGEN implementation
├── utils/ 
├── main.py                 # Main script to launch the attack
├── vLLMApi.py          # FastAPI server to run target RAG agents and the attacker's oracle LLM
├── vLLMApi_guard.py   # Version of the server with Llama Guard defense enabled
├── test_llama_guard.py     # Script to test the effectiveness of the Llama Guard defense
├── stats.py                # Script to generate statistics and plots from experiment results
├── KnowledgeBase.py        # Helper class for managing ChromaDB knowledge bases
├── CustomEmbedder.py       # Helper class for sentence-transformer embedding models
└── README.md               # This file

Setup and Installation
1. Prerequisites
Python 3.9+

NVIDIA GPU with CUDA support (at least 16GB VRAM recommended for running agents and the oracle).

An environment manager like Conda or venv.

2. Installation
Clone the repository:

git clone https://github.com/your-username/pirates-of-the-rag.git
cd pirates-of-the-rag

Create and activate a virtual environment:

# Using conda
conda create -n pirate_rag python=3.9
conda activate pirate_rag

# Or using venv
python3 -m venv venv
source venv/bin/activate

Install the required packages:
(A requirements.txt file should be created containing all necessary packages like vllm, fastapi, uvicorn, transformers, torch, chromadb, pandas, seaborn, matplotlib, evaluate, etc.)

pip install -r requirements.txt

Prepare the Knowledge Bases:
You will need to download the datasets used for the agents' knowledge bases (HealthcareMagic, Mini-Wikipedia, Mini-BioASQ) and place them in an appropriate directory. Then, run a preprocessing script (if provided) to populate the chroma_db/ vector stores.

How to Run the Experiments
Reproducing the experiments involves running multiple services in parallel (preferably in separate terminal sessions).

Step 1: Launch the Target RAG Agents
You need to start the servers for the RAG agents you wish to attack. These will listen on different ports.

Launch Agent A (ChatDoctor):

python vllm_server.py --agent_name chatdoctor --port 8010

Launch Agent B (Wikipedia):

python vllm_server.py --agent_name wikipedia --port 8030

Launch Agent C (BioASQ):

python vllm_server.py --agent_name bioasq --port 8020

Step 2: Launch the Attacker's Oracle LLM
In a new terminal, start the server for the attacker's smaller, local LLM (f*). This is the "oracle" that generates questions and anchors. It runs on a different port and ideally a different GPU to avoid resource conflicts.

python vllm_server.py --agent_name oracle --port 8000

Step 3: Run the Attack
Once the target agent and the oracle are running, you can launch the main attack script.

python main.py --chat_agent [AGENT_NAME] --nr_topics_to_use 3 --anchor_beta 1 --anchor_similarity_threshold 0.8 --oracolo [ORACLE_MODEL_NAME]

--chat_agent: The agent to attack (chatdoctor, wikipedia, or bioasq).

--oracolo: A name for the oracle model being used (e.g., Llama-3.2-1B).

The script will run the attack and save all logs and results into the experiments/[experiment_id]/ directory.

Step 4: Analyze the Results
After the attack is complete, run the stats.py script to process the logs and generate the final metrics and plots.

python stats.py --experiment_id [EXPERIMENT_ID_FROM_PREVIOUS_STEP]

Testing the Llama Guard Defense
To evaluate the effectiveness of a guardrail defense, follow these steps:

Launch a Guarded Agent: Stop the regular agent server and launch the version with Llama Guard enabled using vllm_with_guardian.py.

# Example for Agent B (Wikipedia)
python vllm_with_guardian.py --agent_name wikipedia --port 8030

Run the Test Script: Execute the test_llama_guard.py script. It will load the attack prompts from a completed experiment and send them to the guarded agent to collect safety statistics.

# Ensure the 'agent' variable inside the script is set to your desired experiment_id
python test_llama_guard.py

The results will be saved to a pickle file inside the corresponding experiment directory.

Citation
If you use this work, please cite our paper:

@inproceedings{pirates-of-the-rag-2025,
  title={Pirates of the RAG: Adaptively Attacking LLMs to Leak Knowledge Bases},
  author={Your, Author Names Here},
  booktitle={Proceedings of the European Conference on Artificial Intelligence (ECAI)},
  year={2025},
  url={https://arxiv.org/abs/your-paper-link-here}
}

Ethical Considerations
This research and the accompanying code are released for academic and research purposes only. The goal is to expose potential vulnerabilities in RAG systems to help the community develop more secure and robust privacy-preserving technologies. We strongly condemn any malicious use of this code. We advocate for responsible disclosure and believe that understanding attack vectors is a crucial first step toward building effective defenses.

License
This project is licensed under the MIT License. See the LICENSE file for details.