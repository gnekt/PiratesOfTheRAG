import pandas as pd
import argparse  # Import argparse for command-line argument parsing
import pickle
from huggingface_hub import hf_hub_download
from CustomEmbedder import CustomEmbedder
from langchain_core.documents import Document
from typing import List
from tqdm import tqdm
from datasets import load_dataset
from langchain_chroma import Chroma


# Parse command-line arguments
parser = argparse.ArgumentParser(description='Process agent name for document embedding.')
parser.add_argument('--agent_name', type=str, help='Name of the agent (bioasq, wikipedia, chatdoctor)', default="chatdoctor")
args = parser.parse_args()


# Load kb
agent_name = args.agent_name


kb = pd.read_csv('data/{agent_name}/{agent_name}.csv'.format(agent_name=agent_name))

docs: List[Document] = []

if agent_name == "chatdoctor":
      for index, row in tqdm(kb.iterrows(), total=kb.shape[0]):
            text = row["passage"].replace("output:", " Doctor:").replace("input:", "Patient:")
            doc = Document(text, metadata={'index': index})
            docs.append(doc)

if agent_name == "bioasq":
      for index, row in tqdm(kb.iterrows(), total=kb.shape[0]):
            doc = Document(f"{row['passage']}", metadata={'index': index})
            docs.append(doc)

if agent_name == "wikipedia":
      for index, row in tqdm(kb.iterrows(), total=kb.shape[0]):
            doc = Document(f"passage: {row['passage']}", metadata={'index': index})
            docs.append(doc)

if agent_name == "bioasq":
      embedder = CustomEmbedder("Alibaba-NLP/gte-large-en-v1.5")
if agent_name == "wikipedia":
      embedder = CustomEmbedder("intfloat/e5-large-v2")     
if agent_name == "chatdoctor":
      embedder = CustomEmbedder("BAAI/bge-large-en-v1.5")

# Load retriever 
for i in range(0, len(docs), 5000):
   chromadb = Chroma.from_documents(collection_name=agent_name, 
                              documents=docs[i:i+5000], 
                              embedding=embedder.embedder, 
                              persist_directory="./chroma_db", 
                              collection_metadata={"hnsw:space": "cosine"})