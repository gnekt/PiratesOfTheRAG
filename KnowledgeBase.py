import pandas as pd
import pickle
from CustomEmbedder import CustomEmbedder
from langchain_core.documents import Document
from typing import List
from tqdm import tqdm
from langchain_chroma import Chroma
from langchain_community.vectorstores.faiss import DistanceStrategy
import numpy as np
from langchain.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceEmbeddings


class KnowledgeBase:
    def __init__(self, chroma_path: str, embedder_name: str, collection_name: str = "knowledge_base", device: str = "cuda:0"):
        self.embedder = CustomEmbedder(embedder_name, device=device)
        self.chroma = Chroma(persist_directory=chroma_path, embedding_function=self.embedder.embedder, collection_name=collection_name,collection_metadata={"hnsw:space": "cosine" })
        
        
    def get_relevant_documents(self, query: str, k: int = 10):
        return self.chroma.similarity_search_with_score(query, k=k)

    def get_embedding(self):
        return np.asarray(self.kb._collection.get(include=['embeddings'])["embeddings"], dtype=np.float32)
