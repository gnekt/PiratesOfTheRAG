import numpy as np
from tqdm import tqdm
from langchain.embeddings import HuggingFaceBgeEmbeddings, HuggingFaceEmbeddings
import os



class CustomEmbedder:
  """ Embedder class to embed the documents """
  
  def __init__(self, embedder_name: str, device: str = "cuda:0"):
    """
    Initializes the Embedder object.

    Args:
      model_name (str): The name of the HuggingFace model to use for embeddings.
    """
    model_kwargs = {'device': device}
    encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
    if embedder_name == "BAAI/bge-large-en-v1.5" or embedder_name == "BAAI/bge-small-en-v1.5":
        self.embedder = HuggingFaceBgeEmbeddings(
            model_name=embedder_name,
            encode_kwargs=encode_kwargs,
            model_kwargs=model_kwargs,
            query_instruction="Represent this sentence for searching relevant passages: "
        )
        self.embedder.query_instruction = "Represent this sentence for searching relevant passages: "
    if embedder_name == "Alibaba-NLP/gte-large-en-v1.5" or embedder_name == "Snowflake/snowflake-arctic-embed-l":
        if embedder_name == "Alibaba-NLP/gte-large-en-v1.5":
            model_kwargs = {'device': device,"trust_remote_code": True}
        self.embedder = HuggingFaceEmbeddings(
            model_name=embedder_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
    if embedder_name == "intfloat/e5-large-v2":
        self.embedder = HuggingFaceEmbeddings(
            model_name=embedder_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
    