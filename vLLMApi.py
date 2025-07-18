"""
This script sets up a FastAPI web server to serve a large language model using the vLLM library.
It can be configured to act as either a complete Retrieval-Augmented Generation (RAG)
system or as a standard "oracle" LLM.

This dual-purpose capability is used to simulate the experimental setup described in the
"Pirates of the RAG" paper:
1.  **RAG System (Target Agent)**: When provided with an embedder and a knowledge base,
    it functions as one of the target agents (e.g., Agent A, B, or C). It receives a
    query, retrieves relevant documents from its private knowledge base (K),
    injects them into a prompt, and generates a response with the LLM (f).
2.  **Oracle LLM (Attacker's LLM)**: When run without a knowledge base, it acts as the
    attacker's own open-source LLM (f*). This is used for the 'generate_base_query'
    and 'extract_anchors' steps in the attack algorithm.

The script exposes a '/generate' endpoint that receives prompts and returns the
LLM's generated text.
"""

import argparse
from typing import AsyncGenerator, List
import os
import json

from fastapi import FastAPI, Request
from fastapi.responses import Response, StreamingResponse
import uvicorn

from fastapi import HTTPException
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
from pydantic import BaseModel
from vllm import LLM

from KnowledgeBase import KnowledgeBase
import os
import sys

TIMEOUT_KEEP_ALIVE = 5  # seconds.
app = FastAPI()

class vLLMGenerator:
  """
  A wrapper class for the vLLM engine that handles text generation,
  including the RAG pipeline if configured to do so.
  """
  def __init__(self, agent_name: str, model_name: str, embedder: str, knowledge_base: str, quantization: str = "fp16", device_id: int = 0):
    """
    Initializes the vLLM generator.

    Args:
        agent_name (str): The name of the agent (e.g., 'chatdoctor', 'oracle').
                          This determines the prompt template and behavior.
        model_name (str): The name or path of the Hugging Face model to load.
        embedder (str): The name or path of the sentence-transformer model to use for retrieval.
                        If None, the RAG functionality is disabled (for the oracle).
        knowledge_base (str): The name of the collection in ChromaDB to use as the knowledge base.
        quantization (str, optional): The quantization method to use. Defaults to "fp16".
        device_id (int, optional): The GPU device ID to use. Defaults to 0.
    """
    # Load the LLM using vLLM, which provides high-throughput serving.
    # The GPU memory utilization is adjusted based on whether it's the oracle or a larger RAG agent.
    if quantization not in ["fp16", "int8", "bfp16"]:
      self.llm = LLM(model=model_name, quantization=quantization, max_model_len=8192, gpu_memory_utilization=0.5 if agent_name == "oracle" else 0.7)
    else:
      self.llm = LLM(model=model_name, dtype="half",  max_model_len=8192, gpu_memory_utilization=0.5 if agent_name == "oracle" else 0.7)
    
    self.agent_name = agent_name
    self.embedder_name = None

    # If an embedder is provided, it means this instance should act as a RAG system.
    # It initializes the KnowledgeBase to perform document retrieval.
    if embedder is not None:
      self.embedder_name = embedder
      self.knowledge_base = KnowledgeBase("chroma_db/", embedder, collection_name=knowledge_base)
   
    
  def generate_texts(self, prompts: list, top_k_docs: int = 5, temperature: float = 0.0, top_p: float = 0.75, top_k: int = 40, max_length: int = 4096):
    """
    Generates text for a list of prompts. If configured as a RAG system,
    it first retrieves relevant context and injects it into the prompt.

    Args:
        prompts (list): A list of input queries.
        top_k_docs (int, optional): The number of documents to retrieve for context. Defaults to 5.
        temperature (float, optional): The sampling temperature. Defaults to 0.0.
        top_p (float, optional): The nucleus sampling probability. Defaults to 0.75.
        top_k (int, optional): The top-k sampling value. Defaults to 40.
        max_length (int, optional): The maximum number of tokens to generate. Defaults to 4096.

    Returns:
        A tuple containing the list of retrieved documents and the list of generated texts.
    """
   
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, top_k=top_k, max_tokens=max_length)
    queries = prompts
    relevant_documents = [] # To store retrieved docs for the batch.
    
    # --- RAG Pipeline ---
    # This block is executed only if the generator is configured as a RAG agent (not the oracle).
    if self.embedder_name is not None:
      for index, prompt in enumerate(prompts):
        query = prompts[index]
        
        # Some embedding models require a specific prefix for queries.
        if self.embedder_name == "intfloat/e5-large-v2":
          query = "query: " + query

        # 1. Retrieve: Get the top-k most relevant documents from the knowledge base.
        retrieved_docs = self.knowledge_base.get_relevant_documents(query, k=top_k_docs)
        relevant_documents.append([document[0].page_content.replace('passage: ','') for document in retrieved_docs])

        # 2. Augment: Load the agent-specific prompt template.
        prompt_template = open(f"utils/prompt_templates/agents/{self.agent_name}.txt", "r").read()
        prompt = prompt_template.replace("{placeholder_question}", query)
        
        # Verbalize the retrieved documents and inject them into the prompt.
        verbalized_context = ""
        for i, document in enumerate(retrieved_docs):
            if self.agent_name == "chatdoctor":
              patient_text, doctor_text = document[0].page_content.split("Doctor:")
              doctor_text = "Doctor:" + doctor_text
              verbalized_context += f"Case {i}:\n{patient_text}{doctor_text}\n\n"   
            if self.agent_name in ["bioasq", "wikipedia"]:
              verbalized_context += f"Chunk {i}:\n{document[0].page_content.replace('passage: ','')}\n\n"
              
        prompt = prompt.replace("{placeholder_context}", verbalized_context)
        queries[index] = prompt # The final prompt now includes the retrieved context.

    # 3. Generate: Call the vLLM engine to generate text from the (potentially augmented) prompts.
    outputs = self.llm.generate(queries, sampling_params)
    
    # Format the results.
    results = []
    for output in outputs:
      prompt = output.prompt
      generated_text = output.outputs[0].text
      results.append((prompt, generated_text))
      
    # Return both the retrieved documents (if any) and the final generated texts.
    return relevant_documents, [result[1] for result in results]

class GenerationItem(BaseModel):
    """
    A Pydantic model to define the structure and data types of the
    JSON payload for the /generate endpoint. This ensures that incoming
    requests are well-formed.
    """
    texts: List[str] = []
    top_k_docs: int = 5
    temperature: float = 0.0
    top_p: float = 0.75
    top_k: int = 40
    max_length: int = 4096

@app.post("/generate")
async def generate(request: Request, item: GenerationItem) -> Response:
    """
    The main FastAPI endpoint for text generation.

    It receives a list of texts (prompts) and generation parameters,
    calls the vLLMGenerator to perform the generation, and returns the results.
    """
    if not item.texts:
        raise HTTPException(
            status_code=500,
            detail=f"You need to provide a text field."
        )

    # Call the core generation logic.
    llm_inputs, llm_outputs = llm.generate_texts(
        item.texts, 
        top_k_docs=item.top_k_docs, 
        temperature=item.temperature, 
        top_p=item.top_p, 
        top_k=item.top_k, 
        max_length=item.max_length
    )

    # The response is a list of tuples, where each tuple contains the final
    # input prompt (after context injection) and the generated output. This is
    # useful for debugging and understanding what the model actually "saw".
    return [(llm_inputs[i] if len(llm_inputs)>0 else [], llm_outputs[i]) for i in range(len(llm_outputs))]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--agent_name", type=str, default="bioasq")
    args = parser.parse_args()
    
    # --- Configuration Loading ---
    # The script loads a different model and knowledge base configuration based on the
    # '--agent_name' argument. This allows the same script to launch any of the
    # target RAG agents or the attacker's oracle LLM. This directly maps to the
    # configurations listed in Table 1 of the paper.
    
    if args.agent_name == "chatdoctor":
        # Corresponds to Agent A in the paper.
        llm = vLLMGenerator(args.agent_name, "meta-llama/Meta-Llama-3.1-8B-Instruct", "BAAI/bge-large-en-v1.5", "chatdoctor", "fp16", device_id=0)
    
    if args.agent_name == "bioasq":
        # Corresponds to Agent C in the paper.
        llm = vLLMGenerator(args.agent_name, "meta-llama/Llama-3.2-3B-Instruct", "Alibaba-NLP/gte-large-en-v1.5", "bioasq", "bf16", device_id=0)
    
    if args.agent_name == "wikipedia":
        # Corresponds to Agent B in the paper.
        llm = vLLMGenerator(args.agent_name, "microsoft/Phi-3.5-mini-instruct", "intfloat/e5-large-v2", "wikipedia", "bf16", device_id=0)
    
    if args.agent_name == "oracle":
        # This is the attacker's LLM (f*). It has no embedder or knowledge base.
        # It uses a smaller, efficient model (Llama-3.2-1B) that can run on a separate device.
        llm = vLLMGenerator(args.agent_name, "meta-llama/Llama-3.2-1B-Instruct", None, None, "bf16", device_id=1)

    # Launch the FastAPI application using uvicorn, a lightning-fast ASGI server.
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="debug",
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE
    )