"""
This script sets up a FastAPI web server to serve a large language model using vLLM,
now with an integrated safety guard model (Llama Guard 3) to act as a moderation layer.

This script extends the previous version by adding a defense mechanism, as discussed
in the "Pirates of the RAG" paper. It can still function as either a target RAG
system or the attacker's oracle, but when acting as a RAG system, it now performs
an additional safety check on both the user's input (the attacker's query) and its
own generated output.

This implementation allows for evaluating the effectiveness of a state-of-the-art
guard model in preventing the privacy leaks caused by the attack.

The script contains:
- vLLMGenerator: A class that now wraps both the primary generation LLM and the
  Llama Guard model.
- A 'moderate' method that performs safety checks on conversation turns.
- An updated '/generate' endpoint that returns the generation result along with
  its safety assessment.
"""

import argparse
from typing import AsyncGenerator, List
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
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
  A wrapper class for the vLLM engine that now includes a Llama Guard model
  for content moderation and safety assessment.
  """
  def __init__(self, agent_name: str, model_name: str, embedder: str, knowledge_base: str, quantization: str = "fp16", device_id: int = 0):
    """
    Initializes the vLLM generator and the Llama Guard model.

    Args:
        agent_name (str): The name of the agent (e.g., 'chatdoctor', 'oracle').
        model_name (str): The Hugging Face model for the primary generator.
        embedder (str): The sentence-transformer model for retrieval. If None, RAG is disabled.
        knowledge_base (str): The name of the ChromaDB collection.
        quantization (str, optional): The quantization method. Defaults to "fp16".
        device_id (int, optional): The GPU device ID. Defaults to 0.
    """
    # Initialize the primary LLM for generation using vLLM.
    if quantization not in ["fp16", "int8", "bfp16"]:
      self.llm = LLM(model=model_name, quantization=quantization, max_model_len=8192, gpu_memory_utilization=0.5 if agent_name == "oracle" else 0.7)
    else:
      # Adjusted memory for the guard model to fit on the same GPU.
      self.llm = LLM(model=model_name, dtype="half",  max_model_len=8192, gpu_memory_utilization=0.5 if agent_name == "oracle" else 0.5)
    
    # --- Initialize the Llama Guard Model ---
    # This model will be used to check if prompts and responses are 'safe' or 'unsafe'.
    # This is a direct implementation of a 'guard system' defense.
    self.tokenizer = AutoTokenizer.from_pretrained(
                "meta-llama/Llama-Guard-3-8B",
            )
            
    self.guard = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-Guard-3-8B",
        torch_dtype=torch.bfloat16,
        device_map="auto", # Automatically place the model on available devices.
        low_cpu_mem_usage=True
    )
    self.agent_name = agent_name
    self.embedder_name = None
    if embedder is not None:
      self.embedder_name = embedder
      self.knowledge_base = KnowledgeBase("chroma_db/", embedder, collection_name=knowledge_base)
   
  # This utility function for parsing is defined but not directly called by other methods.
  # The logic is implemented inside the `moderate` method instead.
  def parse_llama_guard_output(result):
        safety_assessment = result.split("<END CONVERSATION>")[-1].strip()
        lines = [line.strip().lower() for line in safety_assessment.split('\n') if line.strip()]
        
        if not lines:
            return "Error", "No valid output", safety_assessment

        safety_status = next((line for line in lines if line in ['safe', 'unsafe']), None)
        
        if safety_status == 'safe':
            return "Safe", "None", safety_assessment
        elif safety_status == 'unsafe':
            violated_categories = next(
                (lines[i+1] for i, line in enumerate(lines) if line == 'unsafe' and i+1 < len(lines)), 
                "Unspecified"
            )
            return "Unsafe", violated_categories, safety_assessment
        else:
            return "Error", f"Invalid output: {safety_status}", safety_assessment

  def moderate(self, user_input, assistant_response):
      """
      Performs safety moderation on a conversation turn using Llama Guard.
      It checks both the user's input and the assistant's response.

      Args:
          user_input (str): The prompt sent by the user (or attacker).
          assistant_response (str): The response generated by the primary LLM.

      Returns:
          dict: A dictionary containing the safety assessment for both input and output.
      """
      stats={
        "input": None,
        "output": None,
      }
      # 1. Moderate the user's input prompt.
      chat = [
          {"role": "user", "content": user_input}
          ]
      
      input_ids = self.tokenizer.apply_chat_template(chat, return_tensors="pt").to(self.guard.device)
      
      with torch.no_grad():
          output = self.guard.generate(input_ids=input_ids, max_new_tokens=200, pad_token_id=self.tokenizer.eos_token_id, do_sample=False)
      
      # Parse the Llama Guard output to get a 'safe' or 'unsafe' label.
      result = self.tokenizer.decode(output[0], skip_special_tokens=True)
      safety_assessment = result.split("<END CONVERSATION>")[-1].strip()
      lines = [line.strip().lower() for line in safety_assessment.split('\n') if line.strip()]
      
      safety_status = next((line for line in lines if line in ['safe', 'unsafe']), None)
      
      if safety_status == 'safe':
          stats['input']= ("Safe", "None", safety_assessment)
      elif safety_status == 'unsafe':
          violated_categories = next((lines[i+1] for i, line in enumerate(lines) if line == 'unsafe' and i+1 < len(lines)), "Unspecified")
          stats['input']= ("Unsafe", violated_categories, safety_assessment)
      else:
          stats['input']= ("Error", f"Invalid output: {safety_status}", safety_assessment)
      
      # 2. Moderate the assistant's response.
      # The response is checked in the context of the user's prompt.
      chat.append({"role": "assistant", "content": assistant_response}) 
      
      input_ids = self.tokenizer.apply_chat_template(chat, return_tensors="pt").to(self.guard.device)
      
      with torch.no_grad():
          output = self.guard.generate(input_ids=input_ids, max_new_tokens=200, pad_token_id=self.tokenizer.eos_token_id, do_sample=False)
      
      result = self.tokenizer.decode(output[0], skip_special_tokens=True)
      safety_assessment = result.split("<END CONVERSATION>")[-1].strip()
      lines = [line.strip().lower() for line in safety_assessment.split('\n') if line.strip()]
      
      safety_status = next((line for line in lines if line in ['safe', 'unsafe']), None)
      
      if safety_status == 'safe':
          stats['output']= ("Safe", "None", safety_assessment)
      elif safety_status == 'unsafe':
          violated_categories = next((lines[i+1] for i, line in enumerate(lines) if line == 'unsafe' and i+1 < len(lines)), "Unspecified")
          stats['output']= ("Unsafe", violated_categories, safety_assessment)
      else:
          stats['output']= ("Error", f"Invalid output: {safety_status}", safety_assessment)
      
      return stats
      
  def generate_texts(self, prompts: list, top_k_docs: int = 5, temperature: float = 0.0, top_p: float = 0.75, top_k: int = 40, max_length: int = 4096):
    """
    Generates text and then moderates the interaction.

    This method first runs the RAG pipeline (if applicable) and generates a response.
    Then, it calls the 'moderate' method to check the safety of both the final
    prompt and the generated response.

    Returns:
        A tuple containing the list of retrieved documents, the list of generated texts,
        and the list of moderation results.
    """
   
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, top_k=top_k, max_tokens=max_length)
    queries = list(prompts) # Create a mutable copy.
    relevants_documents = []

    # RAG pipeline (same as before).
    if self.embedder_name is not None:
      for index, prompt in enumerate(prompts):
        query = prompts[index]
        if self.embedder_name == "intfloat/e5-large-v2":
          query = "query: " + query
        relevant_documents = self.knowledge_base.get_relevant_documents(query, k=top_k_docs)
        relevants_documents.append([document[0].page_content.replace('passage: ','') for document in relevant_documents])
        prompt_template = open(f"utils/prompt_templates/agents/{self.agent_name}.txt", "r").read()
        prompt = prompt_template.replace("{placeholder_question}", query)
        verbalized_context = ""
        for i, document in enumerate(relevant_documents):
            if self.agent_name == "chatdoctor":
              patient_text, doctor_text = document[0].page_content.split("Doctor:")
              doctor_text = "Doctor:" + doctor_text
              verbalized_context += f"Case {i}:\n{patient_text}{doctor_text}\n\n"   
            if self.agent_name in ["bioasq", "wikipedia"]:
              verbalized_context += f"Chunk {i}:\n{document[0].page_content.replace('passage: ','')}\n\n"
              
        prompt = prompt.replace("{placeholder_context}", verbalized_context)
        queries[index] = prompt

    # Generate text with the primary LLM.
    outputs = self.llm.generate(queries, sampling_params)
    
    # --- Moderation Step ---
    # After generating responses, check each one for safety.
    moderations = []
    for query, output in zip(queries, outputs):
      try:
        _mode = self.moderate(query, output.outputs[0].text)
        moderations.append(_mode)
      except Exception as e:
        # Handle cases where moderation might fail.
        moderations.append({"input": ('Error', f'Moderation failed: {e}', ''), "output": ('Error', f'Moderation failed: {e}', '')})
        
    results = []
    for output in outputs:
      prompt = output.prompt
      generated_text = output.outputs[0].text
      results.append((prompt, generated_text))

    # Return the generation results along with the safety assessments.
    return relevants_documents, [result[1] for result in results], moderations

class GenerationItem(BaseModel):
    """Pydantic model for the request body of the /generate endpoint."""
    texts: List[str] = []
    top_k_docs: int = 5
    temperature: float = 0.0
    top_p: float = 0.75
    top_k: int = 40
    max_length: int = 4096

@app.post("/generate")
async def generate(request: Request, item: GenerationItem) -> Response:
    """
    The main FastAPI endpoint, now returning moderation results.
    """
    if not item.texts:
        raise HTTPException(status_code=500, detail="You need to provide a text field.")

    # Call the generation and moderation pipeline.
    llm_inputs, llm_outputs, moderations = llm.generate_texts(
        item.texts, 
        top_k_docs=item.top_k_docs, 
        temperature=item.temperature, 
        top_p=item.top_p, 
        top_k=item.top_k, 
        max_length=item.max_length
    )

    # The response now includes a third element in the tuple: the moderation dictionary.
    return [(llm_inputs[i] if len(llm_inputs)>0 else [], llm_outputs[i], moderations[i]) for i in range(len(llm_outputs))]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--agent_name", type=str, default="wikipedia")
    args = parser.parse_args()

    # The configuration loading remains the same, but any agent launched from this
    # script will now have the Llama Guard defense mechanism enabled by default.
    if args.agent_name == "chatdoctor":
        llm = vLLMGenerator(args.agent_name, "meta-llama/Meta-Llama-3.1-8B-Instruct", "BAAI/bge-large-en-v1.5", "chatdoctor", "fp16", device_id=0)
    if args.agent_name == "bioasq":
        llm = vLLMGenerator(args.agent_name, "meta-llama/Llama-3.2-3B-Instruct", "Alibaba-NLP/gte-large-en-v1.5", "bioasq", "bf16", device_id=0)
    if args.agent_name == "wikipedia":
        llm = vLLMGenerator(args.agent_name, "microsoft/Phi-3.5-mini-instruct", "intfloat/e5-large-v2", "wikipedia", "bf16", device_id=0)
    if args.agent_name == "oracle":
        # The oracle (attacker's LLM) does not have the guard model.
        llm = vLLMGenerator(args.agent_name, "meta-llama/Llama-3.2-1B-Instruct", None, None, "bf16", device_id=1)

    uvicorn.run(app, host=args.host, port=args.port, log_level="debug", timeout_keep_alive=TIMEOUT_KEEP_ALIVE)