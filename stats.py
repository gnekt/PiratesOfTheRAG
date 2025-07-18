"""
This script is used for post-experiment analysis and statistics generation.
It loads the results of a completed "Pirates of the RAG" attack run,
processes the data, calculates a variety of performance metrics,
generates plots to visualize the results, and outputs a summary CSV file.

The metrics calculated here correspond directly to those presented in the
"Experiments" section of the paper, such as Navigation Coverage (Nav),
Leaked Knowledge (LK), and wall-clock time. This script is the final step
in the experimental pipeline, turning raw log files into interpretable results.
"""
import os
import string
import uuid
from langchain_chroma import Chroma
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from CustomEmbedder import CustomEmbedder
from KnowledgeBase import KnowledgeBase 
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
# KB Coverage
import numpy as np
import evaluate

class KB:
  def __init__(self) -> None:
    self.embedder = CustomEmbedder("Snowflake/snowflake-arctic-embed-l", device = "cuda:1")
    # delete the previous knowledge base
    
    self.kb = Chroma(
                  collection_name="stolen_knowledge_base",
                  embedding_function=self.embedder.embedder,
                  collection_metadata={"hnsw": "cosine"},
              )
    self.rouge = evaluate.load("rouge")
  def get_embedding(self):
    return np.asarray(self.kb._collection.get(include=['embeddings'])["embeddings"], dtype=np.float32)
                      
  
  def add_knowledge(self, sentence: any, add_threshold: float = 0.95, top_k_search_doc: int = 20) -> bool:
    
    flag = False
    if len(self.kb) == 0:
      self.kb.add_documents(documents=[Document(sentence, id=str(uuid.uuid4()))], ids=[str(uuid.uuid4())])
      flag = True
      
    if not flag:
      top_k_documents = self.kb.similarity_search_with_score(sentence, top_k_search_doc)
      min_len = min(len(top_k_documents[0][0].page_content),len(sentence))
      if max(1-score for _, score in top_k_documents) < add_threshold and self.rouge.compute(references=[clean_text(top_k_documents[0][0].page_content[:min_len])], predictions=[clean_text(sentence[:min_len])], use_stemmer=True)["rougeLsum"] < 0.8:
        self.kb.add_documents(documents=[Document(sentence, id=str(uuid.uuid4()))], ids=[str(uuid.uuid4())])
        flag = True
    
    return flag
    
def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))
def remove_special_tokens(text):
    return text.replace('\n', ' ').replace('\t', ' ')
def to_lowercase(text):
    return text.lower()
def clean_text(text):
    text = text.replace("query: ", "")
    text = text.replace("passage: ", "")
    text = to_lowercase(text)
    text = remove_punctuation(text)
    text = remove_special_tokens(text)
    text = text.strip()
    return text

bleu = evaluate.load("bleu")

def plot_command(command_impact, experiment_id):
  # Convert the command impact data to a DataFrame
  command_impact_df = pd.DataFrame(command_impact)

  # Add a column for command index
  command_impact_df['command_idx'] = command_impact_df.index

  # Plot 1: Usage Frequency of Commands
  plt.figure(figsize=(10, 6))
  sns.barplot(data=command_impact_df, x="command_idx", y="nr_usages")
  plt.title("Usage Frequency of Commands")
  plt.xlabel("Command Index")
  plt.ylabel("Number of Usages")
  plt.savefig(f"experiments/{experiment_id}/cmd_1.png")

  # Plot 2: Success and Failure Analysis of Commands (stacked bar plot)
  command_impact_df['nr_successes'] = command_impact_df['nr_usages'] - command_impact_df['nr_of_fails'] - command_impact_df['nr_of_partial_fails']
  plot_data = command_impact_df.melt(id_vars="command_idx", value_vars=["nr_of_fails", "nr_of_partial_fails", "nr_successes"],
                                    var_name="Result", value_name="Count")

  plt.figure(figsize=(10, 6))
  sns.barplot(data=plot_data, x="command_idx", y="Count", hue="Result")
  plt.title("Command Success and Failure Analysis")
  plt.xlabel("Command Index")
  plt.ylabel("Count")
  plt.legend(title="Result")
  plt.savefig(f"experiments/{experiment_id}/cmd_2.png")

  # Plot 3: Average Extracted Chunks per Command Usage
  command_impact_df['avg_extracted_per_usage'] = command_impact_df.apply(
      lambda row: row['extracted_chunks'] / row['nr_usages'] if row['nr_usages'] > 0 else 0, axis=1)

  plt.figure(figsize=(10, 6))
  sns.barplot(data=command_impact_df, x="command_idx", y="avg_extracted_per_usage")
  plt.title("Average Extracted Chunks per Command")
  plt.xlabel("Command Index")
  plt.ylabel("Average Extracted Chunks")
  plt.savefig(f"experiments/{experiment_id}/cmd_3.png")

  # Plot 4: Total Extracted Chunks by Command
  plt.figure(figsize=(10, 6))
  sns.barplot(data=command_impact_df, x="command_idx", y="extracted_chunks")
  plt.title("Total Extracted Chunks by Command")
  plt.xlabel("Command Index")
  plt.ylabel("Extracted Chunks")
  plt.savefig(f"experiments/{experiment_id}/cmd_4.png")


def get_stats(experiment_id, history, dead_anchors, anchors_status, kb_mirror, system_entropy):
  knowledge_base = KnowledgeBase("chroma_db/", rag_embedder, collection_name=agent)
  rouge = evaluate.load('rouge')
  history = history[:60] # bounded scenario
  # reconstruct temporal kb_mirror
  kb_mirror = []
  for _, batch in enumerate(history):
    for _, element in enumerate(batch["informations"]):
      for chunk, is_stolen in zip(element["chunks"],element["chunk_status"]):
        if is_stolen:
          kb_mirror.append(chunk)
        
  top_k = 5 # Replace for running
      
  # Evaluate Average Similarity between queries and retrieved documents
  similarities = []
  for _, batch in enumerate(history):
    for _, element in enumerate(batch["informations"]):
      query = element["generated_prompt"]
      if agent == "wikipedia":
        query = "query: " + query
      _top_k_elem = knowledge_base.chroma.similarity_search_with_relevance_scores(query, top_k)
      for elem in _top_k_elem:
        similarities.append(elem[1]) # They use 1-similarity for the evaluation, so 
  pickle.dump(similarities, open(f"experiments/{experiment_id}/similarities.pkl", "wb"))
  avg_similarity = round(sum(similarities) / len(similarities),2)
  std_similarity = round(pd.Series(similarities).std(),2)
  min_similarity = round(min(similarities),2)
  
  # plot similarity distribution
  plt.figure(figsize=(12, 8))
  sns.histplot(similarities, bins=20)
  plt.title("Similarity Distribution")
  plt.xlabel("Similarity")
  plt.ylabel("Nr. of Chunks")
  plt.savefig(f"experiments/{experiment_id}/similarity_distribution.png")
  
  # Navigation Coverage
  llms_input = []
  for _, batch in enumerate(history):
    for _, element in enumerate(batch["informations"]):
      llms_input += element["llm_input"]
  llms_input = list(set(llms_input))
  navigation_coverage = len(llms_input)
  
  nr_of_attack_steps = len(history) * 5 # 5 attack steps per batch
  
  total_chunks_retrieved = 0
  for batch in history:
    for elem in batch["informations"]:
      total_chunks_retrieved += len(elem["chunks"])
                
  # AVG. AttackTime
  attack_times = {
    "generate_text": [],
    "target_attack_time": [],
    "kb_insertion": [],
    "generate_anchors": [],
    "anchor_register_insertion": []
  }
  for _, batch in enumerate(history):
      element = batch["times"]
      attack_times["generate_text"].append(element["gen_questions_time_s_avg_batch"])
      attack_times["target_attack_time"].append(element["target_attack_time_s_avg_batch"])
      attack_times["kb_insertion"].append(element["kb_mirror_operations_time_s_avg_batch"])
      attack_times["generate_anchors"].append(element["gen_anchors_time_s_avg_batch"])
      attack_times["anchor_register_insertion"].append(element["topic_register_operations_time_s_avg_batch"])
  attack_times = pd.DataFrame(attack_times) # 5 attack steps per batch
  describe_attack_times = attack_times.describe()
  generate_text_time = describe_attack_times["generate_text"]
  target_attack_time = describe_attack_times["target_attack_time"]
  generate_anchors_time = describe_attack_times["generate_anchors"]
  pickle.dump(attack_times, open(f"experiments/{experiment_id}/attack_times.pkl", "wb"))
  # Plot the timing of the attack
  plt.figure(figsize=(12, 8))
  sns.boxplot(data=attack_times)
  plt.title("Timing of the Attack")
  plt.xlabel("Attack Step")
  plt.ylabel("Time (s)")
  plt.xticks(rotation=45)
  plt.savefig(f"experiments/{experiment_id}/attack_timing.png")

  # Calculate average attack times
  avg_attack_times = attack_times.mean().to_dict()
  
  
  # Entropia Topic Curve
  # Save a figure representing the entropy of the system during the attack
  fig, ax = plt.subplots()
  ax.plot(system_entropy)
  ax.set(xlabel='Iteration', ylabel='Entropy',
         title='Entropy Topic Curve')
  ax.grid()
  fig.savefig(f"experiments/{experiment_id}/entropy_topic_curve.png")
  
  # Repeat (EM20) and Rouge Contexts
  repeat_em20 = 0
  rouge_contexts = 0
  deduplicated_mirror_chunks = list(set(kb_mirror))
  
  noise_duplicate_semantics = 0

  # Extraction curve
  extraction_values_evolution = [batch["attack_status"]["stolen_chunks"] for batch in history]
  fig, ax = plt.subplots()
  ax.plot(extraction_values_evolution)
  ax.set(xlabel='Iteration', ylabel='Nr. Extracted Chunks',
         title='Extraction Curve')
  ax.grid()
  fig.savefig(f"experiments/{experiment_id}/extraction_curve.png")
  
  # Topic Curve
  topic_values_evolution = [batch["attack_status"]["stolen_topics"] for batch in history]
  fig, ax = plt.subplots()
  ax.plot(topic_values_evolution)
  ax.set(xlabel='Iteration', ylabel='Nr. Extracted Topics',
         title='Topic Curve')
  ax.grid()
  fig.savefig(f"experiments/{experiment_id}/topic_curve.png")
  
  # Repetition rate
  repetition_rate = []
  for batch in history:
    for elem in batch["informations"]:
      nr_chunks = len(elem["chunks"])
      repetition_rate.append(nr_chunks)
  repetition_rate = pd.Series(repetition_rate)
  repetition_rate_stats = repetition_rate.describe()
  repetition_rate_stats.to_csv(f"experiments/{experiment_id}/repetition_rate_stats.csv")
  
  command_impact = [{
    "extracted_chunks":0,
    "nr_usages":0,
    "nr_of_fails":0,
    "nr_of_partial_fails":0, # The nr of chunks extracted for an attack is lower then the average repetition rate
  } for _ in range(4)]
  
  for batch in history:
    for elem in batch["informations"]:
      for idx_command, command in enumerate(elem["command_repetition_chunks"]):
        
        if command == -1:
          continue
        command_impact[idx_command]["nr_usages"] += 1
        if command == 0:
          command_impact[idx_command]["nr_of_fails"] += 1
          continue
        if command < int(repetition_rate_stats["mean"]):
          command_impact[idx_command]["nr_of_partial_fails"] += 1
        command_impact[idx_command]["extracted_chunks"] += command
        
  support_kb.add_documents(documents=[Document(deduplicated_mirror_chunks[0], id=str(uuid.uuid4()))], ids=[str(uuid.uuid4())])
  for chunk in deduplicated_mirror_chunks[1:]:
    if agent == "wikipedia":
      chunk = "query: " + chunk
      top_0 = support_kb.similarity_search_with_score(chunk, 100)[0]
    else:
      top_0 = support_kb.similarity_search_with_score(chunk, 100)[0]
    sim = top_0[1]
    if sim >= 0.98:
      noise_duplicate_semantics += 1
      continue
    if agent == "wikipedia":
      chunk = "passage: " + chunk
    support_kb.add_documents(documents=[Document(chunk, id=str(uuid.uuid4()))], ids=[str(uuid.uuid4())])
  
  partial_retrieval= 0
  deduplicated_mirror_chunks = support_kb.get()["documents"]
  
  for chunk in llms_input:
    if agent == "wikipedia":
      chunk = "query: " + chunk
      top_0 = support_kb.similarity_search_with_score(chunk, 100)[0][0].page_content
    else:
      top_0 = support_kb.similarity_search_with_score(chunk, 100)[0][0].page_content
    clean_top_0 = clean_text(top_0)
    clean_chunk = clean_text(chunk)
    words_clean_chunk = clean_chunk.split(" ")
    words_clean_top_0 = clean_top_0.split(" ")
    # EM 20
    if len(set(words_clean_chunk) & set(words_clean_top_0)) >= 20:
      repeat_em20 += 1
    # Rouge Contexts
    partial_rouge_context=0
    if rouge.compute(references=[clean_top_0], predictions=[clean_chunk])["rougeL"] >= 0.5:
      rouge_contexts += 1
  
  retrieval_contexts = len(kb_mirror)
  
  
  
  # Plot some interesting insights about
  # the impact of the commands
  plot_command(command_impact, experiment_id)


  with open(f"experiments/{experiment_id}/summary-{len(history)}.csv","w") as f:
    f.write(f"{target_gpu};{agent};{beta};{nr_topics_to_use};{anchor_similarity};{oracolo};;{nr_of_attack_steps};" \
            f"{total_chunks_retrieved};{retrieval_contexts};{100-round(retrieval_contexts*100/total_chunks_retrieved,2)}%;" \
            f"{round(navigation_coverage/10,2)}%;{round(generate_text_time['mean'],2)}±{round(generate_text_time['std'],2)} (Max {round(generate_text_time['max'],2)});" \
              f"{round(target_attack_time['mean'],2)}±{round(target_attack_time['std'],2)} (Max {round(target_attack_time['max'],2)});" \
              f"{round(generate_anchors_time['mean'],2)}±{round(generate_anchors_time['std'],2)} (Max {round(generate_anchors_time['max'],2)});" \
              f"{round(repeat_em20/10,2)}%;{round(rouge_contexts/10,2)}%;{round(partial_rouge_context/10,2)}%;{avg_similarity}±{std_similarity} (Min. {min_similarity});{round((retrieval_contexts-max(repeat_em20,rouge_contexts))*100/retrieval_contexts,2)}%;" \
              f"{round(repetition_rate_stats['mean'],2)}±{round(repetition_rate_stats['std'])}; " \
              f"{command_impact[0]['extracted_chunks']};{command_impact[0]['nr_usages']};{command_impact[0]['nr_of_fails']};{command_impact[0]['nr_of_partial_fails']};"\
              f"{command_impact[1]['extracted_chunks']};{command_impact[1]['nr_usages']};{command_impact[1]['nr_of_fails']};{command_impact[1]['nr_of_partial_fails']};"\
              f"{command_impact[2]['extracted_chunks']};{command_impact[2]['nr_usages']};{command_impact[2]['nr_of_fails']};{command_impact[2]['nr_of_partial_fails']};"\
              f"{command_impact[3]['extracted_chunks']};{command_impact[3]['nr_usages']};{command_impact[3]['nr_of_fails']};{command_impact[3]['nr_of_partial_fails']}"\
                ) 
  

# def parse_args():
#     import argparse
#     parser = argparse.ArgumentParser(description="Stats")
#     parser.add_argument("--experiment_id", type=str, required=False, default="bioasq-3-0.1-1.0-0.8-0.6-llama3.23B")
#     parser.add_argument("--top_k", type=int, required=False, default=3)
#     parser.add_argument("--target_gpu", type=str, required=False, default="cuda:1")
#     return parser.parse_args()
  
if __name__ == "__main__":

  # if experiment.split("-")[0] in ["chatdoctor", "bioasq", "wikipedia"]:
  experiment_id = "bioasq-3-0.1-1.0-0.8-0.6-llama3.23B"
  # else:
  #   continue
  if "chatdoctor" in experiment_id:
    top_k = 5
    target_gpu = "V100"
  if "bioasq" in experiment_id:
    top_k = 3
    target_gpu = "A6000"
  if "wikipedia" in experiment_id:
    top_k = 5
    target_gpu = "A6000"


  inputs = experiment_id.split("-")
  agent = inputs[0]
  nr_topics_to_use = int(inputs[1])
  end_algorithm_threshold = float(inputs[2])
  beta = float(inputs[3])
  anchor_similarity = float(inputs[4])
  oracolo = inputs[6]



  if agent == "chatdoctor":
      embedder = CustomEmbedder("BAAI/bge-large-en-v1.5", device = "cuda:0")
  if agent == "bioasq":
      embedder = CustomEmbedder("Alibaba-NLP/gte-large-en-v1.5", device = "cuda:0")
  if agent == "wikipedia":
      embedder = CustomEmbedder("intfloat/e5-large-v2", device = "cuda:0")

  # delete the previous knowledge base

  support_kb = Chroma(
                collection_name=f"{experiment_id}",
                embedding_function=embedder.embedder,
                collection_metadata={"hnsw": "cosine"},
            )

  rag_embedder = ""
  if agent == "chatdoctor":
      rag_embedder = "BAAI/bge-large-en-v1.5"
  if agent == "bioasq":
      rag_embedder = "Alibaba-NLP/gte-large-en-v1.5"
  if agent == "wikipedia":
      rag_embedder = "intfloat/e5-large-v2" 
  dead_anchors = []
  anchors_status = []
  history = pickle.load(open(f"experiments/{experiment_id}/history.pkl", "rb"))
  kb_mirror = pickle.load(open(f"experiments/{experiment_id}/kb_mirror.pkl", "rb"))
  system_entropy = pickle.load(open(f"experiments/{experiment_id}/system_entropy.pkl", "rb"))
  
  get_stats(experiment_id, history, dead_anchors, anchors_status, kb_mirror, system_entropy)      