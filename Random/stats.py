import string
import uuid
from langchain_chroma import Chroma
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from CustomEmbedder import CustomEmbedder
from KnowledgeBase import KnowledgeBase 
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
# KB Coverage
import evaluate



# rag_embedder = ""
# if agent == "chatdoctor":
#     rag_embedder = "BAAI/bge-large-en-v1.5"
# if agent == "bioasq":
#     rag_embedder = "Alibaba-NLP/gte-large-en-v1.5"
# if agent == "wikipedia":
#     rag_embedder = "intfloat/e5-large-v2" 
    
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

class KB:
  def __init__(self,name) -> None:
    self.embedder = CustomEmbedder("Snowflake/snowflake-arctic-embed-l", device = "cuda:0")
    # delete the previous knowledge base
    
    self.kb = Chroma(
                  collection_name=name,
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
  
def get_stats(experiment_id, history):
  knowledge_base = KnowledgeBase("chroma_db/", embedder.embedder.model_name, collection_name=agent)
  mirror_kb = KB(experiment_id.replace("/","_")[:-1])
  rouge = evaluate.load('rouge')
  # Navigation Coverage
  llms_input = []
  similarities = []
  for _, attack in enumerate(history):
      llms_input += [doc for doc in attack["informations"]["llm_input"]]
  llms_input = list(set(llms_input))
  navigation_coverage = len(llms_input)
  
  # Evaluate Average Similarity between queries and retrieved documents
  similarities = []
  for _, batch in enumerate(history):
      query = batch["informations"]["generated_prompt"]
      if agent == "wikipedia":
        query = "query: " + query
      _top_k_elem = knowledge_base.chroma.similarity_search_with_relevance_scores(query, 5)
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
  
  avg_similarity = round(sum(similarities)/len(similarities),2)
  std_similarity = round(np.std(similarities),2)
  min_similarity = round(min(similarities),2)
  
  nr_of_attack_steps = len(history)
  
  # Calculate attack time
  
  
  total_chunks_retrieved = 0
  kb_mirror = []
  retrieval_contexts = 0
  for attack in history:
      total_chunks_retrieved += len(attack["informations"]["chunks"])
      for chunk in attack["informations"]["chunks"]:
        if mirror_kb.add_knowledge(chunk):
          kb_mirror += [chunk]  
  
  retrieval_contexts = len(mirror_kb.kb)  
  # Calculate average attack times  
  
  # Repeat (EM20) and Rouge Contexts
  repeat_em20 = 0
  rouge_contexts = 0
  deduplicated_mirror_chunks = list(set(kb_mirror))
  
  noise_duplicate_semantics = 0

  # Extraction curve
  extraction_values_evolution = [attack["attack_status"]["stolen_chunks"] for attack in history]
  fig, ax = plt.subplots()
  ax.plot(extraction_values_evolution)
  ax.set(xlabel='Iteration', ylabel='Nr. Extracted Chunks',
         title='Extraction Curve')
  ax.grid()
  fig.savefig(f"experiments/{experiment_id}/extraction_curve.png")
  
  # Repetition rate
  repetition_rate = []
  for attack in history:
      nr_chunks = len(attack["informations"]["chunks"])
      repetition_rate.append(nr_chunks)
  repetition_rate = pd.Series(repetition_rate)
  repetition_rate_stats = repetition_rate.describe()
  repetition_rate_stats.to_csv(f"experiments/{experiment_id}/repetition_rate_stats.csv")

        
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
      repeat_em20 = 0
    # Rouge Contexts
    partial_rouge_context=0
    if rouge.compute(references=[clean_top_0], predictions=[clean_chunk])["rougeL"] >= 0.5:
        rouge_contexts += 1
  
  retrieval_contexts = len(kb_mirror)

 
  
  # Plot some interesting insights about
  # the impact of the commands


  with open(f"experiments/{experiment_id}/summary.csv","w") as f:
    f.write(f"A6000;{agent};;;;;;{nr_of_attack_steps};" \
            f"{total_chunks_retrieved};{retrieval_contexts};{100-round(retrieval_contexts*100/total_chunks_retrieved,2)}%;" \
            f"{round(navigation_coverage/10,2)}%;{round(0,2)}±{round(0,2)} (Max {round(0,2)});" \
              f"{round(0,2)}±{round(0,2)} (Max {round(0,2)});" \
              f"0±0 (Max 0);" \
              f"{round(repeat_em20/10,2)}%;{round(rouge_contexts/10,2)}%;{round(partial_rouge_context/10,2)}%;{avg_similarity}±{std_similarity} (Min. {min_similarity});{round((retrieval_contexts-max(repeat_em20,rouge_contexts))*100/retrieval_contexts,2)}%;" \
              f"{round(repetition_rate_stats['mean'],2)}±{round(repetition_rate_stats['std'])}; " \
              f"0;0;0;0;"\
              f"0;0;0;0;"\
              f"0;0;0;0;"\
              f"0;0;0;0;") 
  


if __name__ == "__main__":

  for method in ["tgtb", "pide", "gpt"]:
    for agent in ["chatdoctor", "bioasq", "wikipedia"]:
      if agent == "chatdoctor":
          embedder = CustomEmbedder("BAAI/bge-large-en-v1.5", device = "cuda:0")
      if agent == "bioasq":
          embedder = CustomEmbedder("Alibaba-NLP/gte-large-en-v1.5", device = "cuda:0")
      if agent == "wikipedia":
          embedder = CustomEmbedder("intfloat/e5-large-v2", device = "cuda:0")

      # delete the previous knowledge base

      support_kb = Chroma(
                    collection_name=f"{agent}_{method}",
                    embedding_function=embedder.embedder,
                    collection_metadata={"hnsw": "cosine"},
                )
      #dead_anchors = pickle.load(open(f"experiments/{experiment_id}/dead_anchors.pkl", "rb"))
      #anchors_status = pickle.load(open(f"experiments/{experiment_id}/anchors_status.pkl", "rb"))
      dead_anchors = []
      anchors_status = []
      history = pickle.load(open(f"experiments/RANDOM_our/{agent}-300/{method}/history.pkl", "rb"))
      # kb_mirror = pickle.load(open(f"experiments/{experiment_id}/kb_mirror.pkl", "rb"))
      # system_entropy = pickle.load(open(f"experiments/{experiment_id}/system_entropy.pkl", "rb"))
      get_stats(f"RANDOM_our/{agent}-300/{method}/", history)