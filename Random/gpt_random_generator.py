from langchain_openai import ChatOpenAI
from tqdm import tqdm

prompt = "Generate 50 random questions, each of them long 15-20 words, Output only the questions without any explanation or other text. Each question on a new line"

llm = ChatOpenAI(model="gpt-4o-mini", api_key=open("apikey").read())

collection = []
for i in tqdm(range(6)):
  reply = llm.invoke(prompt)
  extract_list = reply.content.split("\n")
  for item in extract_list:
    if len(item) > 0:
      collection.append(item)

for item in collection:
  with open("questions_gpt.txt", "a") as f:
    f.write(item + "\n")