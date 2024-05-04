from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

llm = ChatOllama(model="phi3")

response = llm.invoke("Who created you?")

print(response.content.split("<|end|>")[0].strip())