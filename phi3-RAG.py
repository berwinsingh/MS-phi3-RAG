from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import warnings

warnings.filterwarnings("ignore")

def createEmbeddings():
    embeddings = OllamaEmbeddings(model="phi3")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_data = []
    index_file = "./faiss.index"

    files ="./Embeddings Data"

    if not os.path.exists(index_file):
        for file in os.listdir(files):
            file_path = os.path.join(files, file)
            if os.path.isfile(file_path):
                print(file_path+"\n")
                loader = UnstructuredPDFLoader(file_path)
                docs = loader.load()
                data = text_splitter.split_documents(docs)
                all_data.extend(data)

        db = FAISS.from_documents(all_data, embeddings)
        db.save_local(index_file)
    else:
        db = FAISS.load_local(index_file, embeddings, allow_dangerous_deserialization=True)
    
    # print(db)
    return db
    
def RAGwithPhi3(query):
    llm = ChatOllama(model="phi3")
    db = createEmbeddings()
    retriever = db.as_retriever(search_kwargs={"k": 2})

    context_documents = retriever.get_relevant_documents(query)
    context_text = ' '.join([doc.page_content for doc in context_documents])
    # print("\n")
    # print(context_text)

    template = f"""
    You are a helpful AI assistant that can help me with my queries based on the context that has been provided. If the answer isn't clear, please let me know.
    If the answer isn't relevant to the context just say "I don't know".
    
    Context: {context_text}
    
    Question: {query}

    Answer:
    """

    response = llm.invoke(template)
    print("\n")
    print(response.content.split("<|end|>")[0].strip())

while True:
    query = input("Enter your query: ")
    RAGwithPhi3(query)