from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
import os

def createEmbeddings(query):
    embeddings = OllamaEmbeddings()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    all_data = []
    chorma_db = "phi3_chroma_db"

    files ="../Embeddings Data"

    if not os.path.exists(chorma_db):
        for file in files:
            print(file+"\n")
            loader = UnstructuredPDFLoader(file)
            docs = loader.load()
            data = text_splitter.split(docs)
            all_data.extend(data)

        db = Chroma.from_documents(all_data, embeddings, persist_directory=chorma_db)
    else:
        db = Chroma(persist_directory=chorma_db, embedding_function=embeddings)
    
    docs = db.similarity_search(query)
    print(docs)

    return docs
    
def RAGwithPhi3(query):
    llm = ChatOllama(model="phi3")
    db = createEmbeddings(query)
    retriever = db.as_retriever(search_kwargs={"k": 2})

    template = """
    You are an helpful AI assistant that can help me with my queries based on the context that has been provided.

    Context: {context}
    
    Question: {query}

    Answer:
    """

    qa_prompt = PromptTemplate(template=template, input_variables=["context","question"])
    chain_type_kwargs = { "prompt": qa_prompt, "verbose": True }

    qa = RetrievalQA.from_chain_type(
        llm= llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs=chain_type_kwargs,
        verbose = True
        )

    respose = qa.invoke(query)
    print(respose)

RAGwithPhi3("Hi")