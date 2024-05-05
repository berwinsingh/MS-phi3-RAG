# RAG with Microsoft Phi 3

This repository explores the performance of [Microsoft's Phi3 model](https://azure.microsoft.com/en-us/blog/introducing-phi-3-redefining-whats-possible-with-slms/) within a Retrieval-Augmented Generation (RAG) setup. This is an experimental approach to understand how well Phi3 performs compared to other models like Claude, Mistral, GPT, or Llama.

## Overview
The Phi3 model shows promise, particularly for straightforward queries, though it struggles with more complex questions. However, with further tuning, its performance could potentially improve.

## Installation Steps
Follow these steps to set up and run the application:

1. **Install Dependencies**:
   - Install all required packages from the **requirements.txt** file using the command:
     ```
     pip install -r requirements.txt
     ```

2. **Data Setup**:
   - Create a folder named **Embeddings Data** in the root directory. This folder should contain all the PDFs you intend to use for creating the VectorDB using Meta Faiss.

3. **Running the Application**:
   - Use the following command to run the application:
     ```
     streamlit run phi3-RAG.py
     ```

## First Time Setup
When you run the application for the first time:

1. **Initial Index Creation**:
   - If no **faiss.index** file exists, the application will take some time to process your initial query as it generates vector embeddings for the documents you've provided.

2. **Local Storage**:
   - Once the vector embeddings are created, they are stored locally in a file named **faiss.index**. This file facilitates faster semantic searches and provides context to the LLM for future queries.

3. **Maintenance**:
   - You do not need to regenerate the **faiss.index** unless you add new documents or delete the index file.

## Usage Notes
- This setup is not the definitive method for a RAG pipeline but serves as a functional example to demonstrate the capabilities of the Phi3 model within a RAG context.
- For optimal performance, consider adjusting the complexity of your queries and tuning the model as needed.
