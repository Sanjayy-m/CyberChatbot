import argparse
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_cohere import CohereEmbeddings
import threading
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_chroma import Chroma
import shutil

import shutil
import os

CHROMA_PATH = r"Proj/data"





PROMPT_TEMPLATE = """
You are a cybersecurity awareness bot with RAG implementation you provide information regarding how to protect oneself from cyber threats based on the retrieved context, If the context does not match the query then generate your own answer but make sure that you generate something don't give none output
Please provide the context that are irrelevant rather than not providing

so Answer the question based on the following context:

The retrieved context is
{context}

---

The question is : {question}
If the context does not match the query then generate your own answer but make sure that you generate something don't give none output
Please provide the context that are irrelevant rather than not providing
If you get any greeting questions like Hi, Hello, etc just greet them if the question is like what can you do say that I am awareness bot and tell information regarding how to protect oneself from cyber attacks if the question is bye then greet them with bye
"""

DATA_PATH = r"Proj/InfoOnCyberThreats.pdf" #specify the path to your data folder in this case it is InfoOnCyberThreats.pdf

def loadChroma():
    def load_documents():
        document_loader = PyPDFLoader(DATA_PATH)
        return document_loader.load()
    
    def split_documents(documents: list[Document]):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 800,
            chunk_overlap=80,
            length_function=len,
        )
        return text_splitter.split_documents(documents)
    
    def calculate_chunk_ids(chunks):
    
        # This will create IDs like "data/InfoOnCyberThreats.pdf:6:2"
        # Page Source : Page Number : Chunk Index
    
        last_page_id = None
        current_chunk_index = 0
    
        for chunk in chunks:
            source = chunk.metadata.get("source")
            page = chunk.metadata.get("page")
            current_page_id = f"{source}:{page}"
    
            # If the page ID is the same as the last one, increment the index.
            if current_page_id == last_page_id:
                current_chunk_index += 1
            else:
                current_chunk_index = 0
    
            # Calculate the chunk ID.
            chunk_id = f"{current_page_id}:{current_chunk_index}"
            last_page_id = current_page_id
    
            # Add it to the page meta-data.
            chunk.metadata["id"] = chunk_id
    
        return chunks
    
    
    def add_to_chroma(chunks: list[Document]):
        # Initialize the database
        db = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=CohereEmbeddings(
                cohere_api_key="gNNfn4USH8AqKSWg0pLzCYVr4cAqDrN7Tz8HVOW8",
                model="small"
            )
        )
    
        # Calculate Page IDs
        chunks_with_ids = calculate_chunk_ids(chunks)
    
        # Add or update the documents
        existing_items = db.get(include=[])  # IDs are always included by default
        existing_ids = set(existing_items["ids"])
        print(f"Number of existing documents in DB: {len(existing_ids)}")
    
        # Only add documents that don't exist in the DB
        new_chunks = []
        for chunk in chunks_with_ids:
            if chunk.metadata["id"] not in existing_ids:
                new_chunks.append(chunk)
    
        if len(new_chunks):
            print(f"👉 Adding new documents: {len(new_chunks)}")
            new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
            db.add_documents(new_chunks, ids=new_chunk_ids)
            # Persistence is now automatic, so db.persist() is not needed
        else:
            print("✅ No new documents to add")
    
    
    def clear_database():
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)
            print("✅ Database cleared successfully.")
        else:
            print("⚠️ No database found at the specified path.")
    
    # Call the function to clear the database
    # clear_database()
    
    
    document = load_documents()
    chunks = split_documents(document)
    add_to_chroma(chunks)

db_lock = threading.Lock()
def query_rag(query_text: str):
    with db_lock:
        embedding_function = CohereEmbeddings(cohere_api_key="gNNfn4USH8AqKSWg0pLzCYVr4cAqDrN7Tz8HVOW8",model="small")
        loadChroma()
        
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        
        #if not db.get_document_count():
       #     st.error("Database is empty. Please populate the Chroma database.")
        #    return "The database is empty. No context found."
        #else:
        #    st.success("Database is not empty. Please populate the Chroma database.")
    
        results = db.similarity_search_with_score(query_text, k=5)      #performing similarity search to extract meaningful content from the document k value 5 specifies that top 5 content will be extracted
    
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    
        # print(results)
    
        prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        prompt = prompt_template.format(context=context_text, question=query_text)
        return prompt

# print(query_rag("What is cybersecurity")[0])
