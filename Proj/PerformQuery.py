import argparse
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_cohere import CohereEmbeddings

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

CHROMA_PATH = r"Proj/data"     #specify the path to your vector database 

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

def query_rag(query_text: str):
    embedding_function = CohereEmbeddings(cohere_api_key="gNNfn4USH8AqKSWg0pLzCYVr4cAqDrN7Tz8HVOW8",model="small")        #put in your cohere api key here
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    results = db.similarity_search_with_score(query_text, k=5)      #performing similarity search to extract meaningful content from the document k value 5 specifies that top 5 content will be extracted

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    # print(results)

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    return prompt

# print(query_rag("What is cybersecurity")[0])
