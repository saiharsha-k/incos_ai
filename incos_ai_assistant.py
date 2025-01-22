import streamlit as st
from transformers import pipeline
from pinecone import Pinecone
import numpy as np
from langchain.llms import HuggingFaceHub
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Pinecone as LangchainPinecone

@st.cache_resource
def initialize_models():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    qa_model = HuggingFaceHub(repo_id="Sai-Harsha-k/incosai_qa_finetuned_model", model_kwargs={"temperature": 0.5, "max_length": 512})
    return embedding_model, qa_model
    
@st.cache_resource
def initialize_pinecone():
    api_key = "pcsk_2uxcgr_7EXRxqcQDew4CqgB2B9Q1M9EgwqpPCw4HAL7wjcLgHSN7g6ToZoAnEtBvjsHA3J"
    pc = Pinecone(api_key=api_key)
    index = pc.Index("corpus-embeddings")
    return LangchainPinecone(index, embedding_model, "text")

def generate_answer(question: str, vectorstore, qa_model):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    qa_chain = RetrievalQA.from_chain_type(llm=qa_model, chain_type="stuff", retriever=retriever)
    return qa_chain.run(question)

if user_question:
    try:
        embedding_model, qa_model = initialize_models()
        vectorstore = initialize_pinecone()
        answer = generate_answer(user_question, vectorstore, qa_model)
        st.write(answer)
    except Exception as e:
        st.error(f"An error occurred: {e}")
