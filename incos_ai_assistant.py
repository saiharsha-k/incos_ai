import streamlit as st
from transformers import pipeline
from pinecone import Pinecone
import numpy as np
from langchain import LangChain

st.set_page_config(
    page_title="INCOS AI Assistant",
    page_icon="🎓",
    layout="centered"
)

@st.cache_resource
def initialize_pinecone():
    api_key = "pcsk_2uxcgr_7EXRxqcQDew4CqgB2B9Q1M9EgwqpPCw4HAL7wjcLgHSN7g6ToZoAnEtBvjsHA3J"
    pc = Pinecone(api_key=api_key)
    index = pc.Index("corpus-embeddings")
    return index

@st.cache_resource
def initialize_models():
    embedding_model = pipeline("feature-extraction", model="sentence-transformers/all-MiniLM-L6-v2")
    qa_model = pipeline("question-answering", model="Sai-Harsha-k/incosai_qa_finetuned_model")
    return embedding_model, qa_model

def get_embeddings(text: str, model) -> np.ndarray:
    embeddings = model(text, padding=True, truncation=True, max_length=512)
    return np.mean(embeddings[0], axis=0)

def search_pinecone(query_embedding: np.ndarray, index, top_k: int = 3):
    return index.query(vector=query_embedding.tolist(), top_k=top_k, include_metadata=True)

def generate_answer(question: str, context, qa_pipeline):
    context_text = " ".join([doc["metadata"]["text"] for doc in context["matches"]])
    result = qa_pipeline(question=question, context=context_text)
    return result['answer']

# LangChain setup
lc = LangChain()

st.title("INCOS AI Assistant")
st.write("How can I help you with your coursework?")

user_question = st.text_input("Ask me anything related to your coursework!")

if user_question:
    try:
        pinecone_index = initialize_pinecone()
        embedding_model, qa_model = initialize_models()
        query_embedding = get_embeddings(user_question, embedding_model)
        context = search_pinecone(query_embedding, pinecone_index)
        answer = generate_answer(user_question, context, qa_model)
        st.write(answer)
    except Exception as e:
        st.error(f"An error occurred: {e}")
