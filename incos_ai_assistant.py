import streamlit as st
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import torch
import numpy as np

# Page configuration
st.set_page_config(
    page_title="INCOS AI Assistant",
    page_icon="🎓",
    layout="centered"
)

# Initialize Pinecone and models
@st.cache_resource
def initialize_pinecone():
    api_key = "pcsk_2uxcgr_7EXRxqcQDew4CqgB2B9Q1M9EgwqpPCw4HAL7wjcLgHSN7g6ToZoAnEtBvjsHA3J"
    pc = Pinecone(api_key=api_key)
    index = pc.Index("corpus-embeddings")  
    return index

@st.cache_resource
def initialize_models():
    # QA model
    qa_model_name = "Sai-Harsha-k/incosai_qa_finetuned_model"
    qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
    qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)
    
    # Embedding model
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    return qa_tokenizer, qa_model, embedding_model

def get_embeddings(text: str, model: SentenceTransformer) -> np.ndarray:
    """Generate embeddings for input text using sentence-transformers."""
    return model.encode([text])[0]

def search_pinecone(query_embedding: np.ndarray, index, top_k: int = 3):
    return index.query(
        vector=query_embedding.tolist(),
        top_k=top_k,
        include_metadata=True
    )

def generate_answer(question: str, context, tokenizer, model):
    context_texts = " ".join([doc["metadata"]["text"] for doc in context["matches"]])
    inputs = tokenizer(
        question,
        context_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits
        
        # Get the most likely beginning and end of answer
        answer_start = torch.argmax(start_scores)
        answer_end = torch.argmax(end_scores)
        
        # Convert tokens to string
        answer = tokenizer.decode(inputs["input_ids"][0][answer_start:answer_end+1])
    
    return answer

# Streamlit App Interface
st.title("INCOS AI Assistant")
st.write("How can I help you with your coursework?")

# User Input
user_question = st.text_input("Ask me anything related to your coursework!")

if user_question:
    try:
        # Initialize resources
        pinecone_index = initialize_pinecone()
        qa_tokenizer, qa_model, embedding_model = initialize_models()
        
        # Generate embeddings using sentence-transformers
        query_embedding = get_embeddings(user_question, embedding_model)
        
        # Retrieve relevant documents
        results = search_pinecone(query_embedding, pinecone_index)
        
        # Generate answer
        answer = generate_answer(user_question, results, qa_tokenizer, qa_model)
        
        # Display the answer
        st.write("### Answer:")
        st.write(answer)
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
