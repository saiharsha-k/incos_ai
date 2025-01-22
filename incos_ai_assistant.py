import streamlit as st
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModel
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
    # QA model for answering questions
    qa_model_name = "Sai-Harsha-k/incosai_qa_finetuned_model"
    qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
    qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)
    
    # Separate model for generating embeddings (e.g., using BERT base)
    embedding_model_name = "bert-base-uncased"
    embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
    embedding_model = AutoModel.from_pretrained(embedding_model_name)
    
    return qa_tokenizer, qa_model, embedding_tokenizer, embedding_model

# Utility function to generate embeddings
def get_embeddings(text: str, tokenizer, model) -> np.ndarray:
    """Generate embeddings for input text using a language model."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        # Use mean pooling of the last hidden states
        embeddings = outputs.last_hidden_state.mean(dim=1).numpy()
    return embeddings

# Search Pinecone for relevant documents
def search_pinecone(query_embedding: np.ndarray, index, top_k: int = 3):
    return index.query(
        vector=query_embedding.tolist(),
        top_k=top_k,
        include_metadata=True
    )

# Generate an answer based on the retrieved context
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
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits)
        
        answer = tokenizer.decode(inputs["input_ids"][0][answer_start:answer_end+1])
    
    return answer

# Streamlit App Interface
st.title("INCOS AI Assistant")
st.write("How can I help you with your coursework?")

# User Input
user_question = st.text_input("Ask me anything related to your coursework!")

if user_question:
    # Initialize resources
    pinecone_index = initialize_pinecone()
    qa_tokenizer, qa_model, embedding_tokenizer, embedding_model = initialize_models()
    
    # Generate embeddings for the question using the embedding model
    query_embedding = get_embeddings(user_question, embedding_tokenizer, embedding_model)
    
    # Retrieve relevant documents from Pinecone
    results = search_pinecone(query_embedding, pinecone_index)
    
    # Generate an answer using the QA model
    answer = generate_answer(user_question, results, qa_tokenizer, qa_model)
    
    # Display the generated answer
    st.write("### Answer:")
    st.write(answer)
