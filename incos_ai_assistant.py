import streamlit as st
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
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
    # Replace with your Hugging Face model name
    model_name = "Sai-Harsha-k/incosai_qa_finetuned_model"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    return tokenizer, model

# Utility function to generate embeddings
def get_embeddings(text: str, tokenizer, model) -> np.ndarray:
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model.encoder(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()

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
    prompt = f"Question: {question}\nContext: {context_texts}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model.generate(**inputs, max_length=150)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


# Streamlit App Interface
st.title("INCOS AI Assistant")
st.write("How can I help you with your coursework?")

# User Input
user_question = st.text_input("Ask me anything related to your coursework!")

if user_question:
    # Initialize resources
    pinecone_index = initialize_pinecone()
    tokenizer, model = initialize_models()

    # Generate embeddings for the question
    query_embedding = get_embeddings(user_question, tokenizer, model)

    # Retrieve relevant documents from Pinecone
    results = search_pinecone(query_embedding, pinecone_index)

    # Display retrieved context
    #st.write("### Retrieved Context:")
    for i, doc in enumerate(results["matches"]):
        st.write(f"{i+1}. {doc['metadata']['text']}")

    # Generate an answer
    answer = generate_answer(user_question, results, tokenizer, model)

    # Display the generated answer
    st.write("### Answer:")
    st.write(answer)
