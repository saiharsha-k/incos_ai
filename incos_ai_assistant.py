import streamlit as st
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import numpy as np

# Streamlit page configuration
st.set_page_config(
    page_title="Incos AI Assistant",
    page_icon="🕵️",
    layout="centered"
)

# Initialize Hugging Face finetuned model and tokenizer
@st.cache_resource
def initialize_finetuned_model():
    model_name = "Sai-Harsha-k/incosai_qa_finetuned_model"  # Replace with your Hugging Face model name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    return tokenizer, model

# Initialize Pinecone
@st.cache_resource
def initialize_pinecone():
    pc = Pinecone(api_key="pcsk_2uxcgr_7EXRxqcQDew4CqgB2B9Q1M9EgwqpPCw4HAL7wjcLgHSN7g6ToZoAnEtBvjsHA3J")
    index = pc.Index("corpus-embeddings")
    return index

# Function to retrieve context from Pinecone
def retrieve_context(query_embedding, pinecone_index, top_k=3):
    results = pinecone_index.query(
        vector=query_embedding.tolist(),
        top_k=top_k,
        include_metadata=True
    )
    return results["matches"] if "matches" in results else []

# Function to generate answer using the fine-tuned model
def generate_answer(question, context, tokenizer, model):
    context_text = " ".join([match["metadata"]["content"] for match in context])
    if not context_text:
        return "No relevant context found. Please try rephrasing your question."
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    inputs = tokenizer(question, context_text, return_tensors="pt", padding=True, truncation=True).to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        answer_start = outputs.start_logits.argmax()
        answer_end = outputs.end_logits.argmax() + 1
        answer = tokenizer.convert_tokens_to_string(
            tokenizer.convert_ids_to_tokens(inputs.input_ids[0][answer_start:answer_end])
        )
    return answer

# Main application
st.title("Incos AI Assistant")
st.write("Ask a question that I can help you with!")

# Initialize models and Pinecone
tokenizer, model = initialize_finetuned_model()
pinecone_index = initialize_pinecone()

# Input for user question
user_question = st.text_input("Enter your question:")

if user_question:
    # Generate embeddings for the user question using SentenceTransformer
    sentence_transformer = SentenceTransformer("all-MiniLM-L6-v2")
    query_embedding = sentence_transformer.encode(user_question)

    # Retrieve relevant contexts from Pinecone
    st.write("Searching for relevant context...")
    retrieved_contexts = retrieve_context(query_embedding, pinecone_index)

    # Display retrieved contexts (optional for debugging)
    if retrieved_contexts:
        st.write("Relevant contexts found:")
        for idx, match in enumerate(retrieved_contexts):
            st.write(f"Context {idx + 1}: {match['metadata']['content']}")
    else:
        st.write("No relevant contexts found.")

    # Generate an answer to the user's question
    st.write("Generating answer...")
    answer = generate_answer(user_question, retrieved_contexts, tokenizer, model)

    # Display the answer
    st.subheader("Answer:")
    st.write(answer)
