import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline
from pinecone import Pinecone

st.set_page_config(
    page_title="INCOS AI Assistant",
    page_icon="🎓",
    layout="centered"
)

@st.cache_resource
def initialize_qa_chain():
    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    
    api_key = "pcsk_2uxcgr_7EXRxqcQDew4CqgB2B9Q1M9EgwqpPCw4HAL7wjcLgHSN7g6ToZoAnEtBvjsHA3J"
    pc = Pinecone(api_key=api_key)
    index = pc.Index("corpus-embeddings")
    
    vectorstore = Pinecone(
        index,
        embeddings.embed_query,
        "text"  # Name of the text field in your metadata
    )
    
    # Initialize QA model
    qa_pipeline = pipeline("question-answering", model="Sai-Harsha-k/incosai_qa_finetuned_model")
    llm = HuggingFacePipeline(pipeline=qa_pipeline)
    
    # Create retrieval QA chain 
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    
    return qa_chain

# Streamlit App Interface
st.title("INCOS AI Assistant")
st.write("How can I help you with your coursework?")

# User Input
user_question = st.text_input("Ask me anything related to your coursework!")

if user_question:
    try:
        qa_chain = initialize_qa_chain()
        answer = qa_chain.run(user_question)
        
        st.write("### Answer:")
        st.write(answer)
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
