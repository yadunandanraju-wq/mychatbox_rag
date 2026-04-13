import streamlit as st
import os

# Import LangChain and related modules
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Page configuration
st.set_page_config(
    page_title="Question Answering with Groq Chat",
    page_icon="🤖",
    layout="wide"
)

st.title("Question Answering with Groq Chat")

# ============================================================================
# SIDEBAR: API Key Input (Password Protected)
# ============================================================================
with st.sidebar:
    st.header("Configuration")
    
    # API Key input with password masking
    groq_api_key = st.text_input(
        "Enter your Groq API Key:",
        type="password",
        help="Your API key will be used only for this session and not stored."
    )
    
    if groq_api_key:
        os.environ['GROQ_API_KEY'] = groq_api_key
        st.success("✅ API Key set successfully")
    else:
        st.warning("⚠️ Please enter your Groq API Key to proceed")

# ============================================================================
# MAIN APP: Initialize RAG Chain
# ============================================================================

def initialize_rag_chain():
    """Initialize the RAG chain with embeddings, vectorstore, and LLM"""
    try:
        # Load CSV document
        loader = CSVLoader(r'fake_startup_founders_europe.csv', encoding="latin-1")
        doc = loader.load()
        
        # Create embeddings for all the documents
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Store embeddings in vectordatabase
        vectorstore = FAISS.from_documents(doc, embeddings)
        
        # Initialize LLM with Groq
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
        )
        
        # Setting up the retrieval function using modern LCEL approach
        template = """Answer the question based only on the following context:

{context}

Question: {question}
"""
        
        prompt = ChatPromptTemplate.from_template(template)
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        retriever = vectorstore.as_retriever()
        
        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        return chain
    except Exception as e:
        st.error(f"Error initializing RAG chain: {str(e)}")
        return None

# Initialize chain only if API key is provided
if groq_api_key:
    chain = initialize_rag_chain()
    
    if chain:
        # User input for the query
        question = st.text_input("Enter your question:")
        
        if st.button("Submit") and question:
            with st.spinner("Processing..."):
                result = chain.invoke(question)
            st.write("**Result:**")
            st.write(result)
else:
    st.info("👈 Please enter your Groq API Key in the sidebar to get started.")
