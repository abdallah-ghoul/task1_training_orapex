import os
import streamlit as st
from dotenv import load_dotenv

# LangChain components
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# NEW: Import Google Gemini components instead of OpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

# 1. Load environment variables (gets the GOOGLE_API_KEY from the .env file)
load_dotenv()

# Set up the Streamlit UI
st.title("🤖 Orapex Intern: My First Gemini RAG App")
st.write("Ask a question based on the custom document!")

@st.cache_resource 
def setup_rag_pipeline():
    # --- A. LOAD ---
    loader = TextLoader("sample.txt")
    documents = loader.load()

    # --- B. CHUNK ---
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    # --- C. EMBED & STORE ---
    # NEW: Use Google's embedding model 
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vectorstore = Chroma.from_documents(chunks, embeddings)

    # --- D. RETRIEVE & GENERATE ---
    # NEW: Use Gemini 1.5 Flash (fast, smart, and cost-effective)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    
    # Create the RAG Chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", 
        retriever=vectorstore.as_retriever()
    )
    return qa_chain

# Ensure the file exists before running
if os.path.exists("sample.txt"):
    qa_chain = setup_rag_pipeline()

    user_query = st.text_input("Enter your question:")

    if user_query:
        with st.spinner("Searching database and generating answer with Gemini..."):
            response = qa_chain.invoke({"query": user_query})
            
            st.success("Done!")
            st.write("### Answer:")
            st.write(response["result"])
else:
    st.error("⚠️ Please create a 'sample.txt' file in the same directory as this script.")