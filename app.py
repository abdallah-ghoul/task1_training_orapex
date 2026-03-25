import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()

# Page setup
st.set_page_config(page_title="Orapex RAG App", page_icon="🤖")
st.title("🤖 Orapex Intern: My First RAG App")
st.write("Ask me anything about Orapex!")

@st.cache_resource
def setup_rag_pipeline():
    # A. LOAD
    loader = TextLoader("sample.txt")
    documents = loader.load()

    # B. CHUNK
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)

    # C. EMBED & STORE
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(chunks, embeddings)

    # D. RETRIEVER + CHAIN
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    return chain

# Setup pipeline
qa_chain = setup_rag_pipeline()

# Chat input
question = st.text_input("Ask a question:")

if question:
    with st.spinner("Thinking..."):
        response = qa_chain.invoke({"query": question})
        st.success(response["result"])