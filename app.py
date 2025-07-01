import os
import streamlit as st
from PyPDF2 import PdfReader
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain

st.header("My First ChatBot")

# Upload PDF
with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file and start asking questions", type="pdf")

# Process the uploaded file
if file is not None:
    # Extract text from PDF
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()


    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=500,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)


    # Generate embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create FAISS vector store
    vector_store = FAISS.from_texts(chunks, embeddings)

    # Get user question
    user_question = st.text_input("Ask your question")

    if user_question:
        # Perform similarity search
        match = vector_store.similarity_search(user_question)

        # Load transformers pipeline for QA
        qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

        # Wrap with LangChain's HuggingFacePipeline
        llm = HuggingFacePipeline(pipeline=qa_pipeline)

        # Load QA chain
        chain = load_qa_chain(llm, chain_type="stuff")

        # Run chain
        response = chain.run(input_documents=match, question=user_question)

        # Show result
        st.write(response)
