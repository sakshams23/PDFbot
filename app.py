import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import streamlit as st
from dotenv import load_dotenv

def extract_text_from_pdfs(pdf_docs):
    text = ""
    for pdf_file in pdf_docs:
        reader = PdfReader(pdf_file)
        for page in reader.pages:
            text += page.extract_text()
    return text

def create_faiss_index(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_text(text)
    embeddings = GoogleGenerativeAIEmbeddings()
    return FAISS.from_texts(docs, embeddings)

def main():
    st.set_page_config(
        page_title="PDFbot",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Sidebar for uploading PDF files
    with st.sidebar:
        st.title("Upload PDF Files")
        st.info("You can upload multiple PDF files to chat with.")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files",
            accept_multiple_files=True,
            type=["pdf"]
        )

        if st.button("Submit"):
            with st.spinner("Processing PDFs..."):
                if pdf_docs:
                    text = extract_text_from_pdfs(pdf_docs)
                    st.session_state.faiss_index = create_faiss_index(text)
                    st.success("PDFs processed successfully!")
                else:
                    st.error("Please upload PDF files first.")

    # Main content area for displaying chat messages
    st.title("Chat with PDF Files")
    st.write("Welcome! Upload your PDF files and ask questions based on the document contents.")

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Upload some PDFs and ask me a question!"}
        ]

    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"<div style='text-align: right;'>{message['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='text-align: left;'>{message['content']}</div>", unsafe_allow_html=True)

    user_prompt = st.chat_input("Ask a question about the uploaded PDFs")

    if user_prompt:
        st.session_state.messages.append({"role": "user", "content": user_prompt})

        if "faiss_index" in st.session_state:
            with st.spinner("Thinking..."):
                retriever = st.session_state.faiss_index.as_retriever()
                chain = load_qa_chain(ChatGoogleGenerativeAI(), chain_type="stuff")
                response = chain.run(input_documents=retriever.get_relevant_documents(user_prompt), question=user_prompt)
        else:
            response = "Please upload and process PDFs first."

        st.session_state.messages.append({"role": "assistant", "content": response})
        st.write(response)

    st.sidebar.button("Clear Chat History", on_click=lambda: st.session_state.clear())

if __name__ == "__main__":
    main()
