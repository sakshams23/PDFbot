import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
import google.generativeai as genai
from langchain_community.vectorstores.faiss import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import streamlit as st
from PIL import Image

def main():
    st.set_page_config(
        page_title="PDFbot",
        page_icon="",
        layout="wide",  # full-screen layout
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
                    # Backend logic should be called here to process PDFs
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

    # Display chat messages with custom styling
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f"<div style='text-align: right; color: #ffffff;'>{message['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='text-align: left; color: #ffffff;'>{message['content']}</div>", unsafe_allow_html=True)

    # Input chat prompt
    user_prompt = st.chat_input("Ask a question about the uploaded PDFs")

    if user_prompt:
        # Append user message
        st.session_state.messages.append({"role": "user", "content": user_prompt})
        # Display user input immediately
        with st.chat_message("user"):
            st.write(user_prompt)

        # Process the response
        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    # Backend logic should process the response
                    response = "Mock response based on the PDF content."
                    st.write(response)

            # Save response in chat history
            if response:
                st.session_state.messages.append({"role": "assistant", "content": response})

    # Sidebar Button: Clear Chat History
    st.sidebar.button("Clear Chat History", on_click=lambda: st.session_state.clear())

if __name__ == "__main__":
    main()
