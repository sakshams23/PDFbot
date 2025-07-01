import os
import io
import PyPDF2
import pytesseract
from PIL import Image
from dotenv import load_dotenv
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain

# --------------------- Load API Key ---------------------
load_dotenv()  # For local dev, loads .env
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", st.secrets.get("GOOGLE_API_KEY"))
if not GOOGLE_API_KEY:
    st.error("Google API Key is missing.")
    st.stop()

from google import generativeai as gen_ai
gen_ai.configure(api_key=GOOGLE_API_KEY)

# --------------------- Session State ---------------------
if 'vector_store' not in st.session_state:
    st.session_state['vector_store'] = None
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# --------------------- PDF Extraction ---------------------
def extract_text_from_pdfs(file_bytes_list, ocr=False):
    text = ""
    for file_bytes in file_bytes_list:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        for page in pdf_reader.pages:
            page_text = page.extract_text() or ""
            text += page_text
            if ocr and hasattr(page, 'images'):
                for image in page.images:
                    image_bytes = image.data
                    image_file = io.BytesIO(image_bytes)
                    text += pytesseract.image_to_string(Image.open(image_file)) + "\n"
    return text

# --------------------- Text Chunking ---------------------
def split_text_into_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

# --------------------- Vector Store Creation ---------------------
def create_vector_store(chunks):
    if not chunks:
        st.warning("No text content found in the uploaded file.")
        return
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        st.session_state['vector_store'] = vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")

# --------------------- Chat Chain Setup ---------------------
def setup_conversational_chain():
    prompt_template = """
    Answer the question in a short, precise, detailed, friendly and engaging way, drawing from the provided context if possible.
    Use simple, human-like English. If the question is not related to the context, answer based on general knowledge.

    Context: {context}
    Question: {question}

    Answer:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # ✅ No need to pass `client=gen_ai`
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain

# --------------------- Get Bot Response ---------------------
def get_response(user_question):
    vector_store = st.session_state['vector_store']
    if not vector_store:
        return {"output_text": ["No documents found."]}
    
    relevant_docs = vector_store.similarity_search(user_question)
    conversational_chain = setup_conversational_chain()
    response = conversational_chain(
        {"input_documents": relevant_docs, "question": user_question},
        return_only_outputs=True
    )
    return response

# --------------------- Streamlit UI ---------------------
def main():
    st.set_page_config(page_title="Talk to PDF Bot", page_icon="📘", layout="wide")
    st.sidebar.title("Upload PDF Files")
    uploaded_files = st.sidebar.file_uploader("Upload your PDF files", type=["pdf"], accept_multiple_files=True)
    ocr = st.sidebar.checkbox("Enable OCR for images in PDFs")

    if st.sidebar.button("Clear Conversation"):
        st.session_state['messages'] = []
        st.session_state['vector_store'] = None
        st.experimental_rerun()

    if uploaded_files:
        file_bytes_list = [file.read() for file in uploaded_files]
        with st.spinner("Processing files..."):
            file_text = extract_text_from_pdfs(file_bytes_list, ocr)
            chunks = split_text_into_chunks(file_text)
            create_vector_store(chunks)
        st.success("Files processed successfully!")

    st.title("Talk to PDF Bot 📘")

    if st.session_state['vector_store']:
        st.write("Ask a question about the uploaded PDFs.")

        for msg in st.session_state['messages']:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        if prompt := st.chat_input("Type your question here..."):
            st.session_state['messages'].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = get_response(prompt)
                    full_response = ''.join(response.get('output_text', ["[Error generating response]"]))
                    st.write(full_response)
                    st.session_state['messages'].append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()
