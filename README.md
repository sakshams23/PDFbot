# Streamlit Web Application for PDF Processing with LangChain

This repository contains a Streamlit-based web application that processes PDF files by extracting text, creating embeddings, and enabling an interactive chat interface for querying document content. The application leverages **LangChain** for advanced document processing and querying.

---

## Features

- **Upload PDF**: Upload machine-readable or scanned PDFs for processing.
- **Text Extraction**: Extracts text from PDFs using:
  - `PyPDF2` for machine-readable text.
  - `pytesseract` for OCR-based extraction from scanned PDFs.
- **Text Chunking**: Splits large documents into smaller, manageable chunks using LangChain's `RecursiveCharacterTextSplitter`.
- **Embeddings Creation**: Generates numerical embeddings using models like OpenAI or Google Generative AI for semantic search.
- **Vector Database Storage**: Stores embeddings in FAISS for fast similarity-based search.
- **Interactive Chat Interface**: Ask questions about the PDF content and receive context-aware responses.
- **Clear Chat History**: Option to reset the chat for a new session.

---

## Workflow

### 1. **Upload PDF**
- Users can upload a PDF file via the Streamlit interface.
- The application prepares the file for processing.

### 2. **Text Extraction**
- **PDF Text Extraction**:
  - For machine-readable PDFs, `PyPDF2` extracts text directly.
- **PDF OCR**:
  - For scanned PDFs, OCR is performed using `pytesseract` to extract text.

### 3. **Text Processing with LangChain**
LangChain enhances document processing and querying with the following steps:

- **Split Text**: 
  - Text is divided into smaller chunks (e.g., 1000 characters) with overlaps (e.g., 200 characters) using `RecursiveCharacterTextSplitter`.
- **Create Embeddings**:
  - Each chunk is converted into a semantic embedding using LangChain's embedding models (e.g., OpenAI or GoogleGenerativeAI).
- **Store in Vector Database**:
  - Embeddings are stored in a FAISS vector database for fast retrieval.

### 4. **User Interaction and LangChain's Conversational Flow**
LangChain powers the question-answering flow with these steps:
1. **Perform Similarity Search**:
   - Queries are processed to retrieve the most relevant chunks from the vector database.
2. **Question-Answering Chain**:
   - LangChain's `load_qa_chain` combines the query and relevant chunks to generate context-aware responses using AI models like GPT.
3. **Display Response**:
   - The response is displayed in an intuitive chat interface.

### 5. **Interactive Features**
- **Chat Interface**: Users can ask questions about the uploaded document and receive detailed answers.
- **Clear History**: Reset the interaction history for a fresh start.

---

## Benefits of LangChain Integration

- **Efficient Search**: Quickly retrieves relevant document sections based on user queries.
- **Context-Aware Responses**: Generates precise answers using the document's context.
- **Scalable Workflow**: Handles large PDFs effectively using embeddings and vector search.

---

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Required libraries:
  - `streamlit`
  - `langchain`
  - `pypdf2`
  - `pytesseract`
  - `faiss-cpu` (or GPU version)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/pdf-processing-langchain.git
   cd pdf-processing-langchain
