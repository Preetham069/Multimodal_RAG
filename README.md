# Multimodal RAG System

A multimodal Retrieval-Augmented Generation (RAG) system that can retrieve and reason over both **PDF documents and images** using **Google Gemini, LangChain, and ChromaDB**.

The system accepts natural-language queries through a Streamlit interface, determines whether the query requires PDF information, image information, or both, retrieves the relevant information from the vector database, and generates an answer using Gemini.

---

## 🚀 Project Overview

Traditional RAG systems mainly work with text-based documents.

This project extends the RAG approach to support multiple data modalities:

- 📄 PDF documents
- 🖼️ Images
- 🔎 Semantic retrieval
- 🤖 Gemini-based answer generation
- 🧠 Query-aware multimodal retrieval
- 💾 Persistent ChromaDB vector storage
- 🌐 Streamlit user interface

The current system uses a query classifier to determine whether a question requires:

- PDF retrieval
- Image retrieval
- Multimodal retrieval

The next planned stage is to replace this rule-based routing with an **Agentic RAG architecture**, allowing an AI agent to decide which tools and information sources are required for each query.

---

## 🏗️ System Architecture

```text
                    ┌──────────────────────┐
                    │      User Query      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │   Streamlit UI       │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Query Classification │
                    └──────────┬───────────┘
                               │
                ┌──────────────┼──────────────┐
                │              │              │
                ▼              ▼              ▼
             PDF Query     Image Query    Multimodal
                │              │              │
                └──────────────┼──────────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      ChromaDB        │
                    │   Vector Database    │
                    └──────────┬───────────┘
                               │
                     Relevant Context
                               │
                               ▼
                    ┌──────────────────────┐
                    │    Google Gemini     │
                    │   Multimodal LLM     │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │    Generated Answer   │
                    └──────────────────────┘


 🔄 RAG Pipeline

1. PDF Processing
PDF Document
     ↓
PyMuPDF
     ↓
Page Text Extraction
     ↓
Recursive Character Text Splitter
     ↓
Text Chunks
     ↓
Gemini Embedding 2
     ↓
ChromaDB

2. Image Processing
Image
     ↓
Pillow
     ↓
Gemini Embedding 2
     ↓
ChromaDB

3. Query Processing
User Query
     ↓
Gemini Embedding 2
     ↓
Semantic Similarity Search
     ↓
Relevant PDF / Image Results
     ↓
Context Construction
     ↓
Google Gemini
     ↓
Final Answer


🔎 Query-Aware Retrieval

The system currently identifies three types of queries.

PDF Query

Example:

What technologies are used in the proposed system?

The system retrieves relevant PDF chunks and uses them as context for the answer.

Image Query

Example:

What is shown in the image?

The system retrieves the relevant image and uses it for multimodal answer generation.

Multimodal Query

Example:

What is shown in the image, and what are the main technologies discussed in the PDF?

The system retrieves both PDF and image information and provides both as context to Gemini.



🧩 Retrieval Improvements

The retrieval system includes several improvements to make the results more useful:

Query-aware retrieval
PDF-only retrieval
Image-only retrieval
Multimodal retrieval
Query embedding reuse
Duplicate content filtering
Page-diversified PDF retrieval
Multiple candidate retrieval before final selection
Configurable retrieval count
Gemini API retry handling for temporary service unavailability



🛠️ Technologies Used

Technology	Purpose
Python	Core programming language
LangChain	RAG orchestration and text splitting
Google Gemini	Embeddings and answer generation
Gemini Embedding 2	Text and image embeddings
ChromaDB	Vector database
PyMuPDF	PDF processing
Pillow	Image processing
Streamlit	Web application interface
python-dotenv	Environment variable management


⚙️ Installation
1. Clone the Repository
git clone https://github.com/Preetham069/Multimodal_RAG.git
cd Multimodal_RAG
2. Create a Virtual Environment
python -m venv venv
3. Activate the Virtual Environment

For Windows PowerShell:

.\venv\Scripts\Activate.ps1

If PowerShell prevents activation, run:

Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

Then activate again:

.\venv\Scripts\Activate.ps1
4. Install Dependencies
pip install langchain langchain-community langchain-google-genai google-genai chromadb pypdf unstructured pillow python-dotenv streamlit pymupdf langchain-text-splitters
🔑 Gemini API Configuration

The project uses Google Gemini for:

Text embeddings
Image embeddings
Answer generation

Create a .env file in the project root:

GEMINI_API_KEY=your_gemini_api_key_here

The .env file is excluded from Git using .gitignore.

Never commit your actual API key to GitHub.

🗄️ Build the Vector Database

After placing the required PDF and image files inside the data directory, run:

python build_vector_database.py

This will:

Load the PDF.
Extract PDF text.
Split the text into chunks.
Generate embeddings.
Generate the image embedding.
Store the embeddings in ChromaDB.

The vector database is stored locally and is excluded from Git.

▶️ Run the Application

Start the Streamlit application:

streamlit run app.py

The application will normally be available at:

http://localhost:8501

Open the displayed URL in your browser.

💬 Example Queries
Image Query
What is shown in the image?
PDF Query
What technologies are used in the proposed system?
Multimodal Query
What is shown in the image, and what are the main technologies discussed in the PDF?



🧪 Testing

The project contains separate test programs for validating individual components of the system.

Examples include:

test_gemini.py
test_pdf_loader.py
test_image_embedding.py
test_vector_store.py
test_retriever.py
test_multimodal_retrieval.py
test_rag_pipeline.py
test_image_rag_pipeline.py

The tests were used during development to verify:

Gemini API connectivity
PDF loading
PDF chunking
Text embeddings
Image embeddings
Vector database storage
Semantic retrieval
Multimodal retrieval
RAG generation
Image-based RAG
Streamlit integration



📊 Current Capabilities

The current system supports:

PDF loading
PDF page extraction
Text chunking
Text embeddings
Image embeddings
Persistent vector storage
Semantic similarity retrieval
PDF-only retrieval
Image-only retrieval
Multimodal retrieval
Query-aware retrieval
Duplicate result filtering
Page-diversified retrieval
Gemini-based answer generation
Gemini retry handling
Retrieved source display
Streamlit-based user interface