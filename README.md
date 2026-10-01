# 📚 Gemini RAG · Streamlit

A **Retrieval-Augmented Generation (RAG)** application that allows users to upload a PDF and ask questions about its content.

The application uses **Google Gemini** for embeddings and answer generation, **ChromaDB** as the vector database, and **Streamlit** for the user interface.

## 🚀 Live Demo

🔗 **Try the application:**  
https://nefwx8eb2sxf5pawvfjizf.streamlit.app/

---

## ✨ Features

- 📄 Upload PDF documents
- 🔍 Extract text from PDF files
- ✂️ Split documents into smaller chunks
- 🧠 Generate embeddings using Gemini
- 🗄️ Store embeddings in ChromaDB
- 🔎 Perform semantic similarity search
- 🤖 Generate answers using Gemini
- 📖 Display retrieved document context
- 🎨 Clean Streamlit interface
- ☁️ Deployable on Streamlit Community Cloud

---

## 🏗️ How It Works

The application follows a simple RAG pipeline:

```text
              📄 PDF Upload
                    │
                    ▼
            Text Extraction
                    │
                    ▼
              Text Chunking
                    │
                    ▼
          Gemini Embeddings
                    │
                    ▼
              ChromaDB
             Vector Storage
                    │
                    │
         User asks a question
                    │
                    ▼
          Question Embedding
                    │
                    ▼
        Similarity Search
             in ChromaDB
                    │
                    ▼
          Relevant Chunks
                    │
                    ▼
             Gemini LLM
                    │
                    ▼
              🤖 Answer
