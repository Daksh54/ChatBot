# 💬 TalkTonic (AI ChatApp)

An intelligent **AI-powered chat application** built with **FastAPI**, **Groq LLM**, and **FAISS**.  
TalkTonic enables users to upload documents (PDF, Word, Text, or Images) and interact with their content conversationally using natural language.  
It implements **Retrieval-Augmented Generation (RAG)** to deliver fast, context-aware, and factual responses.

---

## 🧠 Overview

This backend serves as the core of an **AI chat system** that can:
- Read and process documents (PDF, DOCX, TXT, and image-based text)
- Extract meaningful chunks and create embeddings for efficient semantic search
- Use **FAISS vector search** to find relevant content
- Use **Groq LLM (LLaMA 3.3 - 70B Versatile)** for accurate context-based responses
- Maintain chat history with timestamps in a SQLite database

---

## ⚙️ Features

✅ Upload multi-format files (PDF, DOCX, TXT, Images)  
✅ Extract and clean text automatically  
✅ OCR (Optical Character Recognition) for scanned or image-based documents  
✅ Chunking and semantic vectorization using Sentence Transformers  
✅ FAISS vector indexing for fast retrieval  
✅ Context-aware chat generation with Groq LLM  
✅ Persistent chat memory via SQLite  
✅ REST API built on **FastAPI**  
✅ CORS enabled for frontend integration  
✅ Lightweight, modular, and production-ready backend design  

---

## 🧩 Tech Stack

| Layer | Technology |
|-------|-------------|
| **Backend Framework** | FastAPI |
| **Database** | SQLite (SQLAlchemy ORM) |
| **LLM API** | Groq (LLaMA 3.3 70B Versatile) |
| **Vector Search** | FAISS |
| **Embeddings** | Sentence Transformers (all-MiniLM-L6-v2) |
| **OCR Engine** | PyTesseract |
| **Document Parsing** | pdfplumber, python-docx, Pillow |
| **Language** | Python 3.10+ |


