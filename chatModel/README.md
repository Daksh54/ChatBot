# TalkTonic-RAG-Implemented

TalkTonic is an intelligent file-aware chatbot built with Retrieval-Augmented Generation (RAG).  
It allows users to upload documents (PDFs, images, DOCX, TXT), extracts their content using OCR & parsing, chunks the text, creates vector embeddings, and enables conversational Q&A over the content.

---

## ✨ Features

- 📄 Supports **PDF, Image (OCR), DOCX, and TXT**
- 🧠 Uses paraphase miniLM for semantic understanding
- 📚 Document chunking for context-aware retrieval
- 🔍 FAISS for efficient similarity search
- 💬 Chat-style Q&A interface with **Groq API (LLaMA 3)** as the LLM
- 🧾 Summary / query input post-upload + persistent chat history
- 🖼️ Smart OCR fallback for scanned documents using `Tesseract`
- ☁️ Deployed on **Render Free Tier**

---

## 🔧 Tech Stack

| Component        | Tech Used                     |
|------------------|-------------------------------|
| Frontend UI      | Streamlit                     |
| Embedding Model  | paraphaseMiniLM               |
| Vector Store     | FAISS                         |
| OCR Engine       | Tesseract OCR (`pytesseract`) |
| PDF Parsing      | `pdfplumber`                  |
| DOCX Handling    | `python-docx`                 |
| Chat LLM         | Groq API (`llama3-8b-8192`)   |
| Deployment       | Render (Free Tier)            |

---

### 📦 Install Dependencies

```bash
pip install -r requirements.txt
