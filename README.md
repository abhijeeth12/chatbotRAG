# 📄 Smart Document Analyzer with LLMs, Embeddings, and Heading-Aware Image Extraction

An intelligent document ingestion and retrieval backend built using **Flask**, **Sentence Transformers**, **LLMs (via Ollama)**, and **advanced PDF processing** techniques.  
This project focuses on **heading-aware text chunking**, **semantic search**, **automated keyword extraction**, and **image-heading association**, enabling a novel and rich document understanding experience.

---

## ✨ Key Features

- **PDF Uploading and Text Extraction**
  - Handles large PDFs with error resilience.
  - Extracts clean text using a hybrid of `pdfplumber`, `PyMuPDF`, and `PyPDF2`.

- **Heading-Aware Chunking**
  - Instead of naive splitting, the system **identifies document headings** using regular expressions and **groups text under headings**.
  - If no headings are found, it defaults to intelligent paragraph chunking.

- **Automatic Keyword Extraction**
  - After entering the query, **5–7 meaningful keywords** are **dynamically generated** which LLM thinks should present in the answer.
  - These keywords are later used to **enhance retrieval performance**.

- **Semantic Similarity Search**
  - Embeddings are generated using **Sentence Transformers** (`all-MiniLM-L6-v2`).
  - User queries are compared **both against document chunks** and **against extracted keywords** using **cosine similarity**.
  - Ensures retrieval of **the most contextually relevant answers**, even for vague or short queries.

- **Nearest-Heading Image Association**
  - Images extracted from PDFs using **PyMuPDF** are automatically **linked to their nearest headings** based on spatial coordinates.
  - Provides better context for document images during retrieval or visualization.

- **Efficient LLM Querying with Debugging**
  - Sends optimized prompts to **LLMs (like Llama3)** hosted on **Ollama** server.
  - Adds robust **debug logging** for prompt-response lifecycle, making development and troubleshooting faster.

- **Persistent Storage**
  - Uploaded documents, metadata, and embeddings are saved locally in JSON, making the backend restart-safe.

---

## 🚀 Tech Stack

| Technology | Purpose |
| :--- | :--- |
| Flask | Web server and API |
| PyMuPDF (fitz) | Advanced PDF parsing and image extraction |
| pdfplumber | Text extraction from PDFs |
| PyPDF2 | Backup PDF reader |
| Sentence-Transformers | Embedding generation for semantic search |
| Ollama | Local LLM inference server |
| NumPy | Embedding manipulation |
| Scikit-learn | Cosine similarity calculation |

---

## 🔥 Unique Innovations in this Project

- 📚 **Heading-Based Chunking**: Meaningful grouping of text improves LLM prompt quality and semantic search performance.
- 🧠 **Keyword Generation + Semantic Matching**: Dynamically extracted **5–7 keywords** are compared with chunks using **Sentence Transformers** to catch **most relevant chunks** increasing it's accuracy.
- 🖼️ **Heading-Aware Image Extraction**: Instead of just dumping images, each image is tied to its **nearest text heading**, enriching the search and retrieval context.
- 🛠️ **Optimized Debuggable LLM Interactions**: Prompt timings, truncated previews, and error logs are available during LLM interaction — useful for scaling and tuning.
- ⚡ **Resilient Upload and Processing Pipeline**: Detailed error handling ensures corrupted or large PDFs don't crash the system.

---

## 📂 API Endpoints

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/upload` | POST | Upload one or more PDFs. Parses, chunks, extracts keywords and images. |
| `/query` | POST | Accepts a user query, retrieves top relevant document chunks + keywords, sends context to LLM, and returns answer. |

---

## 🛠️ Setup Instructions

1. Clone the repository
2. Install required packages
   ```bash
   pip install -r requirements.txt
