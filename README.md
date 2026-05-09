# GitHub Repository Chat Assistant 🤖📚

### RAG-Based GitHub Repository Chatbot using LangChain, Groq LLM & Streamlit

<p align="center">
  <img src="https://img.shields.io/badge/Python-Backend-blue?style=for-the-badge&logo=python" />
  <img src="https://img.shields.io/badge/LangChain-RAG-green?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Groq-LLM-orange?style=for-the-badge" />
  <img src="https://img.shields.io/badge/Streamlit-Frontend-red?style=for-the-badge&logo=streamlit" />
  <img src="https://img.shields.io/badge/Vector%20Embeddings-AI-purple?style=for-the-badge" />
</p>

---

# 📌 Overview

**GitHub Repository Chat Assistant** is a RAG (Retrieval-Augmented Generation) based AI chatbot that allows users to interact with any GitHub repository using natural language.

Users simply paste a GitHub repository URL, and the system:

* Loads repository files
* Processes the codebase
* Creates embeddings from the content
* Stores vector representations
* Allows users to ask questions about the repository

The chatbot uses:

* **LangChain** for orchestration
* **Groq LLM** for fast AI responses
* **Embeddings + Vector Search** for retrieval
* **Streamlit** for the interactive UI

This project helps developers quickly understand unfamiliar codebases without manually reading every file.

---

# ✨ Features

## 🔗 GitHub Repository Analysis

* Enter any GitHub repository URL
* Automatically fetch repository files
* Parse and process codebase content

## 🧠 RAG-Based AI Chat

* Retrieval-Augmented Generation pipeline
* Context-aware repository understanding
* Ask questions about code structure, logic, and functionality

## 📚 Embedding Generation

* Converts repository content into embeddings
* Stores vector representations for semantic search
* Faster and more accurate retrieval

## ⚡ Groq LLM Integration

* High-speed AI responses
* Contextual code explanations
* Intelligent repository querying

## 🖥️ Interactive Streamlit UI

* Simple and clean interface
* Real-time responses
* Easy repository input workflow

---

# 🛠️ Tech Stack

| Technology              | Purpose                |
| ----------------------- | ---------------------- |
| Python                  | Core Development       |
| Streamlit               | Frontend Interface     |
| LangChain               | RAG Pipeline           |
| Groq LLM                | AI Response Generation |
| FAISS                   | Vector Database        |
| HuggingFace Embeddings  | Embedding Generation   |

---

# ⚙️ How It Works

## 1️⃣ User Enters GitHub Repository URL

The system accepts a public GitHub repository link.

---

## 2️⃣ Repository Files Are Loaded

The application:

* Clones/fetches repository files
* Reads code and documentation
* Splits content into chunks

---

## 3️⃣ Embeddings Are Created

The text/code chunks are converted into vector embeddings for semantic search.

---

## 4️⃣ Vector Database Storage

Embeddings are stored inside a vector database like:

* FAISS

---

## 5️⃣ User Asks Questions

Users can ask:

* “What does this project do?”
* “Explain the authentication flow”
* “Which file handles API routes?”
* “How is the database connected?”
* “Summarize the repository”

---

## 6️⃣ RAG Pipeline Generates Answers

Relevant chunks are retrieved and passed to the Groq LLM to generate contextual answers.

---

# 📸 Screenshots

## Home Interface

<p align="center">
  <img width="1856" height="919" alt="image" src="https://github.com/user-attachments/assets/cb352a89-56bf-4916-8c73-a6ccb0d12216" />

</p>

---

# 👨‍💻 Author

## Atharv Ragdwal

* 💻 Full Stack Developer
* 🤖 AI & LLM Enthusiast
* 🚀 Passionate About Building Intelligent Developer Tools

---

# 🌟 Support

If you liked this project:

⭐ Star the repository
🍴 Fork the project

---

