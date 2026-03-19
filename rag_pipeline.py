import os
import git
import shutil
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

groq_api_key = st.secrets.get("GROQ_API_KEY")

repo_path = "/tmp/repo"
# chroma_path = "/tmp/chroma_db"

# Clone GitHub repository
def clone_repo(repo_url):
    if os.path.exists(repo_path):
        shutil.rmtree(repo_path)  # delete old repo
    
    # if os.path.exists(chroma_path):
    #     shutil.rmtree(chroma_path)

    git.Repo.clone_from(repo_url, repo_path)
    return repo_path


# Load repository files
def load_files(repo_path):

    docs = []

    for root, dirs, files in os.walk(repo_path):

        # ignore unnecessary folders
        dirs[:] = [d for d in dirs if d not in [
            ".git",
            "node_modules",
            "__pycache__",
            "dist",
            "build",
            "venv"
        ]]

        for file in files:

            if file.endswith((
                ".py",
                ".js",
                ".ts",
                ".jsx",
                ".tsx",
                ".java",
                ".cpp",
                ".go",
                ".rs",
                ".md"
            )):

                try:

                    loader = TextLoader(
                        os.path.join(root, file),
                        encoding="utf-8"
                    )

                    documents = loader.load()

                    for doc in documents:
                        doc.metadata["source"] = os.path.join(root, file)

                    docs.extend(documents)

                except:
                    pass

    return docs


# Create vector database
def create_vector_store(docs):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )

    split_docs = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_db = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory = "/tmp/chroma_db"
    )

    return vector_db


# Format retrieved docs
def format_docs(docs):

    return "\n\n".join(doc.page_content for doc in docs)


# Create RAG chain
def create_rag_chain(vector_db):

    retriever = vector_db.as_retriever(search_kwargs={"k": 4})

    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model="openai/gpt-oss-20b"
    )

    prompt = ChatPromptTemplate.from_template(
"""
You are an expert software engineer helping a developer understand a GitHub repository.

You can answer using:
1. Repository code context
2. Conversation history

If the question is about the repository, use the repository context.

If the question is about previous messages (like "summarize what I asked"),
use the chat history.

<context>
{context}
</context>

Chat History:
{chat_history}

User Question:
{question}

Answer clearly and explain code if needed.
"""
)

    rag_chain = (
            {
        "context": lambda x: format_docs(retriever.invoke(x["question"])),
        "question": lambda x: x["question"],
        "chat_history": lambda x: x["chat_history"]
    }

        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever