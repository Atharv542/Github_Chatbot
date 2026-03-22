import streamlit as st
import time

from rag_pipeline import (
    clone_repo,
    load_files,
    create_vector_store,
    create_rag_chain
)

st.set_page_config(page_title="GitHub RAG Assistant")

# ================================
# SESSION STATE INIT
# ================================

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello 👋 Ask me anything about a GitHub repository."
        }
    ]

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "current_repo" not in st.session_state:
    st.session_state.current_repo = None


# ================================
# SIDEBAR
# ================================

st.sidebar.title("Repository Settings")

repo_url = st.sidebar.text_input("Enter GitHub Repository URL")

process_btn = st.sidebar.button("Process Repository")


# ================================
# PROCESS REPO
# ================================

if process_btn:

    if not repo_url:
        st.sidebar.warning("Please enter a GitHub repository URL")

    else:

        # 🔥 FULL RESET (IMPORTANT FIX)
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": f"Now chatting with:\n{repo_url}"
            }
        ]

        st.session_state.rag_chain = None
        st.session_state.retriever = None
        st.session_state.current_repo = repo_url

        with st.spinner("Cloning repository..."):
            path = clone_repo(repo_url)

        with st.spinner("Loading repository files..."):
            docs = load_files(path)

        st.sidebar.success(f"{len(docs)} documents loaded")

        with st.spinner("Creating embeddings..."):
            vector_db = create_vector_store(docs)

        rag_chain, retriever = create_rag_chain(vector_db)

        st.session_state.rag_chain = rag_chain
        st.session_state.retriever = retriever

        st.sidebar.success("Repository indexed successfully!")


# ================================
# UI
# ================================

st.title("GitHub Repository Chat Assistant")

if st.session_state.current_repo:
    st.caption(f"📂 Current Repo: {st.session_state.current_repo}")


# ================================
# CHAT HISTORY
# ================================

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


# ================================
# USER INPUT
# ================================

query = st.chat_input("Ask anything about the repository...")

if query:

    if not st.session_state.rag_chain:
        st.warning("Please process a GitHub repository first.")
        st.stop()

    st.session_state.messages.append(
        {"role": "user", "content": query}
    )

    st.chat_message("user").write(query)

    with st.spinner("🤖 Thinking..."):

        start = time.process_time()

        response = st.session_state.rag_chain.invoke({
            "question": query
        })

        response_time = time.process_time() - start

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )

    st.chat_message("assistant").write(response)

    st.caption(f"⏱ Response time: {response_time:.2f} seconds")

    # ================================
    # RETRIEVED CHUNKS
    # ================================

    with st.expander("📄 Retrieved Code Chunks"):

        docs = st.session_state.retriever.invoke(query)

        for doc in docs:
            st.write(f"📂 File: {doc.metadata.get('source','unknown')}")
            st.code(doc.page_content)