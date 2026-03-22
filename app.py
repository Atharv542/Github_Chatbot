import streamlit as st
import time
import whisper
import tempfile

from streamlit_mic_recorder import mic_recorder

from rag_pipeline import (
    clone_repo,
    load_files,
    create_vector_store,
    create_rag_chain
)


# model = whisper.load_model("base")

st.set_page_config(page_title="GitHub RAG Assistant")


# st.markdown("""
# <style>

# .mic-container{
#     position: fixed;
#     bottom: 18px;
#     right: 20px;
#     z-index: 100;
# }

# .mic-container button{
#     background-color:#2b2b2b;
#     border:none;
#     border-radius:50%;
#     width:40px;
#     height:40px;
#     font-size:18px;
#     cursor:pointer;
# }

# .mic-container button:hover{
#     background-color:#444;
# }

# </style>
# """, unsafe_allow_html=True)


st.sidebar.title("Repository Settings")

repo_url = st.sidebar.text_input("Enter GitHub Repository URL")

process_btn = st.sidebar.button("Process Repository")

st.title("GitHub Repository Chat Assistant")


if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello 👋 Ask me anything about the GitHub repository."
        }
    ]

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None


if process_btn:

    if not repo_url:
        st.sidebar.warning("Please enter a GitHub repository URL")

    else:

        # 🔥 RESET CHAT WHEN NEW REPO IS LOADED
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello 👋 Ask me anything about the new GitHub repository."
            }
        ]

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


#Chat History

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])


# st.markdown('<div class="mic-container">', unsafe_allow_html=True)

# audio = mic_recorder(
#     start_prompt="🎤",
#     stop_prompt="⏹",
#     key="recorder",
#     just_once=True
# )

# st.markdown('</div>', unsafe_allow_html=True)


# voice_query = None

# if audio:

#     with st.spinner("Transcribing audio..."):

#         with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
#             f.write(audio["bytes"])
#             audio_path = f.name

#         result = model.transcribe(audio_path)

#         voice_query = result["text"]

#     st.success(f"Recognized: {voice_query}")


query = st.chat_input("Ask anything about the repository...")
# if voice_query:
#     query = voice_query

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
            "question": query,
            "chat_history": st.session_state.messages
        })

        response_time = time.process_time() - start

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )

    st.chat_message("assistant").write(response)

    st.caption(f"Response time: {response_time:.2f} seconds")

    with st.expander("Retrieved Code Chunks"):

        docs = st.session_state.retriever.invoke(query)

        for doc in docs:
            st.write(f"📄 File: {doc.metadata.get('source','unknown')}")
            st.code(doc.page_content)