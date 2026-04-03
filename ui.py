import streamlit as st
from embedding import embed_chunks
from retriever import search
from generator import generate_answer

# Page config
st.set_page_config(page_title="RAG Chatbot", layout="wide")

# Sidebar
st.sidebar.title("📌 RAG Chatbot Info")
st.sidebar.write("Model: Gemini 2.5 Flash")
st.sidebar.write("Vector DB: FAISS")
st.sidebar.write("Embeddings: Gemini Embedding")

# Title
st.title("📄 Chat Assistant")
st.write("Ask questions based on HR policy")

# Initialize chat history
if "history" not in st.session_state:
    st.session_state.history = []

# Chat input
query = st.chat_input("Ask something about HR policies...")

if query:
    # Store user message
    st.session_state.history.append(("user", query))

    with st.spinner("Thinking... 🤔"):

        # Step 1: Embed query
        query_embedding = embed_chunks([query])[0]

        # Step 2: Retrieve chunks
        results = search(query_embedding, k=3)

        # Step 3: Generate answer
        answer = generate_answer(query, results)

    # Store bot response
    st.session_state.history.append(("bot", answer))

    # Store sources (optional)
    st.session_state.last_sources = results

# Display chat history
for role, message in st.session_state.history:
    if role == "user":
        with st.chat_message("user"):
            st.write(message)
    else:
        with st.chat_message("assistant"):
            st.write(message)

# Show sources for last question
# if "last_sources" in st.session_state:
#     with st.expander("📚 Source Chunks (Click to view)"):
#         for i, chunk in enumerate(st.session_state.last_sources):
#             st.write(f"**Chunk {i+1}:**")
#             st.write(chunk)
#             st.divider()