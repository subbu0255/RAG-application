import streamlit as st
from dotenv import load_dotenv
from src.rag import RAGEngine

load_dotenv()

st.set_page_config(page_title="RAG FAQ Assistant", page_icon="🤖", layout="centered")

st.title("🤖 RAG FAQ Assistant")
st.caption("ChromaDB + Sentence Transformers + Groq")

@st.cache_resource
def load_engine():
    return RAGEngine()

try:
    engine = load_engine()
except Exception as exc:
    st.error(f"Could not initialize the RAG engine: {exc}")
    st.stop()

st.markdown("Ask a question about the FAQ knowledge base.")

question = st.text_input(
    "Your question",
    placeholder="Example: How long does delivery take?"
)

if st.button("Ask", type="primary", disabled=not question.strip()):
    with st.spinner("Searching the knowledge base and generating an answer..."):
        try:
            result = engine.ask(question.strip())
            st.subheader("Answer")
            st.write(result["answer"])

            with st.expander("Retrieved context"):
                for i, item in enumerate(result["contexts"], 1):
                    st.markdown(f"**Source {i}**")
                    st.write(item)
        except Exception as exc:
            st.error(f"Request failed: {exc}")

with st.expander("How this RAG app works"):
    st.write(
        "1. FAQ documents are split into chunks. "
        "2. Sentence Transformers converts chunks into embeddings. "
        "3. ChromaDB stores and retrieves similar chunks. "
        "4. Groq generates an answer using only the retrieved context."
    )
