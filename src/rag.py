import os
import chromadb
from groq import Groq
from sentence_transformers import SentenceTransformer

from .config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, GROQ_MODEL, TOP_K


SYSTEM_PROMPT = """You are an FAQ assistant.

Answer the user's question using ONLY the provided context.
If the answer is not present in the context, say:
"I don't have that information in the FAQ knowledge base."

Do not invent policies, prices, dates, guarantees, or other facts.
Keep the answer concise and helpful.
"""


class RAGEngine:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is missing. Create a .env file and add your Groq API key."
            )

        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        try:
            self.collection = client.get_collection(COLLECTION_NAME)
        except Exception as exc:
            raise RuntimeError(
                "ChromaDB collection not found. Run: python -m src.ingest"
            ) from exc

        self.groq = Groq(api_key=api_key)

    def retrieve(self, question: str):
        query_embedding = self.embedder.encode(
            [question], normalize_embeddings=True
        )[0].tolist()

        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=TOP_K,
            include=["documents", "metadatas", "distances"],
        )

        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]

        contexts = []
        for doc, meta in zip(docs, metas):
            contexts.append(
                f"[Source: {meta.get('source', 'unknown')}]\n{doc}"
            )

        return contexts

    def ask(self, question: str):
        contexts = self.retrieve(question)

        if not contexts:
            return {
                "answer": "I don't have that information in the FAQ knowledge base.",
                "contexts": [],
            }

        context_text = "\n\n---\n\n".join(contexts)

        user_prompt = f"""Context:
{context_text}

Question:
{question}
"""

        response = self.groq.chat.completions.create(
            model=GROQ_MODEL,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )

        answer = response.choices[0].message.content.strip()

        return {
            "answer": answer,
            "contexts": contexts,
        }
