from pathlib import Path
import chromadb
from sentence_transformers import SentenceTransformer

from .config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    DATA_DIR,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP):
    text = " ".join(text.split())
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = max(0, end - overlap)

    return chunks


def ingest():
    DATA_DIR.mkdir(exist_ok=True)
    CHROMA_DIR.mkdir(exist_ok=True)

    documents = []
    metadatas = []
    ids = []

    for path in sorted(DATA_DIR.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        chunks = chunk_text(text)

        for idx, chunk in enumerate(chunks):
            documents.append(chunk)
            metadatas.append({"source": path.name, "chunk": idx})
            ids.append(f"{path.stem}-{idx}")

    if not documents:
        raise RuntimeError(f"No .txt files found in {DATA_DIR}")

    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(documents, normalize_embeddings=True).tolist()

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Rebuild the collection so repeated ingestion does not create stale duplicates.
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(COLLECTION_NAME)

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print(f"Ingested {len(documents)} chunks into '{COLLECTION_NAME}'.")
    print(f"ChromaDB path: {CHROMA_DIR}")


if __name__ == "__main__":
    ingest()
