from pathlib import Path
import hashlib

import chromadb

from .chunker import create_chunks
from .embeddings import EmbeddingModel

COLLECTION_NAME = "research_papers"


def get_collection(persist_dir: str | Path):
    client = chromadb.PersistentClient(path=str(persist_dir))
    return client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def index_pdf(pdf_path: str | Path, persist_dir: str | Path, embedding_model: EmbeddingModel | None = None) -> int:
    path = Path(pdf_path)
    chunks = create_chunks(path)
    if not chunks:
        raise ValueError("No useful text chunks were found in this PDF.")
    model = embedding_model or EmbeddingModel()
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts)
    document_id = hashlib.sha256(path.read_bytes()).hexdigest()[:20]
    collection = get_collection(persist_dir)
    collection.delete(where={"document_id": document_id})
    collection.upsert(
        ids=[f"{document_id}_{chunk['chunk_id']}" for chunk in chunks],
        documents=texts,
        embeddings=embeddings.tolist(),
        metadatas=[{"document_id": document_id, "document_name": chunk["document_name"],
                    "page_number": chunk["page_number"]} for chunk in chunks],
    )
    return len(chunks)
