from pathlib import Path

from .embeddings import EmbeddingModel
from .vector_store import get_collection


class Retriever:
    """Find the most relevant indexed passages for a question."""
    def __init__(self, persist_dir: str | Path, embedding_model: EmbeddingModel | None = None):
        self.collection = get_collection(persist_dir)
        self.embedding_model = embedding_model or EmbeddingModel()

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        if not query.strip():
            return []
        count = self.collection.count()
        if count == 0:
            return []
        query_embedding = self.embedding_model.encode([query])[0]
        result = self.collection.query(query_embeddings=[query_embedding.tolist()], n_results=min(top_k, count))
        found = []
        for i, text in enumerate(result["documents"][0]):
            metadata = result["metadatas"][0][i]
            found.append({"text": text, "page_number": metadata["page_number"],
                          "document_name": metadata["document_name"], "chunk_id": result["ids"][0][i],
                          "distance": result["distances"][0][i]})
        return found
