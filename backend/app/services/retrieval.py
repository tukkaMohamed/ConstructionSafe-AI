from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


class RetrievalService:
    def __init__(
        self,
        chroma_path: Path,
        collection_name: str,
        embedding_model: str,
        top_k: int = 2,
    ):
        self.chroma_path = str(chroma_path)
        self.collection_name = collection_name
        self.top_k = top_k

        print("Loading embedding model...")
        self.embedding_model = SentenceTransformer(embedding_model)

        print("Loading Chroma vector store...")
        self.client = chromadb.PersistentClient(
            path=self.chroma_path
        )

        self.collection = self.client.get_collection(
            name=self.collection_name
        )

        print(
            f"Retrieval service ready. "
            f"Documents in collection: {self.collection.count()}"
        )

    def retrieve(self, question: str) -> list[dict]:
        """Retrieve the most relevant document chunks."""

        question_embedding = self.embedding_model.encode(
            question
        ).tolist()

        results = self.collection.query(
            query_embeddings=[question_embedding],
            n_results=self.top_k,
            include=[
                "documents",
                "metadatas",
                "distances",
            ],
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        retrieved_chunks = []

        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances,
        ):
            retrieved_chunks.append(
                {
                    "text": document,
                    "metadata": metadata,
                    "distance": distance,
                }
            )

        return retrieved_chunks