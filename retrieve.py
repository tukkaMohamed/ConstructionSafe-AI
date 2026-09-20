from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


def main():

    BASE_DIR = Path(__file__).resolve().parent

    CHROMA_DIR = BASE_DIR / "chroma_db"

    print("=" * 60)
    print("RAG RETRIEVAL TEST")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Load embedding model
    # --------------------------------------------------

    print("\nLoading embedding model...")

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    # --------------------------------------------------
    # 2. Load ChromaDB
    # --------------------------------------------------

    print("Loading ChromaDB...")

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    collection = client.get_collection(
        name="construction_safety"
    )

    print(f"Chunks in database: {collection.count()}")

    # --------------------------------------------------
    # 3. User question
    # --------------------------------------------------

    question = "What protection should workers use for their head?"

    print("\nQuestion:")
    print(question)

    # --------------------------------------------------
    # 4. Convert question to embedding
    # --------------------------------------------------

    question_embedding = model.encode(
        question
    ).tolist()

    # --------------------------------------------------
    # 5. Retrieve relevant chunks
    # --------------------------------------------------

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=2
    )

    # --------------------------------------------------
    # 6. Display results
    # --------------------------------------------------

    print("\n" + "=" * 60)
    print("RETRIEVED DOCUMENTS")
    print("=" * 60)

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for i, (document, metadata, distance) in enumerate(
        zip(documents, metadatas, distances),
        start=1
    ):

        print(f"\n--- Result {i} ---")

        print(f"Source: {metadata['source']}")
        print(f"Page: {metadata['page']}")
        print(f"Chunk ID: {metadata['chunk_id']}")
        print(f"Distance: {distance:.4f}")

        print("\nRetrieved Text:")
        print(document)

    print("\n" + "=" * 60)
    print("RETRIEVAL TEST COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()