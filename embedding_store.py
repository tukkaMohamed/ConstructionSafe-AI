from pathlib import Path
from pypdf import PdfReader

import chromadb
from sentence_transformers import SentenceTransformer


def chunk_text(text, chunk_size=1000, overlap=150):
    words = text.split()

    chunks = []
    current_words = []
    current_length = 0

    for word in words:

        word_length = len(word) + 1

        if current_length + word_length > chunk_size:

            if current_words:
                chunks.append(" ".join(current_words))

            overlap_words = []
            overlap_length = 0

            for previous_word in reversed(current_words):

                if overlap_length + len(previous_word) + 1 > overlap:
                    break

                overlap_words.insert(0, previous_word)
                overlap_length += len(previous_word) + 1

            current_words = overlap_words
            current_length = overlap_length

        current_words.append(word)
        current_length += word_length

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


def main():

    BASE_DIR = Path(__file__).resolve().parent

    PDF_PATH = BASE_DIR.parent / "CONSTRUCTION_PPE.pdf"

    CHROMA_DIR = BASE_DIR / "chroma_db"

    print("=" * 60)
    print("EMBEDDINGS + CHROMADB")
    print("=" * 60)

    # --------------------------------------------------
    # 1. Load PDF
    # --------------------------------------------------

    print("\nLoading PDF...")

    if not PDF_PATH.exists():
        print("\nERROR: PDF file not found!")
        return

    reader = PdfReader(str(PDF_PATH))

    # --------------------------------------------------
    # 2. Extract and chunk documents
    # --------------------------------------------------

    all_chunks = []

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text()

        if not text or not text.strip():
            continue

        chunks = chunk_text(
            text.strip(),
            chunk_size=1000,
            overlap=150
        )

        for chunk_id, chunk in enumerate(chunks):

            all_chunks.append({
                "text": chunk,
                "metadata": {
                    "source": "CONSTRUCTION_PPE.pdf",
                    "page": page_number,
                    "chunk_id": chunk_id
                }
            })

    print(f"Total chunks: {len(all_chunks)}")

    # --------------------------------------------------
    # 3. Load embedding model
    # --------------------------------------------------

    print("\nLoading embedding model...")

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    print("Embedding model loaded.")

    # --------------------------------------------------
    # 4. Create ChromaDB
    # --------------------------------------------------

    print("\nCreating ChromaDB...")

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    collection = client.get_or_create_collection(
        name="construction_safety"
    )

    # --------------------------------------------------
    # 5. Prepare data
    # --------------------------------------------------

    documents = [
        item["text"]
        for item in all_chunks
    ]

    metadatas = [
        item["metadata"]
        for item in all_chunks
    ]

    ids = [
        f"construction_chunk_{i}"
        for i in range(len(all_chunks))
    ]

    # --------------------------------------------------
    # 6. Generate embeddings
    # --------------------------------------------------

    print("\nGenerating embeddings...")

    embeddings = model.encode(
        documents,
        show_progress_bar=True
    ).tolist()

    print("Embeddings generated.")

    # --------------------------------------------------
    # 7. Store in ChromaDB
    # --------------------------------------------------

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings
    )

    print("\nDocuments stored in ChromaDB.")

    # --------------------------------------------------
    # 8. Verify database
    # --------------------------------------------------

    count = collection.count()

    print(f"\nChunks stored: {count}")

    print("\n" + "=" * 60)
    print("CHROMADB INGESTION COMPLETED")
    print("=" * 60)

    print(f"\nDatabase location:")
    print(CHROMA_DIR)


if __name__ == "__main__":
    main()