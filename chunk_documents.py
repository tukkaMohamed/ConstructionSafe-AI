from pathlib import Path
from pypdf import PdfReader


def chunk_text(text, chunk_size=1000, overlap=150):
    """
    Split text into chunks without cutting words.
    """

    words = text.split()

    chunks = []

    current_words = []
    current_length = 0

    for word in words:

        word_length = len(word) + 1

        if current_length + word_length > chunk_size:

            if current_words:
                chunks.append(" ".join(current_words))

            # Keep the last words as overlap
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

    print("=" * 60)
    print("DOCUMENT CHUNKING")
    print("=" * 60)

    print(f"\nLoading PDF:")
    print(PDF_PATH)

    if not PDF_PATH.exists():
        print("\nERROR: PDF file not found!")
        return

    reader = PdfReader(str(PDF_PATH))

    all_chunks = []

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text()

        if not text or not text.strip():
            continue

        text = text.strip()

        chunks = chunk_text(
            text,
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

    print(f"\nTotal chunks created: {len(all_chunks)}")

    print("\n" + "=" * 60)
    print("CHUNK PREVIEW")
    print("=" * 60)

    for i, chunk in enumerate(all_chunks):

        print(f"\n--- Chunk {i} ---")
        print(f"Source: {chunk['metadata']['source']}")
        print(f"Page: {chunk['metadata']['page']}")
        print(f"Chunk ID: {chunk['metadata']['chunk_id']}")

        print("\nText:")
        print(chunk["text"])

    print("\n" + "=" * 60)
    print("CHUNKING COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()