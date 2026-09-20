from pathlib import Path
from pypdf import PdfReader


def main():
    BASE_DIR = Path(__file__).resolve().parent

    PDF_PATH = BASE_DIR.parent / "CONSTRUCTION_PPE.pdf"

    print("=" * 60)
    print("RAG DOCUMENT INGESTION")
    print("=" * 60)

    print(f"\nLoading PDF:")
    print(PDF_PATH)

    if not PDF_PATH.exists():
        print("\nERROR: PDF file not found!")
        return

    reader = PdfReader(str(PDF_PATH))

    print(f"\nNumber of pages: {len(reader.pages)}")

    documents = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if text and text.strip():
            documents.append({
                "page": page_number,
                "text": text.strip()
            })

    print(f"Pages with extracted text: {len(documents)}")

    print("\n" + "=" * 60)
    print("DOCUMENT PREVIEW")
    print("=" * 60)

    for document in documents[:3]:
        print(f"\n--- Page {document['page']} ---")
        print(document["text"][:1000])

    print("\n" + "=" * 60)
    print("INGESTION CHECK COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()