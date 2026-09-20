from pathlib import Path
from collections import Counter

from ultralytics import YOLO
from sentence_transformers import SentenceTransformer
import chromadb
import ollama



# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

YOLO_MODEL_PATH = (
    BASE_DIR
    / "runs"
    / "detect"
    / "runs"
    / "construction_yolo11n"
    / "weights"
    / "best.pt"
)

CHROMA_PATH = BASE_DIR / "chroma_db"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3.2"

TOP_K = 2
CONFIDENCE_THRESHOLD = 0.30



# ============================================================
# 1. YOLO DETECTION
# ============================================================

def run_yolo_detection(image_path):

    print("\n" + "=" * 60)
    print("YOLO SAFETY DETECTION")
    print("=" * 60)

    print("\nLoading YOLO model...")
    model = YOLO(str(YOLO_MODEL_PATH))

    print(f"Image: {image_path}")

    results = model.predict(
        source=str(image_path),
        conf=CONFIDENCE_THRESHOLD,
        verbose=False
    )

    result = results[0]

    detections = []

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            class_name = model.names[class_id]

            detections.append({
                "class": class_name,
                "confidence": confidence
            })

    return detections


# ============================================================
# 2. BUILD SAFETY CONTEXT FROM YOLO
# ============================================================

def build_safety_context(detections):

    print("\n" + "=" * 60)
    print("YOLO DETECTION SUMMARY")
    print("=" * 60)

    if not detections:
        print("\nNo objects detected.")
        return "No objects were detected in the image."

    # Count detected classes
    counts = Counter(d["class"] for d in detections)

    print("\nDetected objects:")

    for class_name, count in counts.items():
        print(f"  - {class_name}: {count}")

    # Safety-related detections
    safety_classes = {
        "Hardhat",
        "NO-Hardhat",
        "Mask",
        "NO-Mask",
        "Safety Vest",
        "NO-Safety Vest",
        "Person"
    }

    safety_detections = [
        d for d in detections
        if d["class"] in safety_classes
    ]

    print("\nSafety-related detections:")

    if not safety_detections:
        print("  None")

    for detection in safety_detections:
        print(
            f"  - {detection['class']}: "
            f"{detection['confidence']:.2f}"
        )

    # Create context for RAG
    context_lines = []

    context_lines.append(
        "Computer vision observations from the uploaded image:"
    )

    for class_name, count in counts.items():
        context_lines.append(
            f"- {class_name}: {count} detection(s)"
        )

    context_lines.append(
        "\nImportant: These are visual detections only. "
        "They should not automatically be interpreted as proof "
        "of compliance or non-compliance with a safety requirement."
    )

    return "\n".join(context_lines)


# ============================================================
# 3. RETRIEVE DOCUMENTS
# ============================================================

def retrieve_documents(query):

    print("\n" + "=" * 60)
    print("RAG RETRIEVAL")
    print("=" * 60)

    print("\nLoading embedding model...")
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    print("Loading ChromaDB...")
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))

    collection = client.get_collection(
        name="construction_safety"
    )

    print(f"Chunks in database: {collection.count()}")

    query_embedding = embedding_model.encode(
        query
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    retrieved_chunks = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):

        retrieved_chunks.append({
            "text": document,
            "source": metadata.get("source"),
            "page": metadata.get("page"),
            "chunk_id": metadata.get("chunk_id"),
            "distance": distance
        })

    print("\nRetrieved chunks:")

    for i, chunk in enumerate(retrieved_chunks, 1):

        print(
            f"\nResult {i}: "
            f"{chunk['source']} | "
            f"Page {chunk['page']} | "
            f"Chunk {chunk['chunk_id']} | "
            f"Distance {chunk['distance']:.4f}"
        )

    return retrieved_chunks


# ============================================================
# 4. GENERATE GROUNDED ANSWER
# ============================================================

def generate_answer(question, safety_context, retrieved_chunks):

    print("\n" + "=" * 60)
    print("GENERATING MULTIMODAL RAG ANSWER")
    print("=" * 60)

    # Build document context
    document_context = ""

    for i, chunk in enumerate(retrieved_chunks, 1):

        document_context += f"""
--- Retrieved Document {i} ---
Source: {chunk['source']}
Page: {chunk['page']}
Chunk: {chunk['chunk_id']}

{chunk['text']}
"""

    prompt = f"""
You are a construction safety assistant using a Retrieval-Augmented
Generation system.

You have TWO sources of information:

1. Computer vision observations from a construction-site image.
2. Retrieved safety information from the provided documents.

IMAGE OBSERVATIONS:
{safety_context}

RETRIEVED DOCUMENTS:
{document_context}

USER QUESTION:
{question}

INSTRUCTIONS:

- Answer using the retrieved documents as the authoritative source
  for safety requirements.
- Use the image observations only as visual evidence/context.
- Do NOT invent safety requirements that are not supported by the
  retrieved documents.
- Do NOT claim that a worker is compliant or non-compliant solely
  from an object detection.
- If the image contains both positive and negative detections,
  report them as observations rather than choosing one as fact.
- Clearly distinguish what YOLO detected from what the safety
  document requires.
- Include the document source and page in the answer.
- If the retrieved documents do not contain enough information,
  explicitly say that the provided documents do not contain enough
  information.

Provide a concise but useful answer.
"""

    print("\nSending prompt to Llama 3.2...")

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


# ============================================================
# 5. MAIN PIPELINE
# ============================================================

def main():

    print("=" * 60)
    print("CONSTRUCTION SAFETY MULTIMODAL RAG")
    print("=" * 60)

    # --------------------------------------------------------
    # Image input
    # --------------------------------------------------------

    image_path = input(
        "\nEnter image path:\n> "
    ).strip()

    image_path = Path(image_path)

    if not image_path.exists():
        print(
            f"\nERROR: Image not found:\n{image_path}"
        )
        return

    # --------------------------------------------------------
    # Step 1: YOLO
    # --------------------------------------------------------

    detections = run_yolo_detection(
        image_path
    )

    safety_context = build_safety_context(
        detections
    )

    print("\nSafety context sent to RAG:")
    print("-" * 60)
    print(safety_context)

    # --------------------------------------------------------
    # Step 2: User question
    # --------------------------------------------------------

    question = input(
        "\nEnter your safety question:\n> "
    ).strip()

    if not question:
        print("\nERROR: Question cannot be empty.")
        return

    # --------------------------------------------------------
    # Step 3: Retrieval
    # --------------------------------------------------------

    # IMPORTANT:
    # Retrieval is driven by the user's question.
    # YOLO observations are provided to the LLM as visual context.

    rag_query = question

    retrieved_chunks = retrieve_documents(
        rag_query
    )

    # --------------------------------------------------------
    # Step 4: LLM generation
    # --------------------------------------------------------

    answer = generate_answer(
        question,
        safety_context,
        retrieved_chunks
    )

    # --------------------------------------------------------
    # Step 5: Final output
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("MULTIMODAL RAG ANSWER")
    print("=" * 60)

    print("\n" + answer)

    print("\n" + "=" * 60)
    print("MULTIMODAL RAG PIPELINE COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()