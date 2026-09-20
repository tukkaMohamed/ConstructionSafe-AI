from pathlib import Path
import re

from sentence_transformers import SentenceTransformer
import chromadb
import ollama


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

CHROMA_PATH = BASE_DIR / "chroma_db"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3.2"

TOP_K = 2


# ============================================================
# EVALUATION DATASET
# ============================================================
# Expected answers are based ONLY on CONSTRUCTION_PPE.pdf

EVALUATION_DATASET = [

    {
        "question": "What protection should workers use for their head?",
        "expected": (
            "Workers should wear hard hats where there is a potential "
            "for objects falling from above, bumps to the head from "
            "fixed objects, or accidental head contact with electrical hazards."
        ),
        "expected_chunk": 1
    },

    {
        "question": "When should workers wear hard hats?",
        "expected": (
            "Workers should wear hard hats where there is a potential "
            "for objects falling from above, bumps to the head from "
            "fixed objects, or accidental head contact with electrical hazards."
        ),
        "expected_chunk": 1
    },

    {
        "question": "What hazards require head protection?",
        "expected": (
            "Head protection is required for potential falling objects, "
            "bumps to the head from fixed objects, or accidental head "
            "contact with electrical hazards."
        ),
        "expected_chunk": 1
    },

    {
        "question": "What should workers do if a hard hat has dents or cracks?",
        "expected": (
            "Hard hats should be routinely inspected for dents, cracks, "
            "or deterioration and replaced after a heavy blow or electrical shock."
        ),
        "expected_chunk": 1
    },

    {
        "question": "When should a hard hat be replaced?",
        "expected": (
            "A hard hat should be replaced after a heavy blow or electrical shock."
        ),
        "expected_chunk": 1
    },

    {
        "question": "How should hard hats be maintained?",
        "expected": (
            "Hard hats should be routinely inspected for dents, cracks, "
            "or deterioration and replaced after a heavy blow or electrical shock."
        ),
        "expected_chunk": 1
    },

    {
        "question": "What type of protection is mentioned for the eyes and face?",
        "expected": (
            "Safety glasses or face shields are mentioned for eye and face protection."
        ),
        "expected_chunk": 0
    },

    {
        "question": "What type of protection is mentioned for the feet?",
        "expected": (
            "Work shoes or boots with slip-resistant and puncture-resistant soles "
            "are mentioned for foot protection."
        ),
        "expected_chunk": 0
    },

    {
        "question": "What type of protection is mentioned for the hands?",
        "expected": (
            "Gloves are mentioned for hand protection. Workers should wear the "
            "right gloves for the job, such as heavy-duty rubber gloves for "
            "concrete work, welding gloves for welding, and insulated gloves "
            "and sleeves when exposed to electrical hazards."
        ),
        "expected_chunk": 0
    },

    {
        "question": "What protection is mentioned for hearing?",
        "expected": (
            "Earplugs or earmuffs are mentioned for high-noise work areas, "
            "and earplugs should be cleaned or replaced regularly."
        ),
        "expected_chunk": 1
    },
]


# ============================================================
# LOAD RAG COMPONENTS
# ============================================================

def load_rag():

    print("\nLoading embedding model...")

    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    print("Loading ChromaDB...")

    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH)
    )

    collection = client.get_collection(
        name="construction_safety"
    )

    print(
        f"Chunks in database: {collection.count()}"
    )

    return embedding_model, collection


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve(
    question,
    embedding_model,
    collection
):

    query_embedding = embedding_model.encode(
        question
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    retrieved = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):

        retrieved.append({
            "text": document,
            "source": metadata.get("source"),
            "page": metadata.get("page"),
            "chunk_id": metadata.get("chunk_id"),
            "distance": distance
        })

    return retrieved


# ============================================================
# GENERATION
# ============================================================

def generate_answer(
    question,
    retrieved_chunks
):

    context = ""

    for i, chunk in enumerate(
        retrieved_chunks,
        start=1
    ):

        context += f"""
--- Retrieved Chunk {i} ---

Source: {chunk["source"]}
Page: {chunk["page"]}
Chunk: {chunk["chunk_id"]}

{chunk["text"]}
"""

    prompt = f"""
You are a construction safety assistant evaluating a
Retrieval-Augmented Generation (RAG) system.

Your task is to answer the user's question using ONLY
the retrieved document context provided below.

IMPORTANT RULES:

1. The retrieved context is the authoritative source.
2. Carefully read ALL retrieved chunks before answering.
3. If the answer is explicitly present in the retrieved
   context, you MUST provide that information.
4. Do not say that the information is unavailable if
   the retrieved context actually contains the answer.
5. Do not add information from outside the retrieved context.
6. Preserve the meaning of the document accurately.
7. If the answer contains multiple relevant details,
   include the important details instead of giving only
   a general statement.
8. Do not invent safety requirements.
9. ALWAYS provide the source and page at the end.

Your response MUST use this format:

Answer:
<clear answer based only on the retrieved context>

Source: <source filename>
Page: <page number>

USER QUESTION:
{question}

RETRIEVED DOCUMENT CONTEXT:
{context}
"""

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
# RETRIEVAL CORRECTNESS
# ============================================================

def check_retrieval(
    retrieved_chunks,
    expected_chunk
):

    retrieved_chunk_ids = [
        chunk["chunk_id"]
        for chunk in retrieved_chunks
    ]

    return expected_chunk in retrieved_chunk_ids


# ============================================================
# CITATION CHECK
# ============================================================

def check_citation(
    answer,
    retrieved_chunks
):

    source_found = False
    page_found = False

    for chunk in retrieved_chunks:

        source = str(
            chunk["source"]
        )

        page = str(
            chunk["page"]
        )

        if source.lower() in answer.lower():
            source_found = True

        if re.search(
            rf"\bPage\s*[:\-]?\s*{re.escape(page)}\b",
            answer,
            re.IGNORECASE
        ):
            page_found = True

    return source_found and page_found


# ============================================================
# CONTENT CORRECTNESS
# ============================================================

def check_content_correctness(
    answer,
    expected
):

    answer_lower = answer.lower()

    # Remove citation section
    answer_without_citation = re.split(
        r"source\s*:",
        answer_lower,
        flags=re.IGNORECASE
    )[0]

    expected_words = re.findall(
        r"\b[a-zA-Z]{4,}\b",
        expected.lower()
    )

    if not expected_words:
        return False

    # Ignore generic words
    stop_words = {
        "workers",
        "should",
        "mentioned",
        "according",
        "provided",
        "document",
        "protection",
        "information",
        "question",
        "required",
        "there",
        "where",
        "with",
        "from",
        "their",
        "this",
        "that"
    }

    important_words = [
        word
        for word in expected_words
        if word not in stop_words
    ]

    if not important_words:
        return False

    matches = sum(
        1
        for word in important_words
        if word in answer_without_citation
    )

    coverage = (
        matches / len(important_words)
    )

    return coverage >= 0.50


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("RAG EVALUATION - EXTENDED")
    print("=" * 70)

    print(
        f"\nEvaluation questions: "
        f"{len(EVALUATION_DATASET)}"
    )

    embedding_model, collection = load_rag()

    results = []

    # --------------------------------------------------------
    # Evaluate each question
    # --------------------------------------------------------

    for index, item in enumerate(
        EVALUATION_DATASET,
        start=1
    ):

        question = item["question"]
        expected = item["expected"]
        expected_chunk = item["expected_chunk"]

        print("\n" + "=" * 70)
        print(
            f"QUESTION {index}/{len(EVALUATION_DATASET)}"
        )
        print("=" * 70)

        print(
            f"\nQuestion:\n{question}"
        )

        # Retrieval
        retrieved = retrieve(
            question,
            embedding_model,
            collection
        )

        print("\nRetrieved:")

        for i, chunk in enumerate(
            retrieved,
            start=1
        ):

            print(
                f"  {i}. "
                f"Chunk {chunk['chunk_id']} | "
                f"Distance {chunk['distance']:.4f}"
            )

        retrieval_ok = check_retrieval(
            retrieved,
            expected_chunk
        )

        # Generation
        print("\nGenerating answer...")

        answer = generate_answer(
            question,
            retrieved
        )

        print("\nGenerated Answer:")
        print(answer)

        # Evaluation
        citation_ok = check_citation(
            answer,
            retrieved
        )

        content_ok = check_content_correctness(
            answer,
            expected
        )

        print(
            f"\nRetrieval Correct: "
            f"{'YES' if retrieval_ok else 'NO'}"
        )

        print(
            f"Content Correct: "
            f"{'YES' if content_ok else 'NO'}"
        )

        print(
            f"Citation Correct: "
            f"{'YES' if citation_ok else 'NO'}"
        )

        results.append({
            "question": question,
            "retrieval": retrieval_ok,
            "content": content_ok,
            "citation": citation_ok
        })

    # ========================================================
    # FINAL METRICS
    # ========================================================

    total = len(results)

    retrieval_correct = sum(
        result["retrieval"]
        for result in results
    )

    content_correct = sum(
        result["content"]
        for result in results
    )

    citation_correct = sum(
        result["citation"]
        for result in results
    )

    retrieval_rate = (
        retrieval_correct / total * 100
    )

    content_rate = (
        content_correct / total * 100
    )

    citation_rate = (
        citation_correct / total * 100
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n\n")
    print("=" * 70)
    print("FINAL RAG EVALUATION RESULTS")
    print("=" * 70)

    print(
        f"\nTotal Questions: {total}"
    )

    print(
        f"\nRetrieval Correctness: "
        f"{retrieval_correct}/{total} "
        f"({retrieval_rate:.1f}%)"
    )

    print(
        f"Content Correctness: "
        f"{content_correct}/{total} "
        f"({content_rate:.1f}%)"
    )

    print(
        f"Citation Correctness: "
        f"{citation_correct}/{total} "
        f"({citation_rate:.1f}%)"
    )

    print("\n" + "-" * 70)

    print(
        f"{'#':<4}"
        f"{'Retrieval':<14}"
        f"{'Content':<14}"
        f"{'Citation'}"
    )

    print("-" * 70)

    for i, result in enumerate(
        results,
        start=1
    ):

        retrieval = (
            "YES"
            if result["retrieval"]
            else "NO"
        )

        content = (
            "YES"
            if result["content"]
            else "NO"
        )

        citation = (
            "YES"
            if result["citation"]
            else "NO"
        )

        print(
            f"{i:<4}"
            f"{retrieval:<14}"
            f"{content:<14}"
            f"{citation}"
        )

    print("\n" + "=" * 70)
    print("EXTENDED RAG EVALUATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()