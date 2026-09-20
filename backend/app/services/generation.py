import ollama


class GenerationService:
    def __init__(
        self,
        model_name: str,
        ollama_host: str = "http://127.0.0.1:11434",
    ):
        self.model_name = model_name
        self.client = ollama.Client(host=ollama_host)
        print(f"Generation service ready. Model: {self.model_name}")

    def generate(
        self,
        question: str,
        retrieved_chunks: list[dict],
        safety_context: str | None = None,
    ) -> dict:

        # Build retrieved document context
        context_parts = []

        for i, chunk in enumerate(retrieved_chunks, start=1):
            metadata = chunk.get("metadata", {})

            source = metadata.get("source", "Unknown source")
            page = metadata.get("page", "Unknown page")

            context_parts.append(
                f"""
--- Retrieved Document {i} ---
Source: {source}
Page: {page}

{chunk["text"]}
"""
            )

        retrieved_context = "\n".join(context_parts)

        if not retrieved_context:
            retrieved_context = "No relevant document context was retrieved."

        # Build computer vision context
        visual_context = safety_context or (
            "No image was provided. "
            "Do not make any visual observations."
        )

        prompt = f"""
You are ConstructionSafe AI, a construction safety assistant.

Answer the user's question using the retrieved construction
safety documents as the primary source of truth.

USER QUESTION:
{question}

RETRIEVED SAFETY DOCUMENTS:
{retrieved_context}

COMPUTER VISION OBSERVATIONS:
{visual_context}

IMPORTANT RULES:

1. Answer using the retrieved documents whenever they contain
   information relevant to the user's question.

2. Do not contradict information that is explicitly stated
   in the retrieved documents.

3. Do not say that the documents lack information if the
   retrieved text explicitly contains the answer.

4. Do not invent safety requirements that are not present
   in the retrieved documents.

5. Computer vision detections are visual observations only.
   They are NOT proof of safety compliance or non-compliance.

6. NEVER use a YOLO detection to conclude, imply, or label
   a worker as compliant, non-compliant, safe, unsafe,
   violating a rule, or meeting a safety requirement.

7. If detections such as "NO-Hardhat", "NO-Mask", or
   "NO-Safety Vest" are present, report them ONLY as detections.

   For example:
   "The image contains 2 NO-Hardhat detections."

   Do NOT say:
   "This indicates non-compliance."
   "The worker is unsafe."
   "The worker violates the requirement."

8. When the question involves both the document and the image,
   clearly separate:
   - Document requirement
   - Image observations

   The safety document defines the requirement.
   YOLO only describes what was visually detected.

9. If the retrieved documents truly do not contain enough
   information to answer the question, say:
   "The provided safety documents do not contain enough
   information to answer this question."

10. Answer only what is relevant to the user's question.
Do not add information about unrelated safety categories,
even if that information appears in the retrieved documents.

11. If computer vision observations are provided, you MUST
explicitly mention the relevant detected objects in the
"Image observations" section.

12. Never say "No image observations were provided" or
"None mentioned in the provided image" when the
computer vision observations contain detections.

13. Do not invent visual observations. Mention only objects
that appear in the provided computer vision observations.

14. When reporting YOLO detections, use the detected class names
exactly as provided by the computer vision system.

For example:
- "The image contains 2 NO-Hardhat detections."
- "The image contains 1 Safety Vest detection."

Do NOT translate, reinterpret, or infer a person's condition
from a detection.

For example, never convert:
"NO-Hardhat detection"
into:
"2 people without hard hats."

YOLO class labels must be reported as detections only.

15. Never infer the number of people affected by a PPE-related
detection unless the computer vision system explicitly provides
that relationship.


Answer format:

Answer:
<clear answer>

Image observations:
<mention relevant YOLO observations if an image was provided>

Source:
<document name>

Page:
<page number>
"""

        response = self.client.chat(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        answer = response["message"]["content"].strip()

        # Build source list
        sources = []

        for chunk in retrieved_chunks:
            metadata = chunk.get("metadata", {})

            source = metadata.get("source")
            page = metadata.get("page")

            if source:
                source_text = (
                    f"{source} - Page {page}"
                    if page is not None
                    else source
                )

                if source_text not in sources:
                    sources.append(source_text)

        return {
            "answer": answer,
            "sources": sources,
        }