import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, Request, UploadFile

from backend.app.schemas.query import QueryResponse


router = APIRouter(prefix="/query", tags=["Query"])


@router.post("", response_model=QueryResponse)
def query(
    request: Request,
    question: str = Form(...),
    image: UploadFile = File(None),
):
    # 1. Retrieve relevant safety documents
    retrieved_chunks = request.app.state.retriever.retrieve(question)

    # 2. Run YOLO if an image was uploaded
    detections = []
    safety_context = None

    if image is not None:
        suffix = Path(image.filename or ".jpg").suffix or ".jpg"

        temp_path = None

        try:
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix
            ) as temp_file:
                temp_path = temp_file.name

                image_bytes = image.file.read()
                temp_file.write(image_bytes)

            # YOLO inference
            detections = request.app.state.yolo.detect(
                Path(temp_path)
            )

            # Convert detections into text context for the LLM
            safety_context = request.app.state.yolo.build_safety_context(
                detections
            )

        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

    # 3. Generate grounded answer using RAG + YOLO context
    result = request.app.state.generator.generate(
        question=question,
        retrieved_chunks=retrieved_chunks,
        safety_context=safety_context,
    )

    # 4. Return answer + sources + detections
    return QueryResponse(
        answer=result["answer"],
        sources=result["sources"],
        detections=detections,
    )