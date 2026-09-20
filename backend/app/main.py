from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.core.config import settings
from backend.app.services.retrieval import RetrievalService
from backend.app.services.generation import GenerationService
from backend.app.services.yolo import YOLOService
from backend.app.api.routes.query import router as query_router



@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 60)
    print("Starting ConstructionSafe AI API")
    print("=" * 60)

    # Load RAG retrieval service once
    app.state.retriever = RetrievalService(
        chroma_path=settings.chroma_path,
        collection_name=settings.chroma_collection,
        embedding_model=settings.embedding_model,
        top_k=settings.top_k,
    )

    # Load LLM generation service once
    app.state.generator = GenerationService(
        model_name=settings.ollama_model,
        ollama_host=settings.ollama_host,
    )

    # Load YOLO model once
    app.state.yolo = YOLOService(
        model_path=settings.yolo_model_path,
        confidence_threshold=settings.confidence_threshold,
    )

    print("=" * 60)
    print("ConstructionSafe AI API is ready!")
    print("=" * 60)

    yield

    print("Shutting down ConstructionSafe AI API...")


app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=(
        "Multimodal Construction Safety Assistant "
        "using RAG, YOLO, and Ollama."
    ),
    lifespan=lifespan,
)

app.include_router(query_router)


# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in settings.cors_origins.split(",")
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "message": "ConstructionSafe AI API is running",
        "version": settings.api_version,
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "ConstructionSafe AI API",
    }