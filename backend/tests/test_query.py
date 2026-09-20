from fastapi.testclient import TestClient

from backend.app.main import app


class FakeRetrievalService:
    def __init__(self, *args, **kwargs):
        pass

    def retrieve(self, question):
        return [
            {
                "text": "Workers should wear hard hats where there is a potential for falling objects.",
                "metadata": {
                    "source": "CONSTRUCTION_PPE.pdf",
                    "page": 1,
                },
                "distance": 0.2,
            }
        ]


class FakeGenerationService:
    def __init__(self, *args, **kwargs):
        pass

    def generate(self, question, retrieved_chunks, safety_context=None):
        return {
            "answer": "Workers should wear hard hats when there is a potential for falling objects.",
            "sources": ["CONSTRUCTION_PPE.pdf - Page 1"],
        }


class FakeYOLOService:
    def __init__(self, *args, **kwargs):
        pass

    def detect(self, image_path):
        return [
            {
                "class_name": "Hardhat",
                "confidence": 0.85,
                "bbox": [10.0, 20.0, 100.0, 150.0],
            }
        ]

    def build_safety_context(self, detections):
        return (
            "Computer vision observations from the uploaded image:\n"
            "- Hardhat: 1 detection(s)"
        )


def fake_lifespan(app):
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(app):
        app.state.retriever = FakeRetrievalService()
        app.state.generator = FakeGenerationService()
        app.state.yolo = FakeYOLOService()
        yield

    return lifespan


# Replace the real startup services with lightweight fake services
app.router.lifespan_context = fake_lifespan(app)



def test_query_happy_path():
    with TestClient(app) as client:
        response = client.post(
            "/query",
            data={
                "question": "When should workers wear hard hats?"
            },
        )

        assert response.status_code == 200

        data = response.json()

        assert "answer" in data
        assert "sources" in data
        assert "detections" in data

        assert len(data["sources"]) > 0


def test_query_missing_question():
    with TestClient(app) as client:
        response = client.post("/query")

        assert response.status_code == 422