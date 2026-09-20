import os

import requests
from dotenv import load_dotenv


load_dotenv()

API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://localhost:8000"
).rstrip("/")


def query_backend(question: str, image=None):
    """
    Send a question and optional image to the FastAPI backend.
    """

    url = f"{API_BASE_URL}/query"

    data = {
        "question": question
    }

    files = None

    if image is not None:
        files = {
            "image": (
                image.name,
                image.getvalue(),
                image.type
            )
        }

    response = requests.post(
        url,
        data=data,
        files=files,
        timeout=120
    )

    response.raise_for_status()

    return response.json()


def check_health():
    """
    Check whether the backend API is running.
    """

    url = f"{API_BASE_URL}/health"

    response = requests.get(
        url,
        timeout=10
    )

    response.raise_for_status()

    return response.json()