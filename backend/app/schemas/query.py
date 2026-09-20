from pydantic import BaseModel, Field


class QueryResponse(BaseModel):
    answer: str
    sources: list[str]
    detections: list[dict] = []


class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="User's construction safety question"
    )