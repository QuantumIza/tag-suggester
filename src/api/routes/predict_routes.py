from fastapi import APIRouter, Request
from pydantic import BaseModel
from src.api.services.predict import predict_tags

router = APIRouter()

class Question(BaseModel):
    title: str
    body: str
    vectorizer: str = "bow"  # or tfidf, etc.

@router.post("/predict")
def predict_endpoint(q: Question):
    tags = predict_tags(q.title, q.body, q.vectorizer)
    return {"suggested_tags": tags}
