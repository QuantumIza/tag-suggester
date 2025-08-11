from fastapi import APIRouter
from pydantic import BaseModel
from src.tags_suggester.api.services.model_loader import predict_tags

router = APIRouter()

# 📦 Schéma d’entrée minimal et réaliste
class Question(BaseModel):
    title: str
    body: str

# 🎯 Route POST pour prédiction des tags
@router.post("/")
def predict_endpoint(q: Question):
    try:
        tags = predict_tags(q.title, q.body)
        return {"suggested_tags": tags}
    except Exception as e:
        import traceback
        traceback.print_exc()  # Affiche dans la console
        return {"error": str(e)}

