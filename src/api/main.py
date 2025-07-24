# --- POINT D'ENTREE DE L'API AVEC FASTAPI
from fastapi import FastAPI
from src.api.routes.predict_routes import router as predict_router
app = FastAPI(title="Tag Suggester API")
# INCLUSION DES ROUTES
app.include_router(predict_router, prefix="/predict", tags=["Tagging"])
