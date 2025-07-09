from fastapi import FastAPI
from src.api.routes.embed_routes import router as embed_router

app = FastAPI(title="Tag Suggester API")

# Inclusion des routes
app.include_router(embed_router, prefix="/embed", tags=["Embeddings"])
