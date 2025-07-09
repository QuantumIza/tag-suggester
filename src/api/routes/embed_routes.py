# src/api/routes/embed_routes.py

from fastapi import APIRouter
from src.api.schemas.input_text import InputText
from src.api.utils.embedding_utils import (
    encode_with_sbert,
    encode_with_use,
    encode_with_word2vec
)

router = APIRouter()

# --- SBERT ---
@router.post("/sbert")
def embed_sbert(input: InputText):
    vector = encode_with_sbert(input.text)
    return {"embedding": vector.tolist()}

# --- USE ---
@router.post("/use")
def embed_use(input: InputText):
    vector = encode_with_use(input.text)
    return {"embedding": vector.tolist()}

# --- Word2Vec ---
@router.post("/word2vec")
def embed_word2vec(input: InputText):
    vector = encode_with_word2vec(input.text)
    return {"embedding": vector.tolist()}
