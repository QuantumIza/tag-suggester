# src/api/utils/embedding_utils.py

from sentence_transformers import SentenceTransformer
import numpy as np
import tensorflow_hub as hub
import spacy
import gensim.downloader as api

# --- SBERT ---
def encode_with_sbert(text: str) -> np.ndarray:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(text)

# --- USE ---
def encode_with_use(text: str) -> np.ndarray:
    print(">>> Chargement du modèle USE...")
    model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    return model([text])[0].numpy()

# --- Word2Vec ---
def encode_with_word2vec(text: str) -> np.ndarray:
    model = api.load("word2vec-google-news-300")
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    vectors = [model[token.text] for token in doc if token.text in model]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)
