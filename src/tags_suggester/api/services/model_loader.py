# -----------------------------
# 📦 IMPORTS
# -----------------------------
import os
import json
import joblib
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.sparse import load_npz

# -----------------------------
# 🔧 CONFIG PATH
# -----------------------------
# 🧱 Vise le répertoire racine du projet
# 🧭 Corriger pour viser la racine projet

# 💡 Se positionner à la racine réelle du projet, quelle que soit la profondeur du module
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

CONFIG_PATH = os.path.join(BASE_DIR, "models", "logreg", "config_best_model.json")
print(f"🔍 Chemin recherché : {CONFIG_PATH}")
print(f"✅ Fichier existe : {os.path.exists(CONFIG_PATH)}")

# -----------------------------
# 🚀 SBERT : chargement du modèle pré-entraîné
# -----------------------------
sbert_model = SentenceTransformer("all-MiniLM-L6-v2") # TODO : svg pickle

# -----------------------------
# 🚀 Fonction de chargement différé
# -----------------------------
def load_model_and_vectorizer():
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"❌ Fichier de config introuvable : {CONFIG_PATH}")

    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    best_vect = config["vectorizer"]
    model_path = config["model_path"]

    model = joblib.load(model_path)
    return model, best_vect

# -----------------------------
# 📐 Fonction de vectorisation selon le vecteur
# -----------------------------
def vectorize(text, best_vect):
    if best_vect == "sbert":
        embeddings = sbert_model.encode([text])
        return embeddings

    elif best_vect == "bow":
        vectorizer_path = os.path.join(BASE_DIR, "models", "vectorizers", "bow.joblib")
        vectorizer = joblib.load(vectorizer_path)
        return vectorizer.transform([text])

    elif best_vect == "tfidf":
        vectorizer_path = os.path.join(BASE_DIR, "models", "vectorizers", "tfidf.joblib")
        vectorizer = joblib.load(vectorizer_path)
        return vectorizer.transform([text])

    else:
        raise ValueError(f"❌ Vecteur non reconnu : {best_vect}")

# -----------------------------
# 🎯 Fonction de prédiction des tags
# -----------------------------
def predict_tags_old(title, body):
    text = f"{title} {body}"
    model, best_vect = load_model_and_vectorizer()
    # 🔍 Vérification des classes du modèle
    print("🔍 Exemple classes =", model.classes_[:5])
    print("🔍 Types des classes =", [type(cls) for cls in model.classes_[:5]])
    
    X = vectorize(text, best_vect)
    raw_pred = model.predict(X)
    print("✔️ raw_pred =", raw_pred)
    print("✔️ types =", [type(flag) for flag in raw_pred[0]])
   #  return [tag for tag, flag in zip(model.classes_, raw_pred[0]) if flag]
    return [str(tag) for tag, flag in zip(model.classes_, raw_pred[0]) if bool(flag)]

# -----------------------------
# 🎯 Fonction de prédiction des tags
# -----------------------------
def predict_tags(title, body):
    import os
    import joblib
    from src.tags_suggester.api.services.model_loader import load_model_and_vectorizer
    

    # 📝 Préparation du texte
    text = f"{title} {body}"

    # 🔌 Chargement du modèle et du vectorizer
    model, best_vect = load_model_and_vectorizer()

    # 🔍 Diagnostic rapide
    print("🔍 Exemple classes =", model.classes_[:5])
    print("🔍 Types des classes =", [type(cls) for cls in model.classes_[:5]])

    # 🔢 Encodage du texte
    X = vectorize(text, best_vect)

    # 📊 Prédictions brutes
    raw_pred = model.predict(X)
    print("✔️ raw_pred =", raw_pred)
    print("✔️ types =", [type(flag) for flag in raw_pred[0]])

    # 🗂️ Chargement du MultiLabelBinarizer
    # mlb_path = os.path.join("models", "tags", "multilabel_binarizer_full.pkl")
    # mlb_path = os.path.join(BASE_DIR, "models", "tags", "multilabel_binarizer_full.pkl")
    mlb_path = os.path.join(BASE_DIR, "notebooks", "models", "tags", "multilabel_binarizer_full.pkl")
    mlb = joblib.load(mlb_path)

    # 🧭 Construction du mapping id -> tag
    id_to_tag = {i: tag for i, tag in enumerate(mlb.classes_)}

    # ✅ Application du mapping avec vérification
    predicted_tags = [id_to_tag.get(int(tag), str(tag)) for tag, flag in zip(model.classes_, raw_pred[0]) if bool(flag)]

    return predicted_tags

