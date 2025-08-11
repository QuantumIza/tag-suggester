# -----------------------------
# 📦 IMPORTS
# -----------------------------
import os
import joblib
import json
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.sparse import load_npz
from pathlib import Path

# from src.tags_suggester.api.services.model_loader import load_model_and_vectorizer
# -----------------------------
# 🔧 CONFIG PATH
# -----------------------------
# Repartons depuis le fichier actuel (dans src/tags_suggester/api/services)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
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
def vectorize_old_2(text, best_vect):
    if best_vect == "sbert":
        sbert_model = SentenceTransformer("all-MiniLM-L6-v2")  # pkle?
        embeddings = sbert_model.encode([text])
        return embeddings

    elif best_vect == "bow":
        vectorizer_path = os.path.join(BASE_DIR, "models", "vectorizers", "bow.joblib")
        print(f"# --- CHEMIN DU VECTORIZER : {vectorizer_path}")
        vectorizer = joblib.load(vectorizer_path)
        return vectorizer.transform([text])

    elif best_vect == "tfidf":
        vectorizer_path = os.path.join(BASE_DIR, "models", "vectorizers", "tfidf.joblib")
        print(f"# --- CHEMIN DU VECTORIZER : {vectorizer_path}")
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
def predict_tags_old_2(title, body):
    # --- 1. FUSION CORPUS TITLE ET BODY
    text = f"{title} {body}"
    # --- 2. CHARGEMENT DU MODELE ET DU VECTORIZER
    model, best_vect = load_model_and_vectorizer()
    # --- VERIFICATIONS RAPIDES
    print("# --- Exemple classes =", model.classes_[:5])
    print("# --- Types des classes =", [type(cls) for cls in model.classes_[:5]])
    # --- 3. TRANSFORMATION CORPUS EN VECTEUR PRET A ETRE ENTRAINE PAR LE MODELE
    X = vectorize(text, best_vect)
    # --- 4. PREDICTIONS BRUTES
    raw_pred = model.predict(X)
    print("# --- raw_pred =", raw_pred)
    print("# --- types =", [type(flag) for flag in raw_pred[0]])
    # --- 5. CHARGEMENT DU MULTILABELBINARIZER
    mlb_path = os.path.join(BASE_DIR, "notebooks", "models", "tags", "multilabel_binarizer_full.pkl")
    mlb = joblib.load(mlb_path)
    # --- 6. CONSTRUCTION DU MAPPING id -> tag
    id_to_tag = {i: tag for i, tag in enumerate(mlb.classes_)}
    # --- 7. APPLICATION DU MAPPING
    predicted_tags = [id_to_tag.get(int(tag), str(tag)) for tag, flag in zip(model.classes_, raw_pred[0]) if bool(flag)]
    return predicted_tags




def load_pipeline_components():
    # 📁 Définir le chemin vers le dossier 'models'
    base_dir = Path(__file__).resolve().parent.parent / "models"
    config_path = base_dir / "config_best_model.json"

    # 🔧 Charger la config du pipeline
    with open(config_path, "r") as f:
        config = json.load(f)

    # 🎛️ Récupérer le type de vectorisation et son chemin
    vect_type = config["vectorizer"]
    vectorizer_path = config["vectorizer_path"]

    # 🚀 Charger le vectoriseur selon le type
    if vect_type == "sbert":
        # Ex: 'all-MiniLM-L6-v2' ou chemin vers le modèle SBERT
        model_path = "src/tags_suggester/api/models/sbert/sbert_model"
        print("Contenu du dossier SBERT:", os.listdir(model_path) )
        vectorizer = SentenceTransformer(vectorizer_path)  # --- TODO : huggingface est au même emplacement que USE

    elif vect_type == "use":
        import tensorflow_hub as hub
        # vectorizer_path pointe vers le fichier use_path.json
        with open(vectorizer_path, "r") as f:
            use_config = json.load(f)
        use_local_path = use_config["path"]  # --- TODO : modifier emlacement
        vectorizer = hub.load(use_local_path)

    elif vect_type in ["word2vec", "w2v"]:
        from gensim.models import Word2Vec, KeyedVectors
        vectorizer = KeyedVectors.load(vectorizer_path)

    elif vect_type == "tfidf":
        vectorizer = joblib.load(vectorizer_path)

    elif vect_type == "bow":
        vectorizer = joblib.load(vectorizer_path)

    elif vect_type == "svd":
        vectorizer = joblib.load(vectorizer_path)
        svd_path = config.get("svd_path")
        svd_model = joblib.load(svd_path) if svd_path else None
    else:
        raise ValueError(f"❌ Type de vectorisation inconnu : '{vect_type}'")

    # 🎯 Charger le modèle de classification
    model = joblib.load(config["model_path"])

    # 🏷️ Charger le MultiLabelBinarizer
    mlb = joblib.load(config["mlb_path"])

    # 🔁 Retourner tous les composants requis
    if vect_type == "svd":
        return model, vectorizer, mlb, vect_type, svd_model
    else:
        return model, vectorizer, mlb, vect_type, None



import numpy as np

def vectorize(text, vect_type, vectorizer, svd=None):
    if vect_type in ["tfidf", "bow"]:
        # X_tfidf = vectorizer.fit_transform([text])
        return vectorizer.transform([text])

    elif vect_type == "svd":
        # récupérer le vectorizer de tfidf
        # X_tfidf = vectorizer.fit_transform(text_series)
        # svd = TruncatedSVD(n_components=n_components, random_state=random_state)
        # X_reduced = svd.fit_transform(X_tfidf)
        vec = vectorizer.transform([text])  # --- 
        return svd.transform(vec) if svd else vec

    elif vect_type == "sbert":
        # remarque : le répertoire models--sentence-transformers--all-MiniLM-L6-v2 est situé dans cet emplacement local de mon pc : C:\Users\hp\.cache\huggingface\hub 
        # ou faut-il encore faire sbert_model = SentenceTransformer("all-MiniLM-L6-v2") ? 
        return vectorizer.encode([text])

    elif vect_type == "word2vec":
        words = text.split()
        vectors = [vectorizer[word] for word in words if word in vectorizer]
        if vectors:
            return np.mean(vectors, axis=0).reshape(1, -1)
        else:
            return np.zeros((1, vectorizer.vector_size))

    elif vect_type == "use":
        return vectorizer([text])

    else:
        raise ValueError(f"❌ Type de vectorisation '{vect_type}' inconnu.")




def predict_tags(title, body):
    text = f"{title} {body}"
    model, vectorizer, mlb, vect_type, svd_model = load_pipeline_components()
    X = vectorize(text, vect_type, vectorizer)
    raw_pred = model.predict(X)
    id_to_tag = {i: tag for i, tag in enumerate(mlb.classes_)}
    predicted_tags = [id_to_tag[i] for i, flag in enumerate(raw_pred[0]) if flag]
    return predicted_tags




