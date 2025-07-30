from src.tags_suggester.api.services.model_loader import load_model

# def predict_tags(title, body, vectorizer_name):
#     # Simplification ici, on suppose un vectorizer intégré au modèle
#     model = load_model("logreg", vectorizer_name)
#     text = f"{title} {body}"
#     # prediction = model.predict([text])
#     prediction = model.predict([[text]])  # ← au lieu de [text]

#     return prediction[0] if hasattr(prediction, '__iter__') else [prediction]


import joblib
import os

def predict_tags(title, body, vectorizer_name):
    model = load_model("logreg", vectorizer_name)
    
    # Charger le vectorizer correspondant
    vectorizer_path = os.path.join("notebooks", "models", "vectorizers", f"{vectorizer_name}.joblib")
    vectorizer = joblib.load(vectorizer_path)

    # Fusion du titre + corps
    text = f"{title} {body}"

    # Transformation en vecteurs numériques
    X = vectorizer.transform([text])
    
    # Prédiction
    prediction = model.predict(X)

    # Conversion des prédictions binaires en tags
    if hasattr(model, "classes_"):
        tags = [tag for tag, present in zip(model.classes_, prediction[0]) if present]
    else:
        tags = list(prediction)

    return tags
