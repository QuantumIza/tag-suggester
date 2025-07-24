from src.api.services.model_loader import load_model

def predict_tags(title, body, vectorizer_name):
    # Simplification ici, on suppose un vectorizer intégré au modèle
    model = load_model("logreg", vectorizer_name)
    text = f"{title} {body}"
    prediction = model.predict([text])
    return prediction[0] if hasattr(prediction, '__iter__') else [prediction]
