import os
import joblib

def load_model(model_type, vectorizer_name):
    path = os.path.join("models", model_type, f"{model_type}_{vectorizer_name}.joblib")
    return joblib.load(path)
