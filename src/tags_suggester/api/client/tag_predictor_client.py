import requests

class TagPredictorClient:
    def __init__(self, base_url="http://localhost:8000/predict"):
        self.base_url = base_url

    def predict_tags(self, title, body, vectorizer="bow"):
        payload = {
            "title": title,
            "body": body,
            "vectorizer": vectorizer
        }
        response = requests.post(self.base_url, json=payload)
        return response.json()
