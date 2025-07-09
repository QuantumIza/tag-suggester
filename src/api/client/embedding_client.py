import requests

def get_embedding(text: str, model: str = "sbert", host: str = "http://127.0.0.1:8000") -> list:
    """
    Envoie un texte à l'API FastAPI pour obtenir son embedding.
    :param text: Texte à encoder
    :param model: Nom du modèle ('sbert', 'use', 'word2vec')
    :param host: URL de l'API
    :return: Liste de float (embedding)
    """
    url = f"{host}/embed/{model}"
    response = requests.post(url, json={"text": text})
    if response.status_code == 200:
        return response.json()["embedding"]
    else:
        raise ValueError(f"Erreur API {model} : {response.status_code} - {response.text}")
