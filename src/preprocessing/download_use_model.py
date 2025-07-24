import kagglehub

print("⏳ Téléchargement du modèle USE via KaggleHub...")
path = kagglehub.model_download("google/universal-sentence-encoder/tensorFlow2/universal-sentence-encoder")
print("✅ Modèle téléchargé dans :", path)
