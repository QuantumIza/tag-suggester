# src/setup/setup_nltk.py

import nltk
import os

# Créer un dossier local pour les ressources NLTK
NLTK_DATA_DIR = os.path.join(os.path.dirname(__file__), "../../nltk_data")
NLTK_DATA_DIR = os.path.abspath(NLTK_DATA_DIR)

os.makedirs(NLTK_DATA_DIR, exist_ok=True)

# Télécharger les ressources nécessaires
nltk.download('punkt', download_dir=NLTK_DATA_DIR)
nltk.download('stopwords', download_dir=NLTK_DATA_DIR)
nltk.download('wordnet', download_dir=NLTK_DATA_DIR)
nltk.download('omw-1.4', download_dir=NLTK_DATA_DIR)

print(f"✅ Ressources NLTK téléchargées dans : {NLTK_DATA_DIR}")
