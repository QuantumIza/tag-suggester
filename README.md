# Projet : Catégorisez automatiquement des questions

Ce projet a pour objectif de créer un système de suggestion automatique de tags sur des questions Stack Overflow, à l'aide d'approches supervisées et non supervisées intégrées dans une démarche MLOps.
# Tag Suggester — Stack Overflow Tag Prediction

Ce projet a pour objectif de développer un système de suggestion automatique de tags pour les questions posées sur Stack Overflow, à l’aide de techniques de traitement du langage naturel (NLP) et de machine learning.

## 🚀 Objectifs

- Extraire et nettoyer des données Stack Overflow
- Proposer des tags automatiquement à partir du contenu d’une question
- Comparer deux approches :
  - Non supervisée (LDA)
  - Supervisée (classification multi-label)
- Intégrer une démarche MLOps avec MLflow
- Déployer une API de prédiction sur le cloud

## 🧱 Structure du projet
├── data/ # Données brutes et prétraitées ├── notebooks/ # Notebooks d'exploration et de modélisation ├── src/ # Code source modulaire │ ├── preprocessing/ # Nettoyage et préparation des données │ ├── vectorization/ # Vectorisation des textes │ ├── modeling/ # Entraînement et évaluation des modèles │ ├── api/ # Code de l'API FastAPI │ └── mlops/ # Suivi MLflow, pipelines, etc. ├── tests/ # Tests unitaires ├── requirements.txt # Dépendances du projet ├── .gitignore # Fichiers/dossiers à ignorer par Git └── README.md # Présentation du projet

## 🧪 Technologies utilisées

- Python 3.12
- Pandas, Scikit-learn, NLTK, Gensim
- MLflow
- FastAPI
- StackAPI (ou Stack Exchange API)
- Git & GitHub

## 📦 Installation

```bash
git clone https://github.com/ton_nom_utilisateur/tag-suggester.git
cd tag-suggester
python -m venv venv
source venv/bin/activate  # ou .\venv\Scripts\activate sous Windows
pip install -r requirements.txt
📈 Suivi des expérimentations
L’interface MLflow est accessible localement via :
mlflow ui
Puis dans le navigateur : http://127.0.0.1:5000
🌐 Déploiement
Une API FastAPI permet de tester le modèle en local ou sur le cloud (Heroku, Azure, AWS). Une interface Streamlit ou un notebook permet de simuler une question et d’obtenir les tags suggérés.

📄 Licence
Projet réalisé dans le cadre de la formation OpenClassrooms — Data Scientist.
