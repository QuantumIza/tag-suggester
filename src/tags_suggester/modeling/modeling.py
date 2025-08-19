from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, hamming_loss  # si tu ne l’as pas déjà
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.base import clone
import os
import joblib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import mlflow
import json
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.exceptions import UndefinedMetricWarning



def split_on_indices(X, Y, train_idx, test_idx):
    return X[train_idx], X[test_idx], Y[train_idx], Y[test_idx]


def run_logreg(name, X_train, X_test, y_train, y_test):
    print(f"🚀 Test LogReg sur vecteur : {name}")
    model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    f1 = f1_score(y_test, y_pred, average="micro")
    hamming = hamming_loss(y_test, y_pred)
    
    print(f"📊 F1-micro = {f1:.3f} | Hamming = {hamming:.3f}")
    return f1, hamming

def run_logreg_vector(name, X_train, X_test, y_train, y_test, preprocess=None):
    print(f"🚀 Test LogReg sur vecteur : {name}")
    
    # Pré-traitement conditionnel
    if preprocess == "scale":
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    # Entraînement
    model = OneVsRestClassifier(LogisticRegression(max_iter=1000))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Scores
    f1 = f1_score(y_test, y_pred, average="micro")
    hamming = hamming_loss(y_test, y_pred)
    
    print(f"📊 F1-micro = {f1:.3f} | Hamming = {hamming:.3f}")
    return f1, hamming


def run_cv_evaluation(X, Y, model_callable, n_splits=5, scoring="f1_micro", verbose=True):
    """
    Évalue un modèle multilabel via K-Fold CV

    Parameters:
        X: matrice des vecteurs (numpy array ou sparse matrix)
        Y: matrice des labels multilabel
        model_callable: fonction lambda qui retourne une instance du modèle
                        ex: lambda: ClassifierChain(LogisticRegression(max_iter=1000))
        n_splits: nombre de folds pour K-Fold
        scoring: 'f1_micro', 'f1_weighted', etc.
        verbose: affichage des scores intermédiaires

    Returns:
        List des scores par fold, moyenne, std
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = Y[train_idx], Y[test_idx]

        model = model_callable()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        score = f1_score(y_test, y_pred, average=scoring)
        scores.append(score)

        if verbose:
            print(f"📊 Fold {fold+1} — {scoring} : {score:.4f}")

    print("\n🎯 Moyenne F1 :", np.mean(scores).round(4))
    print("📉 Écart-type :", np.std(scores).round(4))

    return scores





def run_cv_logreg(name, X, Y, n_splits=5, preprocess=None):
    print(f"🔁 Validation croisée LogReg sur {name}")
    
    steps = []
    if preprocess == "scale":
        steps.append(("scaler", StandardScaler()))
    
    steps.append(("clf", OneVsRestClassifier(LogisticRegression(max_iter=1000))))
    
    pipe = Pipeline(steps)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    scores = cross_val_score(pipe, X, Y, cv=kf, scoring="f1_micro")
    print(f"📊 F1-micro moyen : {scores.mean():.3f}")
    return scores.mean()

def save_best_model_cv(df_metrics, splits_dict, y_train, model_type="logreg", save_dir="models"):
    """
    Sauvegarde le meilleur modèle LogReg entraîné selon le vecteur avec le meilleur score F1_cv.
    
    Paramètres :
    - df_metrics : DataFrame contenant les colonnes ['vecteur', 'F1_cv']
    - splits_dict : dict des tuples (X_train, X_test) pour chaque vecteur
    - y_train : matrice multilabel d'entraînement
    - model_type : nom du classifieur (utilisé pour le nom du fichier)
    - save_dir : dossier de sauvegarde des modèles
    """

    # 🔎 Sélection du vecteur optimal
    best_vect_name = df_metrics.sort_values("F1_cv", ascending=False).iloc[0]["vecteur"]
    X_train_best, _ = splits_dict[best_vect_name]

    # 🧪 Entraînement
    model_final = OneVsRestClassifier(LogisticRegression(max_iter=1000))
    model_final.fit(X_train_best, y_train)

    # 📦 Sauvegarde
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{model_type}_final_{best_vect_name}.joblib")
    joblib.dump(model_final, path)

    print(f"✅ Modèle final LogReg sauvegardé : {path}")
    return model_final, best_vect_name, path


def run_model_vector(name, X_train, X_test, y_train, y_test, model_class, model_wrapper=None, preprocess=None):
    """
    Entraîne un modèle sur un vecteur donné avec pipeline (scaling si nécessaire).
    
    Paramètres :
    - name : nom du vecteur
    - X_train, X_test, y_train, y_test : données
    - model_class : classe du modèle à instancier (LogisticRegression, XGBClassifier...)
    - model_wrapper : wrapper comme OneVsRestClassifier ou ClassifierChain (facultatif)
    - preprocess : "scale" ou None
    """
        
    # 🧱 Construction du pipeline
    steps = []
    
    if preprocess == "scale":
        steps.append(("scaler", StandardScaler()))
    
    clf = model_class()
    
    if model_wrapper:
        clf = model_wrapper(clf)
    
    steps.append(("clf", clf))
    pipe = Pipeline(steps)
    
    # 🚀 Entraînement
    pipe.fit(X_train, y_train)
    
    # 🔍 Prédiction
    y_pred = pipe.predict(X_test)
    
    # 📊 Métriques
    f1 = f1_score(y_test, y_pred, average="micro")
    hamming = hamming_loss(y_test, y_pred)
    
    print(f"""# --- F1-micro : {f1:.3f}
              # --- Hamming loss : {hamming:.4f}""")
    return f1, hamming

def train_and_score_vector(name, X_train, X_test, y_train, y_test, model_class, model_wrapper=None, preprocess=None):
    steps = []

    if preprocess == "scale":
        steps.append(("scaler", StandardScaler()))

    clf = model_class()
    if model_wrapper:
        clf = model_wrapper(clf)

    steps.append(("clf", clf))
    pipe = Pipeline(steps)

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    # --- COUVERTURE TAGS
    

    f1 = f1_score(y_test, y_pred, average="micro")
    hamming = hamming_loss(y_test, y_pred)
    coverage = coverage_score(y_test, y_pred)

    return f1, hamming, coverage, pipe  # ⬅️ On ajoute le modèle entraîné en sortie

def train_and_score_vector_full_metrics(name, X_train, X_test, y_train, y_test, model_class, model_wrapper=None, preprocess=None):
    from sklearn.metrics import precision_score, recall_score, f1_score, hamming_loss

    steps = []

    if preprocess == "scale":
        steps.append(("scaler", StandardScaler()))

    clf = model_class()
    if model_wrapper:
        clf = model_wrapper(clf)

    steps.append(("clf", clf))
    pipe = Pipeline(steps)

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    # 🔍 Métriques classiques multilabel
    f1 = f1_score(y_test, y_pred, average="micro")
    hamming = hamming_loss(y_test, y_pred)
    coverage = coverage_score(y_test, y_pred)

    # 🧪 Nouveaux scorings utiles
    precision = precision_score(y_test, y_pred, average="micro")
    recall = recall_score(y_test, y_pred, average="micro")
    f1_macro = f1_score(y_test, y_pred, average="macro")

    return {
        "f1_micro": f1,
        "f1_macro": f1_macro,
        "precision_micro": precision,
        "recall_micro": recall,
        "hamming_loss": hamming,
        "coverage_tags": coverage,
        "model": pipe
    }

def run_cv_model(name, X, Y, model_class, model_wrapper=None, preprocess=None, n_splits=5):
    """
    Validation croisée sur un vecteur donné avec pipeline.
    
    Paramètres :
    - model_class : classe du modèle à instancier
    - model_wrapper : wrapper multilabel (facultatif)
    - preprocess : "scale" ou None
    """
    steps = []
    if preprocess == "scale":
        steps.append(("scaler", StandardScaler()))
    
    clf = model_class()
    if model_wrapper:
        clf = model_wrapper(clf)
    
    steps.append(("clf", clf))
    pipe = Pipeline(steps)
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, Y, cv=kf, scoring="f1_micro")
    
    print(f"📊 Moyenne F1-micro : {scores.mean():.3f}")
    return scores.mean()




def cross_validate_vector(name, X, Y, model_class, model_wrapper=None, preprocess=None, n_splits=5):
    steps = []
    if preprocess == "scale":
        steps.append(("scaler", StandardScaler()))
    
    model = model_class()
    if model_wrapper:
        model = model_wrapper(model)
    
    steps.append(("clf", model))
    pipe = Pipeline(steps)

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, Y, cv=cv, scoring="f1_micro")

    return scores.mean()

def compile_metrics(df_split, df_cv):
    metrics = []

    for name in df_split["vecteur"]:
        f1_split = df_split.query("vecteur == @name")["f1_micro"].values[0]
        hamming = df_split.query("vecteur == @name")["hamming_loss"].values[0]
        f1_cv = df_cv.query("vecteur == @name")["f1_cv"].values[0]
        delta = round(f1_split - f1_cv, 3)

        metrics.append({
            "vecteur": name,
            "F1_split": f1_split,
            "F1_cv": f1_cv,
            "delta_split_cv": delta,
            "hamming_loss": hamming
        })

    return pd.DataFrame(metrics).sort_values("F1_cv", ascending=False)

def select_best_vecteur(df_metrics, min_f1=0.4, min_coverage=0.3, ranking_col="f1_micro"):
    """
    Sélectionne le meilleur vecteur selon des critères multiples.
    Retourne le nom du vecteur.
    """
    filtered = df_metrics[
        (df_metrics[ranking_col] >= min_f1) &
        (df_metrics["coverage_tags"] >= min_coverage)
    ]

    if not filtered.empty:
        best_vect = filtered.sort_values(ranking_col, ascending=False).iloc[0]["vecteur"]
    else:
        best_vect = df_metrics.sort_values(ranking_col, ascending=False).iloc[0]["vecteur"]

    print(f"🎯 Vecteur sélectionné : {best_vect}")
    return best_vect

def save_best_model(df_metrics, splits_dict, y_train,
                    model_class, model_wrapper=None,
                    model_type="model", save_dir="models"):

    os.makedirs(save_dir, exist_ok=True)

    # 🔍 Sélection du vecteur
    best_vect = select_best_vecteur(df_metrics)
    X_train_best, _ = splits_dict[best_vect]

    # ⚙️ Instanciation du modèle
    model = model_class()
    if model_wrapper:
        model = model_wrapper(clone(model))

    # 🚀 Entraînement
    model.fit(X_train_best, y_train)

    # 💾 Sauvegarde du modèle
    model_filename = f"{model_type}_final_{best_vect}.joblib"
    model_path = os.path.join(save_dir, model_filename)
    joblib.dump(model, model_path)

    # 📝 Sauvegarde de la config pour API
    config_path = os.path.join(save_dir, "config_best_model.json")
    with open(config_path, "w") as f:
        json.dump({
            "vectorizer": best_vect,
            "model_path": model_path
        }, f)

    print(f"✅ Modèle sauvegardé : {model_path}")
    print(f"📝 Config enregistrée : {config_path}")

    return model, best_vect, model_path

def save_best_model_old(df_metrics, splits_dict, y_train, model_class, model_wrapper=None, model_type="model", save_dir="models"):
    import os
    os.makedirs(save_dir, exist_ok=True)

    best_vect = df_metrics.iloc[0]["vecteur"]
    X_train_best, _ = splits_dict[best_vect]

    model = model_class()
    if model_wrapper:
        model = model_wrapper(model)

    model.fit(X_train_best, y_train)
    path = os.path.join(save_dir, f"{model_type}_final_{best_vect}.joblib")
    joblib.dump(model, path)

    print(f"✅ Modèle sauvegardé : {path}")
    return model, best_vect, path




def coverage_score(y_true, y_pred):
    """
    Calcule le taux de couverture des tags dans les prédictions multilabel.
    """
    true_positives = (y_true & y_pred).sum()
    total_tags = y_true.sum()
    
    if total_tags == 0:
        return 0.0
    
    coverage = true_positives / total_tags
    return round(coverage, 4)


def cross_validate_vector(name, X, Y, model_class, model_wrapper=None, preprocess=None, scoring="f1_micro", cv=5):
    """
    Effectue une validation croisée sur un vecteur avec pipeline, retourne le score moyen.
    """
    steps = []

    if preprocess == "scale":
        steps.append(("scaler", StandardScaler()))

    clf = model_class()
    if model_wrapper:
        clf = model_wrapper(clf)

    steps.append(("clf", clf))
    pipe = Pipeline(steps)

    scores = cross_val_score(pipe, X, Y, scoring=scoring, cv=cv)
    return round(scores.mean(), 4)

def plot_and_log_barplot(df_scores, metric="f1_cv", title=None, save_path="barplot.png"):

    plt.figure(figsize=(10, 5))
    sns.barplot(data=df_scores, x="vecteur", y=metric, palette="Blues_d")
    plt.ylabel(metric)
    plt.title(title or f"Comparaison des vecteurs selon {metric}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Sauvegarde image
    plt.savefig(save_path)
    plt.close()
    
    # Logging MLflow
    mlflow.log_artifact(save_path)

import json
import joblib
import os

def save_best_trained_model_old_2(df_metrics, trained_models_dict, save_dir="models/logreg"):
    os.makedirs(save_dir, exist_ok=True)

    # 🔍 Sélection du vecteur optimal
    best_vect = select_best_vecteur(df_metrics)

    # 📦 Récupération du modèle déjà entraîné
    model = trained_models_dict[best_vect]

    # 💾 Sauvegarde du modèle
    model_filename = f"model_final_{best_vect}.joblib"
    model_path = os.path.join(save_dir, model_filename)
    joblib.dump(model, model_path)

    # 📄 Génération du fichier de config
    config_path = os.path.join(save_dir, "config_best_model.json")
    config = {
        "vectorizer": best_vect,
        "model_path": model_path
        # (tu peux aussi ajouter "vectorizer_path" ici si besoin pour ton API)
    }
    with open(config_path, "w") as f:
        json.dump(config, f)

    print(f"✅ Modèle final sauvegardé : {model_path}")
    print(f"📝 Config créée : {config_path}")

    return model, best_vect, model_path
def save_best_trained_model_old_2(df_metrics, trained_models_dict, save_dir="models/logreg"):
    os.makedirs(save_dir, exist_ok=True)

    # 🔍 Sélection du vecteur optimal
    best_vect = select_best_vecteur(df_metrics)
    model = trained_models_dict[best_vect]

    # 💾 Sauvegarde du modèle
    model_filename = f"model_final_{best_vect}.joblib"
    model_path = os.path.join(save_dir, model_filename)
    joblib.dump(model, model_path)

    # 🔍 Chemins supplémentaires selon le vecteur
    if best_vect == "tfidf":
        vectorizer_path = "models/tfidf/tfidf_vectorizer.joblib"
    elif best_vect == "bow":
        vectorizer_path = "models/bow/vectorizer_bow_full.pkl"
    elif best_vect == "sbert":
        vectorizer_path = "models/sbert/sbert_model"
    elif best_vect == "use":
        # Tu avais stocké le chemin USE dans un JSON dédié
        with open("models/use_model/use_path.json") as f:
            use_config = json.load(f)
        vectorizer_path = use_config["path"]
    elif best_vect == "word2vec":
        vectorizer_path = "models/word2vec/word2vec_titlebody_full.bin"
    elif best_vect == "svd":
        vectorizer_path = "models/tfidf/tfidf_vectorizer.joblib"
        svd_path = "models/svd/svd_model_10k.pkl"
    else:
        raise ValueError(f"Vectorizer '{best_vect}' non reconnu.")

    # 📄 MultiLabelBinarizer — chemin constant
    mlb_path = "models/tags/multilabel_binarizer_full.pkl"

    # 📄 Génération du fichier config
    config = {
        "vectorizer": best_vect,
        "model_path": model_path,
        "vectorizer_path": vectorizer_path,
        "mlb_path": mlb_path
    }

    # Cas spécial pour SVD : on ajoute le chemin
    if best_vect == "svd":
        config["svd_path"] = svd_path

    config_path = os.path.join(save_dir, "config_best_model.json")
    with open(config_path, "w") as f:
        json.dump(config, f)

    print(f"✅ Modèle final sauvegardé : {model_path}")
    print(f"📝 Config enrichi avec chemins : {config_path}")

    return model, best_vect, model_path

from pathlib import Path
import os, shutil, json, joblib

from pathlib import Path
import json, joblib, pickle

def save_best_trained_model(model, vectorizer, mlb, model_name, vect_type, svd_model=None):
    import joblib, json, os
    from pathlib import Path

    model_dir = Path(__file__).resolve().parent.parent / "models" / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "model.joblib"
    joblib.dump(model, model_path)

    mlb_path = model_dir / "mlb.joblib"
    joblib.dump(mlb, mlb_path)

    vectorizer_path = model_dir / "vectorizer"
    if vect_type in ["tfidf", "svd"]:
        vectorizer_path = vectorizer_path.with_suffix(".joblib")
        joblib.dump(vectorizer, vectorizer_path)
    elif vect_type == "bow":
        vectorizer_path = vectorizer_path.with_suffix(".pkl")
        with open(vectorizer_path, "wb") as f:
            pickle.dump(vectorizer, f)
    elif vect_type in ["sbert", "use", "word2vec"]:
        vectorizer_path = str(vectorizer_path)  # path as str for dynamic loading
        # Le modèle lui-même est conservé ailleurs (pas picklé localement)

    svd_path = None
    if svd_model:
        svd_path = model_dir / "svd.joblib"
        joblib.dump(svd_model, svd_path)

    # ✅ Création du fichier de config
    config = {
        "model_path": str(model_path),
        "vectorizer": vect_type,
        "vectorizer_path": str(vectorizer_path),
        "mlb_path": str(mlb_path)
    }

    if svd_path:
        config["svd_path"] = str(svd_path)

    config_path = model_dir / "config_best_model.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)




import joblib
import json
import os
# --- A METTRE A JOUR - INCOMPLETE
def load_pipeline_components(config_path="./models/config_best_model.json"):
    # Lire le fichier de config
    with open(config_path, "r") as f:
        config = json.load(f)

    # Extraire les chemins des composants
    vectorizer_path = config["vectorizer_path"]
    model_path = config["model_path"]
    mlb_path = config["mlb_path"]

    # Charger les objets
    vectorizer = joblib.load(vectorizer_path)
    model = joblib.load(model_path)
    mlb = joblib.load(mlb_path)

    return vectorizer, model, mlb


def get_true_labels(df, mlb):
    return mlb.transform(df["Tags"])



def evaluate_predictions(y_true, y_pred):
    return {
        "f1_score": f1_score(y_true, y_pred, average="macro"),
        "precision": precision_score(y_true, y_pred, average="macro"),
        "recall": recall_score(y_true, y_pred, average="macro")
    }

from sklearn.metrics import f1_score, precision_score, recall_score

def evaluate_month(month, df, embeddings, model, mlb):
    """
    Évalue les performances du modèle de classification multilabel pour un mois donné.

    Paramètres :
    ----------
    month : str
        Nom ou identifiant du mois (ex: "2023-06").
    df : pd.DataFrame
        DataFrame contenant les textes et les tags du mois.
    embeddings : np.ndarray
        Matrice des embeddings SBERT pré-calculés pour les textes du mois.
    model : sklearn classifier
        Modèle de classification multilabel entraîné (ex: LogisticRegression).
    mlb : MultiLabelBinarizer
        Binariseur utilisé pour transformer les tags en vecteurs binaires.

    Retour :
    -------
    dict
        Dictionnaire contenant les scores F1, précision, recall, le mois et le nombre de textes.
    """
   
    # print(f"\n# --- EVALUATION DU MOIS : {month}")
    # Transformation des tags en vecteurs binaires
    y_true = mlb.transform(df["Tags"])
    # Prédiction des tags à partir des embeddings
    y_pred = model.predict(embeddings)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    # Calcul des métriques de performance
    scores = {
        "mois": month,
        "nb_textes": len(df),
        "f1_score": f1_score(y_true, y_pred, average="macro"),
        "precision": precision_score(y_true, y_pred, average="macro"),
        "recall": recall_score(y_true, y_pred, average="macro")
    }
    
    # print(f"# --- SCORES CALCULES POUR LE MOIS  {month} : {scores}")
    return scores
def get_transformator_path(best_vect, model_dir="models/"):
    if best_vect == "sbert":
        vectorizer_path = model_dir / "sbert" / "sbert_model"
    elif best_vect == "use":
        vectorizer_path = model_dir / "use_model" / "use_path.json"
    elif best_vect in ["word2vec", "w2v"]:
        vectorizer_path = model_dir / "word2vec" / "word2vec_titlebody_full.bin"
    elif best_vect == "bow":
        vectorizer_path = model_dir / "bow" / "vectorizer_bow_full.pkl"
    elif best_vect == "tfidf":
        vectorizer_path = model_dir / "tfidf" / "tfidf_vectorizer.joblib"
    elif best_vect == "svd":
        vectorizer_path = model_dir / "svd" / "tfidf_vectorizer.joblib"
        svd_path = model_dir / "sbert" / "svd_model_10k.joblib"
    else:
        raise ValueError(f"🚫 Type de vecteur inconnu : {best_vect}")
    return vectorizer_path




def train_and_score_vector_full_metrics_custom(
    name,
    X_train,
    X_test,
    y_train,
    y_test,
    model_class,
    model_wrapper=None,
    preprocess=None,
    X_text_train=None,
    X_text_test=None
):
    from sklearn.metrics import precision_score, recall_score, f1_score, hamming_loss
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    import numpy as np

    print(f"\n🚀 Entraînement du modèle : {name}")

    steps = []

    if preprocess == "scale":
        steps.append(("scaler", StandardScaler()))

    clf = model_class()
    if model_wrapper:
        clf = model_wrapper(clf)

    steps.append(("clf", clf))
    pipe = Pipeline(steps)

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    # 🔍 Métriques classiques multilabel
    f1 = f1_score(y_test, y_pred, average="micro")
    hamming = hamming_loss(y_test, y_pred)
    coverage = (y_pred & y_test).sum() / y_test.sum()

    # 🧪 Nouveaux scorings utiles
    precision = precision_score(y_test, y_pred, average="micro")
    recall = recall_score(y_test, y_pred, average="micro")
    f1_macro = f1_score(y_test, y_pred, average="macro")

    print(f"✅ Scores pour {name} :")
    print(f"  - f1_micro: {f1:.4f}")
    print(f"  - f1_macro: {f1_macro:.4f}")
    print(f"  - precision_micro: {precision:.4f}")
    print(f"  - recall_micro: {recall:.4f}")
    print(f"  - hamming_loss: {hamming:.4f}")
    print(f"  - coverage_tags: {coverage:.4f}")

    # 📋 Log d'exemples mal prédits
    if X_text_test is not None:
        mismatches = np.where((y_pred != y_test).any(axis=1))[0]
        print(f"\n🔍 {len(mismatches)} erreurs détectées sur {len(y_test)} exemples.")

        for i in mismatches[:3]:
            print(f"\n❌ Exemple mal prédit #{i}")
            print(f"Texte : {X_text_test.iloc[i][:300]}...")
            print(f"Tags réels : {np.where(y_test[i])[0]}")
            print(f"Tags prédits : {np.where(y_pred[i])[0]}")

    return {
        "f1_micro": f1,
        "f1_macro": f1_macro,
        "precision_micro": precision,
        "recall_micro": recall,
        "hamming_loss": hamming,
        "coverage_tags": coverage,
        "model": pipe
    }



def split_on_indices_custom(X, Y, train_idx, test_idx):
    from scipy.sparse import issparse

    if issparse(X):
        X_train = X[train_idx]
        X_test = X[test_idx]
    else:
        X_train = X[train_idx, :]
        X_test = X[test_idx, :]

    y_train = Y[train_idx]
    y_test = Y[test_idx]

    return X_train, X_test, y_train, y_test
