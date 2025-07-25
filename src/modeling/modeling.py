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

def save_best_model(df_metrics, splits_dict, y_train, model_type="logreg", save_dir="models"):
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
