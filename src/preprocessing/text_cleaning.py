import re
import os
import spacy
from bs4 import BeautifulSoup
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import joblib
from collections import Counter
# ------------------------------------------------------------
# ON STOCKE EN UNE FOIS LA FUSION DES STOP WORDS NLTK + SPACY
# ------------------------------------------------------------
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
nltk_stopwords = set(stopwords.words("english"))
nlp = spacy.load("en_core_web_sm")
spacy_stopwords = nlp.Defaults.stop_words
combined_stopwords = set(spacy_stopwords).union(nltk_stopwords)





# --------------------------------------------
# --- FONCTION POUR LISTER TERMES A CONSERVES
# --------------------------------------------
def load_tech_terms():
    current_dir = os.path.dirname(__file__)  # src/utils/
    config_path = os.path.abspath(os.path.join(current_dir, "../config/tech_terms.txt"))

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return set(line.strip().lower() for line in f if line.strip())
    except FileNotFoundError:
        print(f"# --- FICHIER INTROUVABLE : {config_path}")
        return set()

# --------------------------------------------
# --- FONCTION POUR LISTER TERMES A EXCLURE
# --------------------------------------------
def load_stop_terms(path=None):
    """
    Charge les mots à exclure lors du traitement NLP, à partir du fichier 'stop_terms.txt'
        
    # --- Returns
    set[str]
        Ensemble de termes en minuscules, nettoyés et prêts à être utilisés comme filtre lexical.

    # --- Comportement
        - IGNORE LIGNES VIDES OU CONTENANT UNIQUEMENT ESPACES
        - SI FICHIER ABSENT : AFFICHE MESSAGE ERREUR ET RETOURNE ENSEMBLE VIDE
    
    # --- Exemple
    >>> stop_terms = load_stop_terms()
    >>> "function" in stop_terms  # True si le mot "function" est dans le fichier
    """
    current_dir = os.path.dirname(__file__)
    config_path = os.path.abspath(os.path.join(current_dir, "../config/stop_terms.txt"))

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return set(line.strip().lower() for line in f if line.strip())
    except FileNotFoundError:
        print(f"# --- FICHIER INTROUVABLE : {config_path}")
        return set()

# -----------------------------------------
# --- TERMES A EXCLURE CHARGES A L'IMPORT 
# -----------------------------------------
custom_stop_terms = load_stop_terms()  
# ------------------------------------------------
# --- FONCTION PRINCIPALE DE NETTOYAGE DES CORPUS
# ------------------------------------------------
def clean_text_spacy_custom(text, tech_terms=None):
    """
    Nettoie un texte brut StackOverflow en appliquant un pipeline NLP complet :
    - Suppression HTML, liens et caractères spéciaux
    - Tokenisation, lemmatisation, filtrage lexical
    - Exclusion des stopwords classiques + stop_terms personnalisés
    - Détection des n-grams techniques via la base `tech_terms`

    Paramètres
    ----------
    text : str
        Texte brut à nettoyer (titre ou body)
    tech_terms : set[str], optional
        Liste de termes techniques multi-mots à préserver. Si None, chargée automatiquement.

    Returns
    -------
    str
        Texte nettoyé : lemmes filtrés + n-grams techniques en tête.
    """
    if not isinstance(text, str):
        return ""

    # --------------------------
    # --- NETTOYAGE HTML ET URL
    # --------------------------
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"http\S+", "", text)

    # ---------------------
    # --- N-GRAM DETECTION
    # ---------------------
    doc = nlp(text)
    tokens = [token.text.lower() for token in doc]
    ngrams = set(" ".join(tokens[i:i+n]) for n in [1, 2, 3] for i in range(len(tokens)-n+1))

    # ---------------------------------
    # --- TERMES TECHNIQUES A GARDER
    # ---------------------------------
    if tech_terms is None:
        tech_terms = load_tech_terms()
    preserved = {term for term in tech_terms if term in ngrams}

    final_tokens = []
    for token in doc:
        lemma = token.lemma_.lower()

        # ------------------------------------------------------
        # --- EXCLUSION SI MOT STOP CUSTOM OU CLASSIQUE
        # ------------------------------------------------------
        if (
            token.is_alpha
            and not token.is_stop
            and not token.is_punct
            and not token.like_num
            and lemma not in custom_stop_terms
            and all(lemma not in term.split() for term in preserved)  # évite duplication
        ):
            final_tokens.append(lemma)

    # ------------------------------------------------------------
    # --- ASSEMBLAGE : TECH_TERMS MULTI-MOTS, PUIS  LEMMES FILTRES
    # -------------------------------------------------------------
    return " ".join(sorted(preserved) + final_tokens)



def get_nltk_stopwords():
    import nltk
    from nltk.corpus import stopwords
    try:
        return set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords")
        return set(stopwords.words("english"))

def clean_text_spacy_custom_2(text, tech_terms=None, stopwords_set=None):
    """
    Nettoie un texte StackOverflow avec :
    - Suppression HTML, liens
    - Lemmatisation, filtrage lexical
    - Exclusion des stopwords (spaCy + NLTK + stop_terms.txt)
    - Préservation des n-grams techniques

    Paramètres
    ----------
    text : str
        Texte brut à nettoyer
    tech_terms : set[str], optional
        N-grams techniques multi-mots à préserver
    stopwords_set : set[str], optional
        Ensemble des mots à exclure, injecté depuis le contexte

    Returns
    -------
    str
        Texte nettoyé, prêt à l’analyse thématique
    """
    if not isinstance(text, str):
        return ""

    # --- Nettoyage HTML et URLs
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"http\S+", "", text)

    doc = nlp(text)
    tokens = [token.text.lower() for token in doc]
    ngrams = set(" ".join(tokens[i:i+n]) for n in [1, 2, 3] for i in range(len(tokens)-n+1))

    # --- Détection des termes techniques
    if tech_terms is None:
        tech_terms = load_tech_terms()
    preserved = {term for term in tech_terms if term in ngrams}

    # --- Chargement des stopwords externes si non fournis
    if stopwords_set is None:
        custom_terms = load_stop_terms()
        stopwords_set = combined_stopwords.union(custom_terms)

    # --- Filtrage lexical
    final_tokens = []
    for token in doc:
        lemma = token.lemma_.lower()
        if (
            token.is_alpha
            and not token.is_punct
            and not token.like_num
            and lemma not in stopwords_set
            and all(lemma not in term.split() for term in preserved)
        ):
            final_tokens.append(lemma)

    return " ".join(sorted(preserved) + final_tokens)


# ----------------------------------------------------------------------------------------
# --- FONCTION QUI NETTOIE UN FICHIER SOURCE CONTENANT DES MOTS A USAGE COMMUN A EXCLURE
# ----------------------------------------------------------------------------------------
import re

import re

def extract_oxford_terms(
    path="../src/config/oxford3000.txt",
    encoding="cp1252",
    export_cleaned_path=None,
    verbose=True
):
    """
    Extrait les lemmes du fichier Oxford 3000, en supprimant les annotations grammaticales.
    Peut écrire les résultats dans un fichier intermédiaire pour inspection.

    Paramètres
    ----------
    path : str
        Chemin vers le fichier brut Oxford 3000.
    encoding : str
        Encodage utilisé pour lire le fichier.
    export_cleaned_path : str, optional
        Fichier destination pour écrire les termes extraits.
    verbose : bool
        Affiche un résumé du traitement.

    Returns
    -------
    set[str]
        Ensemble des mots extraits et nettoyés.
    """
    try:
        with open(path, "r", encoding=encoding) as f:
            raw_lines = f.readlines()
    except FileNotFoundError:
        print(f"❌ Fichier introuvable : {path}")
        return set()
    except UnicodeDecodeError as e:
        print(f"⚠️ Problème d'encodage : {e}")
        return set()

    cleaned_terms = set()
    for line in raw_lines:
        term = extract_word_from_oxford_line(line)
        if term:
            cleaned_terms.add(term)

    if export_cleaned_path:
        with open(export_cleaned_path, "w", encoding="utf-8") as f:
            for word in sorted(cleaned_terms):
                f.write(word + "\n")
        if verbose:
            print(f"📝 Terme nettoyés écrits dans : {export_cleaned_path}")

    if verbose:
        print(f"📖 {len(cleaned_terms)} mots extraits depuis : {path}")
        print(f"🔍 Exemple : {sorted(list(cleaned_terms))[:10]}")

    return cleaned_terms

def extract_word_from_oxford_line(line):
    """
    Extrait le mot principal d'une ligne Oxford (avant annotations).

    Exemple : "about prep., adv." → "about"

    Returns
    -------
    str or None
        Le mot extrait, ou None si la ligne est invalide.
    """
    line = line.strip()
    match = re.match(r"^([a-zA-Z\-']+)", line)
    return match.group(1).lower() if match else None


def generate_vague_terms(path="../src/config/vague_terms.txt", verbose=True):
    """
    Génère une liste de mots vagues, modaux et discursifs à exclure du corpus StackOverflow.

    Catégories incluses :
    - Modaux et auxiliaires
    - Atténuateurs
    - Verbes d’opinion
    - Formulations génériques

    Le fichier est enregistré avec des commentaires par bloc.
    """
    terms_by_category = {
        "# Modaux et auxiliaires": [
            "can", "could", "should", "would", "might", "must", "may", "do", "does", "did", "have", "has", "had"
        ],
        "# Atténuateurs et adverbes flous": [
            "maybe", "perhaps", "possibly", "usually", "often", "sometimes", "never", "always", "rarely", "somehow"
        ],
        "# Verbes d’opinion ou de supposition": [
            "think", "believe", "feel", "guess", "seem", "hope", "assume", "suppose", "consider", "understand"
        ],
        "# Formulations génériques / peu discriminantes": [
            "solution", "problem", "question", "example", "tutorial", "answer", "reason", "explain", "show", "use",
            "need", "want", "help", "fix", "change", "update", "check", "try"
        ]
    }

    with open(path, "w", encoding="utf-8") as f:
        for category, words in terms_by_category.items():
            f.write(category + "\n")
            for word in sorted(set(words)):
                f.write(word + "\n")
            f.write("\n")

    if verbose:
        total = sum(len(words) for words in terms_by_category.values())
        print(f"✅ {total} termes vagues enregistrés dans : {path}")

    return path


def clean_doc_spacy_custom(doc, tech_terms=None, stopwords_set=None):
    """
    Nettoie un objet spaCy Doc préanalysé :
    - Lemmatisation
    - Filtrage lexical
    - Préservation des n-grams techniques

    Paramètres
    ----------
    doc : spacy.tokens.Doc
        Document spaCy déjà tokenisé
    tech_terms : set[str], optional
        Liste de n-grams techniques multi-mots à préserver
    stopwords_set : set[str], optional
        Ensemble des mots à exclure

    Returns
    -------
    str
        Texte nettoyé avec les n-grams en tête
    """
    if not doc or not hasattr(doc, "__iter__"):
        return ""

    # --- Construction des ngrams
    tokens = [token.text.lower() for token in doc]
    ngrams = set(" ".join(tokens[i:i+n]) for n in [1, 2, 3] for i in range(len(tokens)-n+1))

    # --- Chargement des termes techniques
    if tech_terms is None:
        tech_terms = load_tech_terms()
    preserved = {term for term in tech_terms if term in ngrams}

    # --- Construction du texte nettoyé
    final_tokens = []
    for token in doc:
        lemma = token.lemma_.lower()
        if (
            token.is_alpha
            and not token.is_punct
            and not token.like_num
            and stopwords_set
            and lemma not in stopwords_set
            and all(lemma not in term.split() for term in preserved)
        ):
            final_tokens.append(lemma)

    return " ".join(sorted(preserved) + final_tokens)
