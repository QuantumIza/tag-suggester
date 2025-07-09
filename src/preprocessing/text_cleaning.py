import re
import os
import spacy
from bs4 import BeautifulSoup
from utils.tech_terms import load_tech_terms
import seaborn as sns
import matplotlib.pyplot as plt
from utils.stop_terms import load_stop_terms  # Assure-toi que ce fichier existe
import pandas as pd



# --------------------------------------
# CHARGER LE MODELE SPACY UNE SEULE FOIS
# --------------------------------------
nlp = spacy.load("en_core_web_sm")

def clean_text_spacy_custom(text, tech_terms=None):
    if not isinstance(text, str):
        return ""
    
    # Nettoyage HTML + URLs
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"http\S+", "", text)
    
    doc = nlp(text)
    tokens = [token.text.lower() for token in doc]

    # Générer les n-grams (1, 2 ou 3 mots)
    ngrams = set()
    for n in [1, 2, 3]:
        ngrams.update(
            " ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)
        )

    # Ajouter les termes techniques détectés
    tech_terms = load_tech_terms()
    preserved = {term for term in tech_terms if term in ngrams}

    final_tokens = []
    for token in doc:
        t = token.text.lower()

        # Ne pas dupliquer les termes techniques multi-mots
        if any(t in term for term in preserved):
            continue

        if (
            not token.is_stop
            and not token.is_punct
            and not token.like_num
            and token.is_alpha
        ):
            final_tokens.append(token.lemma_.lower())

    # Ajouter les termes techniques à la fin
    return " ".join(sorted(preserved) + final_tokens)











