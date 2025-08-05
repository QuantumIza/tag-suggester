
from tags_suggester.preprocessing.text_cleaning import load_stop_terms
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from rapidfuzz import fuzz
import numpy as np
import networkx as nx
from sklearn.manifold import TSNE
import umap.umap_ as umap
from gensim.models import Word2Vec
import logging
import os
from sentence_transformers import SentenceTransformer

def compute_word_frequencies(df, column, top_n=100, exclude_words=None):
    """
    Calcule la fréquence des mots dans une colonne de texte nettoyé.

    Paramètres :
    - df : DataFrame contenant les textes
    - column : nom de la colonne texte nettoyé (chaîne)
    - top_n : nombre de mots les plus fréquents à retourner
    - exclude_words : liste de mots à exclure (si None, chargée depuis stop_terms.txt)

    Retourne :
    - Un DataFrame avec les colonnes ["word", "frequency"]
    """
    if exclude_words is None:
        exclude_words = load_stop_terms()

    all_words = []

    for text in df[column].dropna():
        tokens = text.split()
        all_words.extend(tokens)

    word_freq = Counter(all_words)

    # Exclure les mots spécifiés
    for word in exclude_words:
        word_freq.pop(word, None)

    most_common = word_freq.most_common(top_n)
    return pd.DataFrame(most_common, columns=["word", "frequency"])

def plot_word_frequencies(
    df_freq,
    max_words_display=30,
    palette="mako",
    figsize=(10, 6),
    title="Top mots les plus fréquents"
):
    """
    Affiche un barplot horizontal des mots les plus fréquents à partir d’un DataFrame.

    Paramètres :
    - df_freq : DataFrame avec colonnes ["word", "frequency"]
    - max_words_display : nombre de mots à afficher dans le graphique
    - palette : palette Seaborn
    - figsize : taille de la figure
    - title : titre du graphique
    """
    if df_freq.empty:
        print("⚠️ Le DataFrame est vide. Aucun graphique généré.")
        return

    display_df = df_freq.head(max_words_display)
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=figsize)
    sns.barplot(
        data=display_df,
        x="frequency",
        y="word",
        hue="word",
        palette=palette,
        legend=False
    )
    plt.title(title, fontsize=14)
    plt.xlabel("Fréquence")
    plt.ylabel("Mot")
    plt.tight_layout()
    plt.show()

def generate_wordcloud(df_freq, max_words=100, background_color="white", colormap="viridis", title="Nuage de mots"):
    """
    Génère et affiche un nuage de mots à partir d’un DataFrame de fréquences.

    Paramètres :
    - df_freq : DataFrame avec colonnes ["word", "frequency"]
    - max_words : nombre maximum de mots à afficher
    - background_color : couleur de fond du nuage
    - colormap : palette de couleurs (ex: "mako", "plasma", "cool", "inferno")
    - title : titre du graphique
    """
    if df_freq.empty:
        print("⚠️ Le DataFrame est vide. Aucun nuage généré.")
        return

    # Convertir en dictionnaire {mot: fréquence}
    word_freq_dict = dict(zip(df_freq["word"], df_freq["frequency"]))

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color=background_color,
        max_words=max_words,
        colormap=colormap,
        collocations=False
    ).generate_from_frequencies(word_freq_dict)

    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_distribution(ax, data, title, color, show_median=True):
    """
    Trace un histogramme avec courbe KDE sur un axe donné, et annote avec skewness, kurtosis, interprétation et médiane.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        L’axe matplotlib sur lequel tracer l’histogramme.
    data : pandas.Series or array-like
        Les données numériques à tracer.
    title : str
        Le titre du graphique.
    color : str
        La couleur principale de l’histogramme.
    show_median : bool, optional
        Si True, affiche une ligne verticale sur la médiane (default: True).

    Returns
    -------
    None
    """
    sns.histplot(data, bins=30, kde=True, ax=ax, color=color)
    ax.set_title(title)

    # Médiane
    if show_median:
        median_val = data.median()
        ax.axvline(median_val, color="black", linestyle="--", linewidth=1.2)
        ax.text(median_val, ax.get_ylim()[1]*0.9, f"Médiane = {median_val:.2f}",
                color="black", ha="right", va="top", fontsize=9, rotation=90)

    # Skewness & kurtosis
    sk = skew(data.dropna())
    kt = kurtosis(data.dropna(), fisher=False)
    skew_desc = "symétrique" if abs(sk) < 0.5 else "asymétrie à droite" if sk > 0 else "asymétrie à gauche"
    kurt_desc = "normale" if abs(kt - 3) < 0.5 else "pointue" if kt > 3 else "plate"
    textstr = f"Skewness : {sk:.2f} ({skew_desc})\nKurtosis : {kt:.2f} ({kurt_desc})"
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.95, 0.95, textstr, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right', bbox=props)


def plot_dual_distribution(data1, data2, label1, label2, title, xlabel,
                           color1="skyblue", color2="salmon", bins=30, show_stats=True):
    """
    Trace deux distributions superposées avec courbe KDE, légende, et annotations statistiques.

    Parameters
    ----------
    data1, data2 : array-like
        Séries de données numériques à comparer.
    label1, label2 : str
        Légendes associées aux deux séries.
    title : str
        Titre du graphique.
    xlabel : str
        Label de l’axe des abscisses.
    color1, color2 : str
        Couleurs des deux distributions.
    bins : int
        Nombre de classes de l’histogramme.
    show_stats : bool
        Si True, affiche skewness, kurtosis, interprétation et médianes.

    Returns
    -------
    None
    """
    plt.figure(figsize=(10, 5))
    sns.histplot(data1.dropna(), bins=bins, color=color1, label=label1, kde=True)
    sns.histplot(data2.dropna(), bins=bins, color=color2, label=label2, kde=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Nombre de questions")
    plt.legend()

    if show_stats:
        # Calculs
        sk1, kt1 = skew(data1.dropna()), kurtosis(data1.dropna(), fisher=False)
        sk2, kt2 = skew(data2.dropna()), kurtosis(data2.dropna(), fisher=False)
        median1 = data1.median()
        median2 = data2.median()

        # Interprétations
        def interpret(sk, kt):
            skew_desc = "symétrique" if abs(sk) < 0.5 else "asymétrie à droite" if sk > 0 else "asymétrie à gauche"
            kurt_desc = "normale" if abs(kt - 3) < 0.5 else "pointue" if kt > 3 else "plate"
            return f"Skew: {sk:.2f} ({skew_desc})\nKurt: {kt:.2f} ({kurt_desc})"

        text1 = f"{label1}\n" + interpret(sk1, kt1)
        text2 = f"{label2}\n" + interpret(sk2, kt2)

        # Annotations statistiques
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        plt.text(0.02, 0.98, text1, transform=plt.gca().transAxes,
                 fontsize=10, verticalalignment='top', bbox=props)
        plt.text(0.98, 0.98, text2, transform=plt.gca().transAxes,
                 fontsize=10, verticalalignment='top', horizontalalignment='right', bbox=props)

        # Lignes de médiane
        plt.axvline(median1, color=color1, linestyle="--", linewidth=1.2)
        plt.axvline(median2, color=color2, linestyle="--", linewidth=1.2)
        plt.text(median1, plt.gca().get_ylim()[1]*0.85, f"Médiane {label1} = {median1:.2f}",
                 color=color1, ha="right", va="top", fontsize=9, rotation=90)
        plt.text(median2, plt.gca().get_ylim()[1]*0.75, f"Médiane {label2} = {median2:.2f}",
                 color=color2, ha="left", va="top", fontsize=9, rotation=90)

    plt.tight_layout()
    plt.show()



def detect_fuzzy_duplicates(titles, threshold=0.9, max_pairs=50):
    """
    Détecte les paires de titres très similaires (fuzzy duplicates) via similarité cosinus sur TF-IDF.

    Cette fonction est utilisée dans un but exploratoire pour identifier des titres proches mais non identiques,
    qui pourraient indiquer des doublons flous, des reformulations ou des reposts partiels.

    ⚠️ Remarque : bien que la vectorisation TF-IDF soit une opération de feature engineering,
    elle est ici utilisée temporairement et localement à des fins de diagnostic exploratoire.
    Aucun vecteur n’est conservé ni utilisé pour l’entraînement de modèles.

    Parameters
    ----------
    titles : list or pandas.Series
        Liste des titres à comparer.
    threshold : float, optional
        Seuil de similarité au-dessus duquel deux titres sont considérés comme suspects (default: 0.9).
    max_pairs : int, optional
        Nombre maximum de paires à retourner pour inspection (default: 50).

    Returns
    -------
    pandas.DataFrame
        Tableau contenant les paires de titres similaires avec leur score de similarité.
        Colonnes : ['title_1', 'title_2', 'similarity']
    """
    # Vectorisation TF-IDF
    vectorizer = TfidfVectorizer().fit_transform(titles)
    cosine_sim = cosine_similarity(vectorizer)

    # Extraction des paires au-dessus du seuil (hors diagonale)
    pairs = []
    for i in range(len(titles)):
        for j in range(i + 1, len(titles)):
            sim = cosine_sim[i, j]
            if sim >= threshold:
                pairs.append((i, j, sim))

    # Tri et formatage
    pairs = sorted(pairs, key=lambda x: -x[2])[:max_pairs]
    results = pd.DataFrame(pairs, columns=["index_1", "index_2", "similarity"])
    results["title_1"] = results["index_1"].apply(lambda i: titles.iloc[i])
    results["title_2"] = results["index_2"].apply(lambda i: titles.iloc[i])
    return results[["title_1", "title_2", "similarity"]]





def detect_levenshtein_duplicates(titles, threshold=90, max_pairs=50):
    """
    Détecte les paires de titres similaires via distance de Levenshtein (fuzzy matching).

    Parameters
    ----------
    titles : list or pandas.Series
        Liste des titres à comparer.
    threshold : int, optional
        Seuil de similarité (0 à 100) au-dessus duquel deux titres sont considérés comme similaires (default: 90).
    max_pairs : int, optional
        Nombre maximum de paires à retourner (default: 50).

    Returns
    -------
    pandas.DataFrame
        Tableau contenant les paires de titres similaires avec leur score de similarité.
    """
    pairs = []
    for i in range(len(titles)):
        for j in range(i + 1, len(titles)):
            score = fuzz.ratio(titles[i], titles[j])
            if score >= threshold:
                pairs.append((titles[i], titles[j], score))

    pairs = sorted(pairs, key=lambda x: -x[2])[:max_pairs]
    return pd.DataFrame(pairs, columns=["title_1", "title_2", "similarity"])


def jaccard_similarity(str1, str2):
    set1, set2 = set(str1.lower().split()), set(str2.lower().split())
    intersection = set1 & set2
    union = set1 | set2
    return len(intersection) / len(union) if union else 0

def detect_jaccard_duplicates(titles, threshold=0.7, max_pairs=50):
    """
    Détecte les paires de titres similaires via similarité de Jaccard sur les mots.

    Parameters
    ----------
    titles : list or pandas.Series
        Liste des titres à comparer.
    threshold : float, optional
        Seuil de similarité (0 à 1) au-dessus duquel deux titres sont considérés comme similaires (default: 0.7).
    max_pairs : int, optional
        Nombre maximum de paires à retourner (default: 50).

    Returns
    -------
    pandas.DataFrame
        Tableau contenant les paires de titres similaires avec leur score de similarité.
    """
    pairs = []
    for i in range(len(titles)):
        for j in range(i + 1, len(titles)):
            score = jaccard_similarity(titles[i], titles[j])
            if score >= threshold:
                pairs.append((titles[i], titles[j], score))

    pairs = sorted(pairs, key=lambda x: -x[2])[:max_pairs]
    return pd.DataFrame(pairs, columns=["title_1", "title_2", "similarity"])


def detect_outliers_iqr(series):
    """
    Détecte les outliers d'une série numérique selon la méthode de l'IQR.

    Parameters
    ----------
    series : pandas.Series
        Série numérique à analyser.

    Returns
    -------
    pandas.Series
        Masque booléen indiquant les outliers.
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return (series < lower_bound) | (series > upper_bound)


def plot_boxplots_grid(df, columns, titles, colors, ncols=2, figsize=(12, 8)):
    """
    Affiche une grille de boxplots pour plusieurs colonnes d’un DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Le DataFrame contenant les données.
    columns : list of str
        Liste des noms de colonnes à tracer.
    titles : list of str
        Titres à afficher au-dessus de chaque boxplot.
    colors : list of str
        Couleurs à utiliser pour chaque boxplot.
    ncols : int, optional
        Nombre de colonnes dans la grille (default: 2).
    figsize : tuple, optional
        Taille de la figure matplotlib (default: (12, 8)).

    Returns
    -------
    None
    """
    n_plots = len(columns)
    nrows = (n_plots + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    for i, col in enumerate(columns):
        sns.boxplot(x=df[col], ax=axes[i], color=colors[i])
        axes[i].set_title(titles[i])

    # Supprimer les axes inutilisés si le nombre de plots < nrows * ncols
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def mark_outliers(df, columns, outlier_col="is_outlier", verbose=True, return_outliers=False):
    """
    Marque les lignes contenant des outliers sur au moins une des colonnes spécifiées.

    Parameters
    ----------
    df : pandas.DataFrame
        Le DataFrame à analyser (sera modifié en place).
    columns : list of str
        Liste des colonnes numériques à analyser.
    outlier_col : str, optional
        Nom de la colonne booléenne à créer pour indiquer les outliers (default: "is_outlier").
    verbose : bool, optional
        Si True, affiche un résumé des outliers détectés (default: True).
    return_outliers : bool, optional
        Si True, retourne un DataFrame contenant uniquement les lignes outliers.

    Returns
    -------
    pandas.DataFrame
        Le DataFrame avec une colonne booléenne indiquant les outliers.
        Si return_outliers=True, retourne également un DataFrame filtré avec uniquement les outliers.
    """
   

    combined_mask = np.zeros(len(df), dtype=bool)
    outlier_counts = {}

    for col in columns:
        mask = detect_outliers_iqr(df[col])
        combined_mask |= mask
        outlier_counts[col] = mask.sum()

    df[outlier_col] = combined_mask

    if verbose:
        print("Outliers détectés par variable :")
        for col, count in outlier_counts.items():
            print(f"- {col} : {count}")
        print(f"\nNombre total d’outliers (au moins une variable) : {df[outlier_col].sum()}")
        print(f"Proportion dans l’échantillon : {df[outlier_col].mean():.2%}")

    if return_outliers:
        return df, df[df[outlier_col]]
    else:
        return df


def remove_outliers(df, outlier_col="is_outlier"):
    """
    Retourne une copie du DataFrame sans les lignes marquées comme outliers.

    Parameters
    ----------
    df : pandas.DataFrame
        Le DataFrame contenant une colonne booléenne d’outliers.
    outlier_col : str, optional
        Nom de la colonne indiquant les outliers (default: "is_outlier").

    Returns
    -------
    pandas.DataFrame
        Nouveau DataFrame sans les outliers.
    """
    return df[~df[outlier_col]].reset_index(drop=True)


def plot_tag_occurrences(tag_freq_df, top_n=10, palette="viridis"):
    """
    Affiche un barplot des tags les plus fréquents en nombre d’occurrences,
    avec annotations sur chaque barre et une légende indiquant le nombre total de tags.

    Parameters:
    - tag_freq_df : DataFrame avec colonnes 'Tag' et 'Count'
    - top_n : nombre de tags à afficher
    - palette : palette de couleurs Seaborn
    """
    total = tag_freq_df["Count"].sum()
    df_top = tag_freq_df.sort_values(by="Count", ascending=False).head(top_n)

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(
        data=df_top,
        x="Count",
        y="Tag",
        hue="Tag",
        palette=palette,
        legend=False
    )
    plt.title(f"Top {top_n} des tags les plus fréquents (en occurrences)")
    plt.xlabel("Nombre d’occurrences")
    plt.ylabel("Tag")

    # Annotations sur chaque barre
    for p in ax.patches:
        width = p.get_width()
        plt.text(
            width + 1,
            p.get_y() + p.get_height() / 2,
            f"{int(width)}",
            va="center"
        )

    # Légende textuelle
    plt.figtext(
        0.99, -0.05,
        f"Nombre total de tags dans l’échantillon : {total:,}",
        horizontalalignment='right',
        fontsize=9,
        style='italic'
    )

    plt.tight_layout()
    plt.show()



def plot_tag_distribution(tag_freq_df, top_n=10, palette="viridis"):
    """
    Affiche la distribution des tags les plus fréquents en proportion,
    avec annotations en pourcentage et indication du nombre total de tags.

    Parameters:
    - tag_freq_df : DataFrame avec colonnes 'Tag' et 'Count'
    - top_n : nombre de tags à afficher
    - palette : palette de couleurs Seaborn
    """
    # Calcul de la proportion
    total = tag_freq_df["Count"].sum()
    df = tag_freq_df.copy()
    df["Proportion"] = df["Count"] / total

    # Sélection des top N
    df_top = df.sort_values(by="Proportion", ascending=False).head(top_n)

    # Plot
    plt.figure(figsize=(10, 5))
    ax = sns.barplot(
        data=df_top,
        x="Proportion",
        y="Tag",
        hue="Tag",
        palette=palette,
        legend=False
    )
    plt.title(f"Top {top_n} des tags les plus fréquents (en proportion)")
    plt.xlabel("Proportion des occurrences")
    plt.ylabel("Tag")

    # Annotations en pourcentage
    for p in ax.patches:
        width = p.get_width()
        plt.text(
            width + 0.001,
            p.get_y() + p.get_height() / 2,
            f"{width:.1%}",
            va="center"
        )

    # Ajout d'une légende textuelle
    plt.figtext(
        0.99, -0.05,
        f"Nombre total de tags dans l’échantillon : {total:,}",
        horizontalalignment='right',
        fontsize=9,
        style='italic'
    )

    plt.tight_layout()
    plt.show()


def mark_dominant_tags_by_frequency_and_coverage(tag_freq_df, nb_questions, min_coverage=0.8, min_count_floor=3):
    """
    Marque comme dominants les tags qui apparaissent au moins min_count fois
    (déduit dynamiquement à partir de la taille du corpus et d’un seuil plancher)
    et qui permettent de couvrir min_coverage du corpus.

    Parameters:
    - tag_freq_df : DataFrame avec colonnes 'Tag' et 'Count'
    - nb_questions : nombre total de questions dans le corpus
    - min_coverage : proportion minimale du corpus à couvrir (entre 0 et 1)
    - min_count_floor : seuil plancher minimal pour la fréquence d’un tag

    Returns:
    - DataFrame enrichi avec 'Proportion', 'Cumulative' et 'is_dominant_tag'
    - Liste des tags dominants
    """
    min_count = max(min_count_floor, int(0.001 * nb_questions))

    df = tag_freq_df.copy()
    df = df[df["Count"] >= min_count].sort_values(by="Count", ascending=False).reset_index(drop=True)

    total = tag_freq_df["Count"].sum()
    df["Proportion"] = df["Count"] / total
    df["Cumulative"] = df["Proportion"].cumsum()

    df["is_dominant_tag"] = df["Cumulative"] <= min_coverage

    if not df[df["is_dominant_tag"]].empty:
        last_index = df[df["is_dominant_tag"]].index.max()
        if last_index + 1 < len(df):
            df.loc[last_index + 1, "is_dominant_tag"] = True

    dominant_tags = df[df["is_dominant_tag"]]["Tag"].tolist()
    return df, dominant_tags


def plot_tag_coverage(tag_freq_df_marked, coverage_target=0.8, title_suffix=""):
    """
    Affiche la courbe de couverture cumulative des tags.

    Parameters:
    - tag_freq_df_marked : DataFrame contenant les colonnes 'Cumulative' et 'Count'
    - coverage_target : seuil de couverture à afficher (ex. 0.8 pour 80 %)
    - title_suffix : texte à ajouter au titre pour préciser le contexte (ex. '100 questions')
    """
    df_plot = tag_freq_df_marked.sort_values(by="Count", ascending=False).reset_index(drop=True)

    plt.figure(figsize=(10, 6))
    sns.lineplot(
        data=df_plot,
        x=df_plot.index + 1,
        y="Cumulative",
        marker="o",
        label="Couverture cumulative"
    )

    plt.axhline(y=coverage_target, color="red", linestyle="--", label=f"Seuil de {int(coverage_target * 100)} %")

    plt.title(f"Couverture cumulative des tags {f'({title_suffix})' if title_suffix else ''}")
    plt.xlabel("Nombre de tags inclus")
    plt.ylabel("Proportion cumulée des occurrences")
    plt.xticks(range(1, len(df_plot) + 1))
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def build_tag_cooccurrence_matrix(df_questions, dominant_tags, tag_col="Tags"):
    """
    Construit une matrice de co-occurrence binaire entre tags dominants.

    Parameters:
    - df_questions : DataFrame contenant une colonne de listes de tags
    - dominant_tags : liste des tags à inclure dans la matrice
    - tag_col : nom de la colonne contenant les listes de tags

    Returns:
    - cooc_matrix : DataFrame carrée (tags × tags) avec les co-occurrences
    - tag_matrix : matrice binaire (questions × tags)
    """
    # Création d’une matrice binaire : 1 si le tag est présent dans la question
    tag_matrix = pd.DataFrame([
        {tag: int(tag in tags) for tag in dominant_tags}
        for tags in df_questions[tag_col]
    ])

    # Matrice de co-occurrence : produit matriciel
    cooc_matrix = tag_matrix.T.dot(tag_matrix)

    return cooc_matrix, tag_matrix

def plot_tag_cooccurrence_heatmap(cooc_matrix, figsize=(12, 10), cmap="Blues"):
    """
    Affiche une heatmap des co-occurrences entre tags.

    Parameters:
    - cooc_matrix : DataFrame carrée (tags × tags)
    - figsize : taille de la figure
    - cmap : palette de couleurs
    """
    plt.figure(figsize=figsize)
    sns.heatmap(
        cooc_matrix,
        annot=True,
        fmt="d",
        cmap=cmap,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Nombre de co-occurrences"}
    )
    plt.title("Heatmap des co-occurrences entre tags")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()



def plot_tag_cooccurrence_graph(cooc_matrix, min_edge_weight=1, layout="spring", figsize=(12, 10)):
    """
    Affiche un graphe de co-occurrence entre tags.

    Parameters:
    - cooc_matrix : DataFrame carrée (tags × tags)
    - min_edge_weight : seuil minimal de co-occurrence pour afficher une arête
    - layout : type de mise en page ('spring', 'circular', 'kamada_kawai')
    - figsize : taille de la figure
    """
    # Création du graphe
    G = nx.Graph()

    # Ajout des arêtes avec poids
    for i in cooc_matrix.index:
        for j in cooc_matrix.columns:
            if i != j and cooc_matrix.loc[i, j] >= min_edge_weight:
                G.add_edge(i, j, weight=cooc_matrix.loc[i, j])

    # Choix du layout
    if layout == "spring":
        pos = nx.spring_layout(G, seed=42)
    elif layout == "circular":
        pos = nx.circular_layout(G)
    elif layout == "kamada_kawai":
        pos = nx.kamada_kawai_layout(G)
    else:
        raise ValueError("Layout non reconnu")

    # Taille des nœuds proportionnelle au degré
    node_sizes = [300 + 100 * G.degree(n) for n in G.nodes()]
    edge_widths = [G[u][v]["weight"] for u, v in G.edges()]

    # Affichage
    plt.figure(figsize=figsize)
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color="skyblue")
    nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.6)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold")
    plt.title("Graphe de co-occurrence des tags")
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def vectorize_tfidf(
    text_series,
    label="text",
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    top_n=15,
    show_wordcloud=True
):
    """
    Applique la vectorisation TF-IDF sur une série de textes et affiche les mots les plus informatifs.

    Parameters
    ----------
    text_series : pd.Series
        Série contenant les textes nettoyés à vectoriser.
    label : str, default="text"
        Nom de la source (ex: "title", "body") pour les affichages.
    max_features : int, default=5000
        Nombre maximal de termes conservés dans le vocabulaire.
    ngram_range : tuple (int, int), default=(1, 2)
        Plage des n-grammes à extraire (ex: (1,2) = unigrammes + bigrammes).
    min_df : int, default=2
        Fréquence minimale d’apparition d’un terme pour être conservé.
    top_n : int, default=15
        Nombre de mots à afficher dans le barplot.
    show_wordcloud : bool, default=True
        Si True, affiche un WordCloud des termes pondérés par leur poids TF-IDF.

    Returns
    -------
    X_tfidf : scipy.sparse matrix
        Matrice TF-IDF (documents × termes).
    vocab : np.ndarray
        Tableau des termes du vocabulaire.
    top_words : list of tuples
        Liste des top_n mots avec leur poids TF-IDF total.
    """
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df
    )
    
    X_tfidf = vectorizer.fit_transform(text_series)
    vocab = vectorizer.get_feature_names_out()
    tfidf_sum = np.asarray(X_tfidf.sum(axis=0)).flatten()
    
    top_indices = tfidf_sum.argsort()[::-1][:top_n]
    top_words = [(vocab[i], tfidf_sum[i]) for i in top_indices]
    
    # Affichage console
    print(f"Shape de la matrice TF-IDF ({label}) : {X_tfidf.shape}")
    print(f"Top {top_n} mots ({label}) par poids TF-IDF :")
    for word, score in top_words:
        print(f"{word}: {score:.2f}")
    
    # Barplot
    df_top = pd.DataFrame(top_words, columns=["mot", "poids"])
    plt.figure(figsize=(10, 5))
    sns.barplot(data=df_top, x="poids", y="mot", hue="mot", dodge=False, legend=False, palette="viridis")
    plt.title(f"Top {top_n} mots TF-IDF – {label}")
    plt.xlabel("Poids TF-IDF total")
    plt.ylabel("Mot")
    plt.tight_layout()
    plt.show()
    
    # WordCloud
    if show_wordcloud:
        word_weights = {vocab[i]: tfidf_sum[i] for i in range(len(vocab))}
        wc = WordCloud(width=800, height=400, background_color='white', colormap='viridis')
        wc.generate_from_frequencies(word_weights)
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"WordCloud – {label}")
        plt.tight_layout()
        plt.show()
    
    return X_tfidf, vocab, top_words, vectorizer


def apply_svd_and_plot(X_tfidf, label="text", n_components=2, random_state=42):
    """
    Applique une réduction de dimension par SVD (LSA) sur une matrice TF-IDF
    et affiche les documents projetés dans l’espace latent.

    Parameters
    ----------
    X_tfidf : sparse matrix
        Matrice TF-IDF à réduire.
    label : str
        Nom de la source textuelle (title, body, title + body).
    n_components : int
        Nombre de dimensions à conserver (2 pour visualisation).
    random_state : int
        Graine aléatoire pour reproductibilité.

    Returns
    -------
    X_reduced : ndarray
        Matrice réduite (documents × n_components).
    svd_model : TruncatedSVD
        Modèle SVD entraîné.
    """
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    X_reduced = svd.fit_transform(X_tfidf)

    print(f"Explained variance ({label}) : {svd.explained_variance_ratio_.sum():.2%}")

    # Visualisation 2D
    if n_components == 2:
        df_proj = pd.DataFrame(X_reduced, columns=["dim1", "dim2"])
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df_proj, x="dim1", y="dim2", s=60, color="royalblue", edgecolor="black")
        plt.title(f"Projection SVD – {label}")
        plt.xlabel("Composante 1")
        plt.ylabel("Composante 2")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return X_reduced, svd

def apply_svd_variance(X_tfidf, label="text", n_components=100, random_state=42):
    """
    Applique SVD avec n composantes et affiche la variance cumulée expliquée.

    Parameters
    ----------
    X_tfidf : sparse matrix
        Matrice TF-IDF à réduire.
    label : str
        Nom de la source textuelle.
    n_components : int
        Nombre de dimensions à conserver.
    random_state : int
        Graine aléatoire pour reproductibilité.

    Returns
    -------
    X_reduced : ndarray
        Matrice réduite (documents × n_components).
    svd_model : TruncatedSVD
        Modèle SVD entraîné.
    """
    svd = TruncatedSVD(n_components=n_components, random_state=random_state)
    X_reduced = svd.fit_transform(X_tfidf)
    variance_cum = svd.explained_variance_ratio_.cumsum()

    print(f"Variance expliquée cumulée ({label}) :")
    for k in [10, 20, 50, 100]:
        if k <= n_components:
            print(f" - {k} composantes : {variance_cum[k-1]:.2%}")

    # Courbe de variance
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, n_components + 1), variance_cum, marker='o', color='royalblue')
    plt.title(f"Variance cumulée expliquée – {label}")
    plt.xlabel("Nombre de composantes")
    plt.ylabel("Variance expliquée cumulée")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return X_reduced, svd


def plot_tsne(embeddings, label="text"):
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced = tsne.fit_transform(embeddings)
    df = pd.DataFrame(reduced, columns=["dim1", "dim2"])
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="dim1", y="dim2", s=60, color="darkorange", edgecolor="black")
    plt.title(f"Projection t-SNE – {label}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_umap(embeddings, label="text", n_neighbors=15, min_dist=0.1):
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    reduced = reducer.fit_transform(embeddings)
    df = pd.DataFrame(reduced, columns=["dim1", "dim2"])
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="dim1", y="dim2", s=60, color="mediumseagreen", edgecolor="black")
    plt.title(f"Projection UMAP – {label}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def tokenize_spacy(text, nlp):
    doc = nlp(text.lower())
    return [token.text for token in doc if token.is_alpha and not token.is_stop]

def document_vector_spacy(doc, model, nlp):
    tokens = tokenize_spacy(doc, nlp)
    vectors = [model[word] for word in tokens if word in model]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(model.vector_size)


from sklearn.feature_extraction.text import CountVectorizer

def build_bow_matrix(df, col_title="clean_title", col_body="clean_body",
                     max_df=0.95, min_df=2, stop_words="english"):
    """
    Vectorise le corpus en Bag-of-Words à partir du titre + corps concaténés.
    
    Parameters:
    - df : pd.DataFrame contenant les colonnes textuelles
    - col_title / col_body : noms des colonnes à concaténer
    - max_df / min_df : filtres de fréquence pour CountVectorizer
    - stop_words : langue pour la suppression des stopwords
    
    Returns:
    - corpus : liste des textes concaténés
    - X_bow : matrice sparse BoW
    - vocab : liste des tokens conservés
    """
    corpus = df[[col_title, col_body]].apply(
        lambda x: " ".join(x.dropna().astype(str)), axis=1
    )
    
    vectorizer = CountVectorizer(max_df=max_df, min_df=min_df, stop_words=stop_words)
    X_bow = vectorizer.fit_transform(corpus)
    vocab = vectorizer.get_feature_names_out().tolist()
    
    return corpus, X_bow, vocab, vectorizer





def train_word2vec(corpus, save_path="model_word2vec.bin",
                   vector_size=100, window=5, min_count=3,
                   workers=1, sg=1, epochs=10, verbose=True):
    """
    Entraîne un modèle Word2Vec à partir du corpus texte pré-nettoyé.
    
    Parameters:
    - corpus : liste de phrases (chaînes de caractères) déjà nettoyées
    - save_path : chemin pour sauvegarder le modèle entraîné
    - vector_size, window, min_count, sg, workers, epochs : paramètres Word2Vec
    - verbose : affiche les logs si True
    
    Returns:
    - model : modèle entraîné (gensim Word2Vec)
    """
    if verbose:
        print("🔄 Préparation du corpus...")
    
    # Tokenisation de chaque phrase
    try:
        sentences = [sentence.split() for sentence in corpus]
    except Exception as e:
        print("🚨 Erreur de tokenisation :", e)
        return None

    if verbose:
        print(f"🧠 Corpus prêt avec {len(sentences)} documents.")

    # Activation du logger de Gensim (si souhaité)
    if verbose:
        logging.basicConfig(format="%(levelname)s - %(asctime)s - %(message)s", level=logging.INFO)

    try:
        # Entraînement du modèle Word2Vec
        model = Word2Vec(
            sentences=sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
            sg=sg,
            epochs=epochs
        )

        # Sauvegarde du modèle
        model.save(save_path)
        if verbose:
            print(f"💾 Modèle sauvegardé dans {os.path.abspath(save_path)}")

        return model

    except Exception as e:
        print("❌ Erreur lors de l'entraînement :", e)
        return None



import kagglehub
import tensorflow_hub as hub

use_model = None

def encode_with_use(text: str) -> np.ndarray:
    global use_model
    if use_model is None:
        try:
            path = kagglehub.model_download("google/universal-sentence-encoder/tensorFlow2/universal-sentence-encoder")
            use_model = hub.load(path)
            print(f"✅ USE chargé depuis : {path}")
        except Exception as e:
            print(f"❌ Erreur chargement USE : {str(e)}")
            raise
    return use_model([text])[0].numpy()

import numpy as np

def encode_use_corpus(corpus, model_path=None, batch_size=100, verbose=True):
    if model_path is None:
        model_path = kagglehub.model_download("google/universal-sentence-encoder/tensorFlow2/universal-sentence-encoder")
    use_model = hub.load(model_path)
    print(f"✅ USE chargé depuis : {model_path}")
    embeddings = []
    for i in range(0, len(corpus), batch_size):
        batch = corpus[i:i+batch_size]
        emb = use_model(batch).numpy()
        embeddings.append(emb)
        # if verbose:
        #     print(f"🔄 Batch {i} → {i + len(batch)} encodé")

    return np.vstack(embeddings)



sbert_model = None

def get_sbert_model(model_name="all-MiniLM-L6-v2"):
    global sbert_model
    if sbert_model is None:
        print("🔄 Chargement du modèle SBERT...")
        sbert_model = SentenceTransformer(model_name)
        print("✅ SBERT chargé :", model_name)
    return sbert_model


def encode_sbert_corpus(corpus, model=None, batch_size=32, verbose=True):
    if model is None:
        model = get_sbert_model()
    embeddings = []
    for i in range(0, len(corpus), batch_size):
        batch = corpus[i:i + batch_size]
        emb = model.encode(batch, show_progress_bar=False)
        embeddings.append(emb)
        # if verbose:
        #     print(f"➡️ Batch {i} à {i+len(batch)} encodé")

    return np.vstack(embeddings)


def vectorize_texts(texts, w2v_model):
    vectors = []
    for sentence in texts:
        words = sentence.split()
        word_vecs = [w2v_model.wv[word] for word in words if word in w2v_model.wv]
        if word_vecs:
            vectors.append(np.mean(word_vecs, axis=0))
        else:
            vectors.append(np.zeros(w2v_model.vector_size))
    return np.array(vectors)