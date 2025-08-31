# NOTE TECHNIQUE – APPROCHE MLOPS POUR LA SUGGESTION DE TAGS STACK OVERFLOW

## 1. INTRODUCTION

Ce projet vise à développer un système de suggestion automatique de tags pour les questions posées sur Stack Overflow. L’objectif est d’améliorer la qualité du tagging, notamment pour les nouveaux utilisateurs, en proposant des tags pertinents à partir du contenu textuel des questions.

Dans ce contexte, une démarche MLOps a été adoptée afin d’industrialiser le cycle de vie du modèle supervisé. Cette note technique présente les outils et pratiques utilisés dans le projet, et propose une réflexion sur leur généralisation à d’autres cas d’usage.

---

## 2. SUIVI DES EXPERIMENTATIONS AVEC MLFLOW

**Objectif :** assurer la traçabilité des modèles, des hyperparamètres et des métriques.

- Mise en œuvre : MLFlow a été intégré dans le pipeline d’entraînement pour enregistrer les paramètres, les scores et les artefacts.
- Avantages :
  - Reproductibilité des expérimentations
  - Visualisation des performances dans une interface dédiée
  - Comparaison facile entre plusieurs modèles
- Usage : chaque entraînement est loggé avec son vectoriseur, son modèle, et ses métriques.
MLFlow centralise tous les éléments liés à une expérimentation : les paramètres, les scores, les fichiers de sortie (CSV, graphiques), et le modèle final. Ces artefacts sont accessibles via l’interface web, ce qui facilite leur analyse et leur réutilisation.

MLFlow m’a permis de suivre et documenter précisément mes expérimentations, notamment celle intitulée logreg_stackoverflow. L’interface web offre une vue synthétique des différents runs, avec accès aux paramètres, métriques, artefacts et modèles enregistrés.

![Figure 1 – Vue d’ensemble du run logreg_stackoverflow](images/00-mlflow-dashboard.png)

![Figure 2 – Artefacts générés : barplot des métriques de performance](images/01-mlflow-barplot_f1_micro.png)

![Figure 3 – Artefacts générés : barplot couverture de tags](images/02-mlflow-barplot_coverage_tags.png)

![Figure 4 – Artefacts générés : tableau comparatif des métriques vecteurs](images/03-mlflow-comparatif-vecteurs.png)


---

## 3. ANALYSE DU DRIFT AVEC EVIDENTLYAI

**Objectif :** Suivre l’évolution des performances du modèle de classification des tags au fil du temps, et détecter les dérives éventuelles dans les données ou les prédictions.

###  METHODOLOGIE SUPERVISEE

- Encodage mensuel des textes via **SBERT**
- Prédiction des tags avec un modèle de **régression logistique**
- Évaluation des performances mensuelles (précision, couverture, etc.)
- Génération de rapports HTML avec **EvidentlyAI** pour chaque mois
- Comparaison globale entre janvier et décembre 2014

###  METHIODOLOGIE NON SUPERVISEE

- Calcul de la **distance cosinus** entre centroïdes d’embeddings mensuels
- Visualisation des corpus via **PCA** (projection en 2D)
- Tests statistiques :
  - **Wasserstein distance**
  - **Kolmogorov-Smirnov** (KS test)

###  OUTIL UTILISE

- **EvidentlyAI v0.5.1**, intégré dans le pipeline d’analyse
- Rapports sauvegardés automatiquement dans le dossier `evidently_reports`

###  DISCUSSION

Les résultats montrent des variations significatives dans les embeddings et les performances du modèle selon les mois. Ces dérives peuvent impacter la pertinence des tags proposés. Un suivi régulier, combinant des approches supervisées et non supervisées, est essentiel pour garantir la robustesse du système en production.

---

## 4. CREATION DE L'API AVEC FASTAPI + DEMO STREAMLIT

** Objectif :** exposer le modèle via une API REST et proposer une interface interactive pour tester et démontrer ses performances.

###  FASTAPI

- Création d’un endpoint `/predict/` recevant un titre et un corps de question
- Retour des tags prédits sous forme de liste JSON
- Documentation interactive générée automatiquement via Swagger (`/docs`)
- Déploiement possible sur le cloud (Heroku dans ce projet)

###  STREAMLIT – INTERFACE DE DEMONSTRATION

- Interface épurée et intuitive permettant de :
  - Saisir un titre et une question
  - Envoyer une requête POST à l’API FastAPI
  - Afficher les tags suggérés en temps réel
- Utilisation en local ou en ligne pour tester rapidement le modèle


###  ATOUT STREAMLIT : DEMONSTRATION VIDEO

> L’un des avantages majeurs de Streamlit est la possibilité d’enregistrer une démonstration vidéo du fonctionnement de l’application. Cela permet de :
> - Présenter le projet à des non-techniciens
> - Valoriser le modèle dans un portfolio ou lors d’un pitch
> - Documenter visuellement les fonctionnalités pour les utilisateurs

###  ARCHITECTURE

- Le front Streamlit agit comme client
- Il envoie une requête POST à l’API FastAPI déployée sur le cloud
- L’API retourne les prédictions, qui sont affichées dynamiquement dans l’interface


---

## 5. DEPLOIEMENNT DANS LE CLOUD AVEC HEROKU

**Objectif :** rendre l’API accessible en ligne pour la démonstration.

###  MISE EN PLACE TECHNIQUE

- Déploiement initial prévu via GitHub et Heroku (build automatique)
- Contrainte rencontrée : **limite de taille du slug Heroku** (500 Mo max)
- Création d’un fichier `.slugignore` pour exclure les fichiers non essentiels
- Séparation des environnements :
  - Un dépôt GitHub complet avec tests et CI (GitHub Actions)
  - Un répertoire allégé dédié au déploiement Heroku, avec un `requirements.txt` spécifique

###  DIFFICULTES RENCONTREES

- La limite du slug a cassé la logique de déploiement automatique depuis GitHub
- Obligation de maintenir **deux versions du `requirements.txt`** :
  - Une pour GitHub (tests, dev, CI)
  - Une pour Heroku (prod allégée)
- Le push depuis VS Code ne suffisait plus : il fallait gérer manuellement les deux flux de déploiement

###  RESULTAT

- L’API est bien en ligne sur Heroku
- Mais le processus de déploiement est devenu plus complexe que prévu, avec une gestion fine des dépendances et des fichiers

###  RETOUR D'EXPERIENCE

> Heroku reste une solution rapide pour des démos, mais dès qu’on dépasse le cadre minimal, on se heurte à des limites techniques qui demandent des contournements. Dans mon cas, la scalabilité n’était pas le problème — c’est la gestion du slug et des dépendances qui a complexifié le workflow.


---

## 6. GESTION DE VERSION AVEC GIT ET GITHUB

**Objectif :** assurer le suivi du code source et faciliter la collaboration.

- Organisation :
  - Structure claire du dépôt (`src/`, `tests/`, `models/`, etc.)
  - Branches pour les différentes étapes du projet
- Bonnes pratiques :
  - Commits explicites
  - Utilisation de `.gitignore` pour exclure les fichiers sensibles

---

## 7. INTEGRATION CONTINUE AVEC GITHUB ACTIONS

** Objectif :** automatiser les tests et garantir la fiabilité du code à chaque modification du dépôt.

###  WORKFLOW CI (`ci.yml`)

Le fichier `ci.yml` définit un processus d’intégration continue déclenché à chaque `push`. Il est conçu pour fonctionner sous Windows avec Python 3.11, et comprend les étapes suivantes :

- ** Préparation de l’environnement** :
  - Exécution sur `windows-latest`
  - Installation de Python via `actions/setup-python@v4`
  - Nettoyage complet du projet :
    - Suppression des fichiers `.pyc`, des dossiers `__pycache__`, des métadonnées `*.egg-info`, et des dossiers `dist` / `build`
    - Purge du cache pip pour éviter les conflits

- ** Installation des dépendances** :
  - Mise à jour de `pip`
  - Installation des packages depuis `requirements.txt`
  - Installation du projet en mode *editable* (`pip install -e .`)

- ** Vérification du modèle SpaCy** :
  - Validation de la présence du modèle `en_core_web_sm`
  - Téléchargement automatique si absent

- ** Lancement des tests unitaires** :
  - Exécution de `pytest` sur l’ensemble du projet
  - Vérification du bon fonctionnement des modules et de l’API

###  AVANNTAGES

- **Détection proactive des erreurs** dès le push
- **Nettoyage systématique** pour éviter les artefacts obsolètes
- **Validation des dépendances NLP** (comme SpaCy) pour garantir la reproductibilité
- **Fiabilité renforcée** avant chaque déploiement ou merge
- **Gain de temps** pour les développeurs et meilleure traçabilité des modifications

###  BONNNES PRATIQUES INTEGREES

> Ce workflow montre une attention particulière à la propreté du code et à la robustesse des tests. Il permet de maintenir un environnement sain et cohérent, tout en automatisant les vérifications critiques.



---

## 8. DISCUSSION SUR LA GENERALISATION

Les outils et pratiques utilisés dans ce projet peuvent être réutilisés dans d’autres contextes :

- MLFlow pour tout projet de machine learning nécessitant un suivi des expérimentations
- EvidentlyAI pour le monitoring de modèles en production
- FastAPI pour exposer des modèles via une API
- GitHub Actions pour automatiser les tests et les déploiements

**Limites rencontrées :**
- Complexité de configuration initiale
- Nécessité de bien structurer le code dès le départ

---

## 9. CONCLUSION

La démarche MLOps adoptée dans ce projet a permis de structurer efficacement le développement, le suivi et le déploiement du modèle de suggestion de tags. Les outils choisis sont adaptés à un contexte open source et peuvent être généralisés à d’autres projets.

Cette approche garantit une meilleure qualité, une meilleure traçabilité, et une plus grande robustesse du système mis en place.
