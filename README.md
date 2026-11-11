# 📡 TrendRadar — Détection de tendances Reddit

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![spaCy](https://img.shields.io/badge/spaCy-NLP-09A3D5?style=for-the-badge&logo=spacy&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Visualisation-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/Licence-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-En%20développement-yellow?style=for-the-badge)

> Application web qui collecte des publications **Reddit**, les analyse avec un pipeline NLP **français / anglais** et détecte les sujets émergents grâce au Machine Learning.

---

## 🚀 Fonctionnalités

- 🔍 **Collecte Reddit** : recherche par requête et par subreddits via les endpoints JSON publics de Reddit (sans clé API), avec nouvelles tentatives, flux continu et cache Parquet.
- 🧹 **NLP bilingue** : détection de langue (`langdetect`), nettoyage, stopwords NLTK, lemmatisation et **reconnaissance d'entités nommées** (personnes, organisations, lieux) avec **spaCy** (FR + EN), vectorisation **TF-IDF**.
- 🤖 **Clustering thématique KMeans** sur la matrice TF-IDF (nombre de clusters réglable).
- 🌲 **Random Forest** entraîné sur les thèmes trouvés par KMeans, pour classer de nouvelles publications (accuracy en validation croisée affichée dans l'interface).
- ⚡ **Détection de pics (bursts)** : score Bt par fenêtres glissantes pour repérer les termes et hashtags qui émergent, avec seuil d'alerte réglable.
- 📈 **Prédiction de viralité** : gradient boosting (XGBoost, LightGBM, ou GradientBoosting de scikit-learn s'ils ne sont pas installés), évaluée par F1-macro en validation croisée.
- 📊 **Visualisations interactives Plotly** : tendances, engagement, importance des variables, hashtags, chronologie, pics, viralité, nuage de mots.

---

## 🗂️ Structure du projet

```
TrendRadar-Project/
├── src/
│   ├── collect.py        # Collecte Reddit (JSON public), cache Parquet, flux continu
│   ├── preprocess.py     # Langue, nettoyage, spaCy (lemmes + NER), TF-IDF
│   ├── model.py          # KMeans, Random Forest, modèle de viralité
│   ├── burst.py          # Détection de pics (score Bt, fenêtres glissantes)
│   └── visualize.py      # Graphiques Plotly
├── app.py                # Interface Streamlit (onglets)
├── requirements.txt      # Dépendances Python (+ modèles spaCy FR/EN)
├── packages.txt          # Paquets système (Streamlit Cloud)
├── runtime.txt           # Version de Python (Streamlit Cloud)
└── README.md
```

---

## 🧪 Pipeline

```
Paramètres (requête, subreddits, nb de clusters, seuil Bt)
        │
        ▼
collect.py     →  publications Reddit (titre, texte, score, commentaires, auteur, subreddit…)
        │
        ▼
preprocess.py  →  langue, texte nettoyé, lemmes, entités, matrice TF-IDF
        │
        ├──► model.py   →  clusters KMeans → Random Forest → viralité
        ├──► burst.py   →  termes émergents (score Bt)
        ▼
visualize.py + app.py  →  tableau de bord Streamlit
```

---

## ⚙️ Installation

```bash
git clone https://github.com/BADIAMOHAMEDAYMANE/TrendRadar-Project.git
cd TrendRadar-Project

python -m venv venv
source venv/bin/activate      # Windows : venv\Scripts\activate
pip install -r requirements.txt

streamlit run app.py          # http://localhost:8501
```

---

## ⚠️ Limites connues

- Le Random Forest est entraîné sur les **étiquettes produites par KMeans** (pseudo-labels) : son accuracy mesure la cohérence avec le clustering, pas une vérité terrain annotée.
- Aucun résultat chiffré n'est publié : les métriques dépendent des données collectées au moment de l'exécution.

---

## 🔮 Améliorations futures

- [ ] Jeu de données annoté pour évaluer les thèmes sur une vraie vérité terrain
- [ ] Analyse de sentiment (positif / négatif / neutre)
- [ ] Export des résultats en CSV / PDF

---

## 👤 Auteur

**Mohamed Aymane Badia** — élève ingénieur en 5ᵉ année, IA & Data Science (EMSI)
[![GitHub](https://img.shields.io/badge/GitHub-BADIAMOHAMEDAYMANE-181717?style=flat&logo=github)](https://github.com/BADIAMOHAMEDAYMANE)

---

## 📄 Licence

Ce projet est sous licence **MIT**.
