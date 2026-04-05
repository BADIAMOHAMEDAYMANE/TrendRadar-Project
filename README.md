# 📡 TrendRadar

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Visualisation-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/Licence-MIT-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-En%20développement-yellow?style=for-the-badge)

> Application web de détection et visualisation de tendances Twitter/X, propulsée par le Machine Learning (KMeans + TF-IDF).

---

## 🚀 Fonctionnalités

- 🔍 **Collecte de tweets** simulée par sujet/requête
- 🧹 **Nettoyage NLP** bilingue (français / anglais) avec suppression des stopwords
- 🤖 **Clustering KMeans** pour regrouper les tweets par thématique
- 📊 **Visualisations interactives** : barres d'engagement, treemap des hashtags, nuage de mots
- ⚡ **Auto-refresh** configurable toutes les 5 minutes
- 🎛️ **Interface Streamlit** intuitive avec paramètres dynamiques

---

## 🗂️ Structure du projet

```
trendradar/
├── src/
│   ├── collect.py        # Collecte et simulation de tweets
│   ├── preprocess.py     # Nettoyage NLP + vectorisation TF-IDF
│   ├── model.py          # Clustering KMeans + extraction de hashtags
│   └── visualize.py      # Graphiques Plotly + WordCloud
├── app.py                # Interface Streamlit principale
├── requirements.txt      # Dépendances Python
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

```bash
# 1. Cloner le dépôt
git clone https://github.com/TON_USERNAME/trendradar.git
cd trendradar

# 2. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Windows : venv\Scripts\activate

# 3. Installer les dépendances
pip install -r requirements.txt
```

---

## ▶️ Lancer l'application

```bash
streamlit run app.py
```

L'application sera accessible sur `http://localhost:8501`

---

## 🧪 Pipeline de données

```
Utilisateur (paramètres)
        │
        ▼
  collect.py        →   DataFrame brut (tweets, likes, retweets)
        │
        ▼
  preprocess.py     →   Texte nettoyé + matrice TF-IDF
        │
        ▼
  model.py          →   Clusters thématiques + top hashtags
        │
        ▼
  visualize.py      →   Graphiques Plotly + WordCloud
        │
        ▼
  app.py            →   Interface web Streamlit
```

---

## 📦 Dépendances principales

| Package | Rôle |
|---|---|
| `streamlit` | Interface web |
| `pandas` | Manipulation des données |
| `scikit-learn` | TF-IDF + KMeans |
| `plotly` | Graphiques interactifs |
| `wordcloud` | Nuage de mots |
| `nltk` | Stopwords NLP |

---

## 🔮 Améliorations futures

- [ ] Connexion à la vraie API Twitter/X v2
- [ ] Analyse de sentiment (positif / négatif / neutre)
- [ ] Export des résultats en CSV/PDF
- [ ] Déploiement sur Streamlit Cloud

---

## 👤 Auteur

**Aymane Badia**
[![GitHub](https://img.shields.io/badge/GitHub-AymaneBadia-181717?style=flat&logo=github)](https://github.com/TON_USERNAME)

---

## 📄 Licence

Ce projet est sous licence **MIT** — libre d'utilisation, de modification et de distribution.
