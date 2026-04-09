"""
model.py — Pipeline KMeans (détection) + Random Forest (classification)

Flux semi-supervisé :
  1. KMeans  → assigne un cluster à chaque post (label non supervisé)
  2. Random Forest → apprend à prédire ce cluster depuis le TF-IDF
                     et peut classifier de nouveaux posts en temps réel
"""

import re
import pickle
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Dict
from collections import Counter
from scipy.sparse import spmatrix
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer


# ─── Étiquettes de clusters ──────────────────────────────────────────────────

def _label_cluster(keywords: List[str]) -> str:
    """
    Génère un label lisible à partir des mots-clés dominants d'un cluster.
    Retourne "Sujet · mot1 · mot2 · mot3".
    """
    return " · ".join(keywords[:3])


# ─── KMeans ──────────────────────────────────────────────────────────────────

def detect_trends(
    df: pd.DataFrame,
    matrix: spmatrix,
    vectorizer: TfidfVectorizer,
    n_clusters: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame, KMeans]:
    """
    Applique KMeans sur la matrice TF-IDF et calcule les métriques par cluster.

    Returns:
        (trends_df, df_with_clusters, kmeans_model)
        trends_df colonnes : cluster, label, keywords, post_count,
                             avg_score, avg_comments, engagement_score
        df_with_clusters   : df original enrichi avec la colonne 'cluster'
    """
    n_clusters = min(n_clusters, len(df))

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df = df.copy()
    df["cluster"] = kmeans.fit_predict(matrix)

    feature_names = vectorizer.get_feature_names_out()
    trends = []

    for cid in range(n_clusters):
        center = kmeans.cluster_centers_[cid]
        top_idx = center.argsort()[-10:][::-1]
        keywords = [feature_names[i] for i in top_idx]

        cluster_df = df[df["cluster"] == cid]
        engagement = round(
            cluster_df["score"].sum() + cluster_df["num_comments"].sum() * 1.5, 0
        )

        trends.append({
            "cluster": cid,
            "label": _label_cluster(keywords),
            "keywords": ", ".join(keywords[:6]),
            "post_count": len(cluster_df),
            "avg_score": round(cluster_df["score"].mean(), 1),
            "avg_comments": round(cluster_df["num_comments"].mean(), 1),
            "engagement_score": engagement,
        })

    trends_df = (
        pd.DataFrame(trends)
        .sort_values("engagement_score", ascending=False)
        .reset_index(drop=True)
    )
    return trends_df, df, kmeans


# ─── Random Forest ───────────────────────────────────────────────────────────

def train_classifier(
    df: pd.DataFrame,
    matrix: spmatrix,
    cv: int = 3,
) -> Tuple[RandomForestClassifier, float]:
    """
    Entraîne un Random Forest sur les labels KMeans.

    Args:
        df    : DataFrame avec colonne 'cluster' (issue de detect_trends)
        matrix: matrice TF-IDF correspondante
        cv    : nombre de folds pour la cross-validation

    Returns:
        (rf_model, cv_accuracy)
    """
    if "cluster" not in df.columns:
        raise ValueError("df doit contenir une colonne 'cluster'. Appelez detect_trends() d'abord.")

    labels = df["cluster"].values

    # Pas besoin de cross-val si trop peu de données
    if len(df) < cv * 2:
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(matrix, labels)
        return rf, float("nan")

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    scores = cross_val_score(rf, matrix, labels, cv=cv, scoring="accuracy")
    rf.fit(matrix, labels)  # réentraîne sur tout le corpus

    return rf, float(scores.mean())


def classify_new_posts(
    texts: List[str],
    vectorizer: TfidfVectorizer,
    rf_model: RandomForestClassifier,
    trends_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Classifie de nouveaux posts Reddit avec le Random Forest.

    Returns:
        DataFrame avec colonnes: text, predicted_cluster, topic_label,
                                  confidence (probabilité max)
    """
    from .preprocess import clean_text  # import local pour éviter la circularité

    cleaned = [clean_text(t) for t in texts]
    matrix = vectorizer.transform(cleaned)
    preds = rf_model.predict(matrix)
    proba = rf_model.predict_proba(matrix).max(axis=1)

    # Joindre le label du cluster
    label_map = dict(zip(trends_df["cluster"], trends_df["label"]))

    return pd.DataFrame({
        "text": texts,
        "predicted_cluster": preds,
        "topic_label": [label_map.get(p, f"Cluster {p}") for p in preds],
        "confidence": proba.round(3),
    })


def get_feature_importance(
    rf_model: RandomForestClassifier,
    vectorizer: TfidfVectorizer,
    top_n: int = 15,
) -> pd.DataFrame:
    """Retourne les N features les plus importantes du Random Forest."""
    importances = rf_model.feature_importances_
    features = vectorizer.get_feature_names_out()
    idx = importances.argsort()[-top_n:][::-1]
    return pd.DataFrame({
        "feature": features[idx],
        "importance": importances[idx].round(4),
    })


# ─── Hashtags / Flairs ───────────────────────────────────────────────────────

def extract_hashtags(df: pd.DataFrame) -> pd.DataFrame:
    """Extrait et compte les hashtags du champ 'text'."""
    all_tags = []
    for text in df["text"]:
        all_tags.extend(re.findall(r"#(\w+)", str(text).lower()))
    counts = Counter(all_tags)
    if not counts:
        return pd.DataFrame(columns=["hashtag", "count"])
    return pd.DataFrame(counts.most_common(20), columns=["hashtag", "count"])


def extract_flairs(df: pd.DataFrame) -> pd.DataFrame:
    """Compte les flairs Reddit (équivalent hashtags natifs)."""
    if "flair" not in df.columns:
        return pd.DataFrame(columns=["flair", "count"])
    counts = Counter(f for f in df["flair"] if f and f.strip())
    if not counts:
        return pd.DataFrame(columns=["flair", "count"])
    return pd.DataFrame(counts.most_common(15), columns=["flair", "count"])


# ─── Persistance ─────────────────────────────────────────────────────────────

def save_models(
    vectorizer: TfidfVectorizer,
    kmeans: KMeans,
    rf: RandomForestClassifier,
    path: str = "data/models.pkl",
) -> None:
    import os; os.makedirs("data", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"vectorizer": vectorizer, "kmeans": kmeans, "rf": rf}, f)


def load_models(path: str = "data/models.pkl") -> Optional[Dict]:
    import os
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)