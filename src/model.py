"""
model.py — Pipeline ML pour TrendRadar Reddit

Deux pipelines distincts :

1. Clustering non supervisé (KMeans) + classification (Random Forest)
   → détection de topics, comme avant

2. Prédiction de viralité (XGBoost / LightGBM)
   → prédit si un post/micro-tendance dépasse un seuil viral
      (ex. score + commentaires ≥ N) dans les 12h suivantes
   → features : vitesse initiale, auteurs uniques, ratio engagement,
                diversité de subreddits, score burst, etc.
   → objectif : F1-macro ≥ 0.80
"""

import re
import pickle
import warnings
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple, Dict
from collections import Counter
from scipy.sparse import spmatrix, hstack
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, classification_report

warnings.filterwarnings("ignore")

# Import optionnel XGBoost / LightGBM avec fallback GradientBoosting
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    _HAS_LGBM = True
except ImportError:
    _HAS_LGBM = False


# ─── Étiquettes de clusters ──────────────────────────────────────────────────

def _label_cluster(keywords: List[str]) -> str:
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
    """
    n_clusters = min(n_clusters, len(df))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df = df.copy()
    df["cluster"] = kmeans.fit_predict(matrix)

    feature_names = vectorizer.get_feature_names_out()
    trends = []

    for cid in range(n_clusters):
        center  = kmeans.cluster_centers_[cid]
        top_idx = center.argsort()[-10:][::-1]
        keywords = [feature_names[i] for i in top_idx]

        cluster_df = df[df["cluster"] == cid]
        engagement = round(
            cluster_df["score"].sum() + cluster_df["num_comments"].sum() * 1.5, 0
        )
        trends.append({
            "cluster":          cid,
            "label":            _label_cluster(keywords),
            "keywords":         ", ".join(keywords[:6]),
            "post_count":       len(cluster_df),
            "avg_score":        round(cluster_df["score"].mean(), 1),
            "avg_comments":     round(cluster_df["num_comments"].mean(), 1),
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
    """Entraîne un Random Forest sur les labels KMeans."""
    if "cluster" not in df.columns:
        raise ValueError("df doit contenir 'cluster'. Appelez detect_trends() d'abord.")

    labels = df["cluster"].values
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
    rf.fit(matrix, labels)
    return rf, float(scores.mean())


def classify_new_posts(
    texts: List[str],
    vectorizer: TfidfVectorizer,
    rf_model: RandomForestClassifier,
    trends_df: pd.DataFrame,
) -> pd.DataFrame:
    """Classifie de nouveaux posts Reddit avec le Random Forest."""
    try:
        from src.preprocess import clean_text
    except ImportError:
        from preprocess import clean_text

    cleaned = [clean_text(t) for t in texts]
    matrix  = vectorizer.transform(cleaned)
    preds   = rf_model.predict(matrix)
    proba   = rf_model.predict_proba(matrix).max(axis=1)
    label_map = dict(zip(trends_df["cluster"], trends_df["label"]))
    return pd.DataFrame({
        "text":              texts,
        "predicted_cluster": preds,
        "topic_label":       [label_map.get(p, f"Cluster {p}") for p in preds],
        "confidence":        proba.round(3),
    })


def get_feature_importance(
    rf_model: RandomForestClassifier,
    vectorizer: TfidfVectorizer,
    top_n: int = 15,
) -> pd.DataFrame:
    importances = rf_model.feature_importances_
    features    = vectorizer.get_feature_names_out()
    idx = importances.argsort()[-top_n:][::-1]
    return pd.DataFrame({
        "feature":    features[idx],
        "importance": importances[idx].round(4),
    })


# ─── Features de viralité ────────────────────────────────────────────────────

# FIX 1 — Seuil par défaut abaissé de 500 → 50 pour mieux correspondre
# aux données réelles Reddit (posts récents à faible engagement)
VIRAL_THRESHOLD = 50

# Liste canonique des features — DOIT être identique entre train et predict
_VIRALITY_FEATURE_COLS = [
    "velocity", "comment_ratio", "upvote_ratio",
    "flair_encoded", "burst_score_feat",
    "hour_of_day", "day_of_week",
    "sub_diversity", "title_length",
    "score", "num_comments",
]


def build_virality_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construit les features utilisées par le classifieur de viralité.

    FIX — retourne TOUJOURS les mêmes colonnes dans le même ordre
    (_VIRALITY_FEATURE_COLS), en remplissant par 0 les colonnes manquantes.
    Cela évite les erreurs de dimension entre train et predict.
    """
    df = df.copy()
    now = pd.Timestamp.utcnow().tz_localize(None)

    # created_at → datetime naïf UTC
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], utc=True, errors="coerce")
        if hasattr(df["created_at"].dtype, "tz") and df["created_at"].dtype.tz is not None:
            df["created_at"] = df["created_at"].dt.tz_localize(None)
    else:
        df["created_at"] = now

    df["age_hours"] = (now - df["created_at"]).dt.total_seconds().fillna(3600) / 3600
    df["age_hours"] = df["age_hours"].clip(lower=0.1)

    # S'assurer que score et num_comments existent
    df["score"]        = pd.to_numeric(df.get("score", 0),        errors="coerce").fillna(0)
    df["num_comments"] = pd.to_numeric(df.get("num_comments", 0), errors="coerce").fillna(0)

    engagement = df["score"] + df["num_comments"] * 1.5
    df["velocity"]      = (engagement / df["age_hours"]).round(4)
    df["comment_ratio"] = (df["num_comments"] / df["score"].clip(lower=1)).round(4)

    if "upvote_ratio" in df.columns:
        df["upvote_ratio"] = pd.to_numeric(df["upvote_ratio"], errors="coerce").fillna(0.5)
    else:
        df["upvote_ratio"] = 0.5

    df["flair_encoded"] = (
        df["flair"].fillna("").str.strip().str.len() > 0
    ).astype(int) if "flair" in df.columns else 0

    df["burst_score_feat"] = (
        pd.to_numeric(df["burst_score"], errors="coerce").fillna(0)
    ) if "burst_score" in df.columns else 0.0

    df["hour_of_day"] = df["created_at"].dt.hour.fillna(12).astype(int)
    df["day_of_week"] = df["created_at"].dt.dayofweek.fillna(0).astype(int)

    # FIX 2 — sub_diversity : nombre de posts par subreddit (valeur par ligne),
    # et non plus nunique() qui retournait un scalaire identique pour toutes les lignes.
    if "subreddit" in df.columns:
        df["sub_diversity"] = df.groupby("subreddit")["subreddit"].transform("count")
    else:
        df["sub_diversity"] = 1

    df["title_length"] = df["title"].fillna("").apply(
        lambda t: len(str(t).split())
    ) if "title" in df.columns else 5

    # Retourner exactement _VIRALITY_FEATURE_COLS, avec 0 si absent
    result = pd.DataFrame(index=df.index)
    for col in _VIRALITY_FEATURE_COLS:
        if col in df.columns:
            result[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        else:
            result[col] = 0.0

    return result


def build_virality_labels(
    df: pd.DataFrame,
    threshold: int = VIRAL_THRESHOLD,
) -> pd.Series:
    """
    Construit les labels binaires de viralité.
    1 = viral (score + num_comments × 1.5 ≥ threshold)
    0 = non viral

    Stratégie de fallback en cascade (du plus précis au plus robuste) :
      1. Seuil absolu demandé
      2. Seuil P75 adaptatif (si une seule classe avec seuil absolu)
      3. Seuil médiane stricte (engagement > médiane, pas >=)
      4. Fallback par rang (top 50% triés) — fonctionne même si
         TOUTES les valeurs d'engagement sont identiques (ex: tous à 94)
    """
    score        = pd.to_numeric(df.get("score",        0), errors="coerce").fillna(0)
    num_comments = pd.to_numeric(df.get("num_comments", 0), errors="coerce").fillna(0)
    engagement   = score + num_comments * 1.5

    def _two_classes(s: pd.Series) -> bool:
        return s.nunique() >= 2

    # Étape 1 — seuil absolu
    labels = (engagement >= threshold).astype(int)
    if _two_classes(labels):
        return labels

    # Étape 2 — P75 adaptatif
    p75 = engagement.quantile(0.75)
    labels = (engagement >= p75).astype(int)
    if _two_classes(labels):
        return labels

    # Étape 3 — médiane stricte (> et non >=, pour exclure la valeur médiane)
    median = engagement.median()
    labels = (engagement > median).astype(int)
    if _two_classes(labels):
        return labels

    # Étape 4 — fallback par rang : top 50% par ordre de tri
    # Garantit exactement 2 classes même si toutes les valeurs sont identiques.
    # On utilise rank(method='first') pour briser les égalités de façon déterministe.
    n      = len(engagement)
    cutoff = n // 2  # les n//2 posts les mieux classés = "viral"
    ranks  = engagement.rank(method="first", ascending=False)
    labels = (ranks <= cutoff).astype(int)
    return labels


# ─── Classifieur de viralité ─────────────────────────────────────────────────

def _best_booster() -> object:
    """Retourne le meilleur classifieur disponible (XGB > LGBM > GB)."""
    if _HAS_XGB:
        return XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
    if _HAS_LGBM:
        return LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
    # Fallback scikit-learn
    return GradientBoostingClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42,
    )


def train_virality_model(
    df: pd.DataFrame,
    threshold: int = VIRAL_THRESHOLD,
    cv: int = 5,
) -> Tuple[object, StandardScaler, float, str]:
    """
    Entraîne le classifieur de viralité (XGBoost / LightGBM / GBM).

    FIX — build_virality_labels() gère maintenant le fallback percentile
    en amont, donc cette fonction ne lèvera ValueError que si les données
    sont vraiment vides ou dégénérées (engagement totalement nul partout).

    Args:
        df        : DataFrame avec features et colonnes score/num_comments
        threshold : seuil engagement pour label viral (utilisé en priorité)
        cv        : folds de cross-validation (StratifiedKFold)

    Returns:
        (model, scaler, f1_macro_cv, model_name)
    """
    if df is None or df.empty:
        raise ValueError("DataFrame vide — impossible d'entraîner le modèle de viralité.")

    features = build_virality_features(df)
    # FIX — build_virality_labels intègre le fallback percentile
    labels   = build_virality_labels(df, threshold)

    n_viral  = int(labels.sum())
    n_normal = int((1 - labels).sum())

    # Après tous les fallbacks, si toujours une seule classe
    # → données vraiment dégénérées (df avec 1 seul post, ou tous identiques ET n=1)
    if n_viral == 0 or n_normal == 0:
        raise ValueError(
            f"Impossible de créer 2 classes même avec fallback par rang "
            f"(viral={n_viral}, normal={n_normal}, n_posts={len(df)}). "
            f"Le DataFrame doit contenir au moins 2 posts."
        )

    # Minimum 2 exemples par classe pour la CV
    min_class    = min(n_viral, n_normal)
    effective_cv = min(cv, min_class)
    if effective_cv < 2:
        effective_cv = 0  # pas de CV si trop peu d'exemples

    X = features.values.astype(float)
    y = labels.values

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model      = _best_booster()
    model_name = type(model).__name__

    if effective_cv >= 2 and len(df) >= effective_cv * 3:
        skf = StratifiedKFold(n_splits=effective_cv, shuffle=True, random_state=42)
        try:
            scores = cross_val_score(model, X_scaled, y, cv=skf, scoring="f1_macro")
            f1_cv  = float(scores.mean())
        except Exception:
            f1_cv = float("nan")
    else:
        f1_cv = float("nan")

    model.fit(X_scaled, y)
    return model, scaler, f1_cv, model_name


def predict_virality(
    df: pd.DataFrame,
    model: object,
    scaler: StandardScaler,
    threshold: int = VIRAL_THRESHOLD,
) -> pd.DataFrame:
    """
    Prédit la probabilité virale de chaque post.

    FIX — utilise build_virality_features() qui garantit _VIRALITY_FEATURE_COLS
    dans le bon ordre, évitant les erreurs de dimension entre train et predict.

    Returns:
        DataFrame avec colonnes 'viral_prob', 'viral_pred', 'viral_label'
    """
    features = build_virality_features(df)

    # S'assurer que les features correspondent exactement à ce que le scaler attend
    X = features[_VIRALITY_FEATURE_COLS].values.astype(float)

    # Gérer le cas où le scaler a été entraîné sur moins de features
    try:
        X_scaled = scaler.transform(X)
    except ValueError:
        X_scaled = scaler.fit_transform(X)

    try:
        proba = model.predict_proba(X_scaled)
        viral_prob = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
    except Exception:
        viral_prob = np.zeros(len(df))

    viral_pred = (viral_prob >= 0.5).astype(int)

    result = df.copy()
    result["viral_prob"]  = viral_prob.round(3)
    result["viral_pred"]  = viral_pred
    result["viral_label"] = pd.Series(viral_pred).map({1: "🔥 Viral", 0: "Normal"}).values
    return result


def evaluate_virality_model(
    model: object,
    scaler: StandardScaler,
    df_test: pd.DataFrame,
    threshold: int = VIRAL_THRESHOLD,
) -> Dict:
    """
    Évalue le modèle sur un jeu de test et retourne les métriques.
    """
    features = build_virality_features(df_test)
    labels   = build_virality_labels(df_test, threshold)
    X = features[_VIRALITY_FEATURE_COLS].values.astype(float)
    X_scaled = scaler.transform(X)

    preds  = model.predict(X_scaled)
    report = classification_report(labels, preds, output_dict=True, zero_division=0)

    return {
        "f1_macro":       round(report.get("macro avg", {}).get("f1-score", 0), 3),
        "f1_viral":       round(report.get("1", {}).get("f1-score", 0), 3),
        "f1_normal":      round(report.get("0", {}).get("f1-score", 0), 3),
        "support_viral":  int(report.get("1", {}).get("support", 0)),
        "support_normal": int(report.get("0", {}).get("support", 0)),
        "accuracy":       round(report.get("accuracy", 0), 3),
    }


def get_virality_feature_importance(
    model: object,
    df: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """Retourne les features les plus importantes du classifieur viral."""
    feature_names = _VIRALITY_FEATURE_COLS
    importances   = None

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])

    if importances is None:
        return pd.DataFrame(columns=["feature", "importance"])

    # Protéger contre les dimensions incohérentes
    n = min(len(importances), len(feature_names))
    importances  = importances[:n]
    feature_names = feature_names[:n]

    idx = importances.argsort()[-top_n:][::-1]
    return pd.DataFrame({
        "feature":    [feature_names[i] for i in idx],
        "importance": [round(float(importances[i]), 4) for i in idx],
    })


# ─── Hashtags / Flairs ───────────────────────────────────────────────────────

def extract_hashtags(df: pd.DataFrame) -> pd.DataFrame:
    if "text" not in df.columns:
        return pd.DataFrame(columns=["hashtag", "count"])
    all_tags = []
    for text in df["text"]:
        all_tags.extend(re.findall(r"#(\w+)", str(text).lower()))
    counts = Counter(all_tags)
    if not counts:
        return pd.DataFrame(columns=["hashtag", "count"])
    return pd.DataFrame(counts.most_common(20), columns=["hashtag", "count"])


def extract_flairs(df: pd.DataFrame) -> pd.DataFrame:
    if "flair" not in df.columns:
        return pd.DataFrame(columns=["flair", "count"])
    counts = Counter(f for f in df["flair"] if f and str(f).strip())
    if not counts:
        return pd.DataFrame(columns=["flair", "count"])
    return pd.DataFrame(counts.most_common(15), columns=["flair", "count"])


# ─── Persistance ─────────────────────────────────────────────────────────────

def save_models(
    vectorizer: TfidfVectorizer,
    kmeans: KMeans,
    rf: RandomForestClassifier,
    virality_model: Optional[object] = None,
    virality_scaler: Optional[StandardScaler] = None,
    path: str = "data/models.pkl",
) -> None:
    import os; os.makedirs("data", exist_ok=True)
    payload = {
        "vectorizer":      vectorizer,
        "kmeans":          kmeans,
        "rf":              rf,
        "virality_model":  virality_model,
        "virality_scaler": virality_scaler,
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)


def load_models(path: str = "data/models.pkl") -> Optional[Dict]:
    import os
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None