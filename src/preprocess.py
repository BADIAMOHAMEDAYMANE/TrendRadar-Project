"""
preprocess.py — Nettoyage NLP et vectorisation TF-IDF pour posts Reddit
"""

import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import spmatrix
from typing import Tuple, List

nltk.download("stopwords", quiet=True)

STOPWORDS = (
    set(stopwords.words("french"))
    | set(stopwords.words("english"))
    | {
        # Bruit typique Reddit
        "https", "http", "www", "reddit", "post", "edit", "update",
        "deleted", "removed", "comment", "amp", "nbsp", "quot",
    }
)


# ─── Nettoyage ────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Supprime URLs, mentions, ponctuation et stopwords.
    Conserve accents français et mots >2 caractères.
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+", "", text)               # URLs
    text = re.sub(r"u/\w+|r/\w+", "", text)                  # u/user  r/sub
    text = re.sub(r"@\w+|#\w+", "", text)                    # mentions / hashtags
    text = re.sub(r"[^a-zA-ZÀ-ÿ\s]", " ", text)              # ponctuation / chiffres
    tokens = [
        w.lower() for w in text.split()
        if w.lower() not in STOPWORDS and len(w) > 2
    ]
    return " ".join(tokens)


# ─── Pipeline complet ─────────────────────────────────────────────────────────

def preprocess(
    df: pd.DataFrame,
    max_features: int = 500,
    ngram_range: Tuple[int, int] = (1, 2),
) -> Tuple[pd.DataFrame, TfidfVectorizer, spmatrix]:
    """
    Nettoie les textes et construit la matrice TF-IDF.

    Args:
        df           : DataFrame avec colonne 'text'
        max_features : nombre max de features TF-IDF
        ngram_range  : plage de n-grams

    Returns:
        (df_clean, vectorizer, tfidf_matrix)
    """
    df = df.copy()
    df["clean_text"] = df["text"].apply(clean_text)

    # Supprimer les posts vides après nettoyage
    df = df[df["clean_text"].str.strip().str.len() > 0].reset_index(drop=True)

    if df.empty:
        raise ValueError("Aucun post utilisable après nettoyage NLP.")

    n_docs = len(df)

    # min_df adaptatif : exige qu'un terme apparaisse dans au moins 2 docs,
    # mais seulement si le corpus est assez grand pour le permettre.
    # En dessous de 10 docs, on accepte tous les termes (min_df=1).
    min_df = 2 if n_docs >= 10 else 1

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,   # atténue les termes très fréquents
        min_df=min_df,
    )
    matrix = vectorizer.fit_transform(df["clean_text"])

    return df, vectorizer, matrix


def get_top_terms(text: str, vectorizer: TfidfVectorizer, top_n: int = 5) -> List[str]:
    """Retourne les top N termes TF-IDF d'un texte donné (pour debug/affichage)."""
    vec = vectorizer.transform([clean_text(text)])
    indices = vec.toarray()[0].argsort()[-top_n:][::-1]
    return [vectorizer.get_feature_names_out()[i] for i in indices]