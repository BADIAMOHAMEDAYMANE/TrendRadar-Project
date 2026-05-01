"""
preprocess.py — Nettoyage NLP et vectorisation TF-IDF pour posts Reddit
Utilise spaCy (fr_core_news_md + en_core_web_sm) pour :
  - Détection de langue (langdetect)
  - Lemmatisation
  - Extraction d'entités nommées (NER) : personnes, organisations, lieux

FIX — colonnes NER ('ner_persons', 'ner_orgs', 'ner_locations') TOUJOURS créées,
même si spaCy est absent, pour éviter les KeyError dans app.py et visualize.py.
"""

import re
import warnings
import pandas as pd
import spacy
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import spmatrix
from typing import Tuple, List, Dict

warnings.filterwarnings("ignore")

# ─── Chargement des modèles spaCy ────────────────────────────────────────────

@st.cache_resource
def _load_spacy_models() -> Dict[str, spacy.language.Language]:
    """Charge les modèles spaCy FR et EN (désactive les pipelines inutiles)."""
    models = {}
    # Ordre de priorité pour le chargement des modèles
    fr_candidates = ["fr_core_news_md", "fr_core_news_sm"]
    en_candidates = ["en_core_web_sm", "en_core_web_md"]

    for model_name in fr_candidates:
        try:
            # Utilisation de spacy.load direct pour la compatibilité package
            models["fr"] = spacy.load(model_name, disable=["parser"])
            break
        except (OSError, ImportError):
            continue

    for model_name in en_candidates:
        try:
            models["en"] = spacy.load(model_name, disable=["parser"])
            break
        except (OSError, ImportError):
            continue

    if not models:
        warnings.warn(
            "Aucun modèle spaCy disponible. NER et lemmatisation désactivés. "
            "Assurez-vous que les modèles sont dans requirements.txt"
        )
    return models


# Initialisation sécurisée via le cache Streamlit
_SPACY_MODELS: Dict[str, spacy.language.Language] = _load_spacy_models()

# ─── Détection de langue ──────────────────────────────────────────────────────

def detect_language(text: str) -> str:
    """
    Détecte la langue d'un texte.
    Retourne 'fr', 'en' ou 'unknown'.
    """
    if not text or len(str(text).strip()) < 10:
        return "unknown"
    try:
        from langdetect import detect
        lang = detect(text)
        return lang if lang in ("fr", "en") else "other"
    except Exception:
        pass
    
    # Heuristique de secours
    fr_markers = {"le", "la", "les", "de", "du", "est", "et", "en", "un", "une",
                  "dans", "pour", "que", "qui", "sur", "pas", "je", "vous"}
    words = set(str(text).lower().split())
    if len(words & fr_markers) >= 2:
        return "fr"
    return "en"


# ─── Extraction NER ───────────────────────────────────────────────────────────

def extract_entities(text: str, lang: str = "fr") -> Dict[str, List[str]]:
    """
    Extrait les entités nommées (PER, ORG, LOC/GPE) d'un texte via spaCy.
    Retourne toujours le même dict même si spaCy est absent.
    """
    entities: Dict[str, List[str]] = {"persons": [], "organizations": [], "locations": []}

    if not _SPACY_MODELS:
        return entities

    model_key = lang if lang in _SPACY_MODELS else (
        "fr" if "fr" in _SPACY_MODELS else (
            "en" if "en" in _SPACY_MODELS else None
        )
    )
    if model_key is None:
        return entities

    try:
        nlp = _SPACY_MODELS[model_key]
        doc = nlp(str(text)[:1000])
        for ent in doc.ents:
            label = ent.label_
            value = ent.text.strip()
            if not value:
                continue
            if label in ("PER", "PERSON"):
                entities["persons"].append(value)
            elif label in ("ORG",):
                entities["organizations"].append(value)
            elif label in ("LOC", "GPE", "FAC"):
                entities["locations"].append(value)
    except Exception:
        pass

    return entities


def extract_entities_batch(
    texts: List[str],
    langs: List[str],
) -> List[Dict[str, List[str]]]:
    """
    Extraction NER en batch.
    FIX — retourne toujours une liste de la bonne taille,
    même si spaCy est absent ou plante.
    """
    results: List[Dict[str, List[str]]] = [
        {"persons": [], "organizations": [], "locations": []} for _ in texts
    ]

    if not _SPACY_MODELS:
        return results

    for lang_key in ("fr", "en"):
        if lang_key not in _SPACY_MODELS:
            continue
        indices = [i for i, l in enumerate(langs) if l == lang_key]
        if not indices:
            continue
        nlp = _SPACY_MODELS[lang_key]
        batch_texts = [str(texts[i])[:1000] for i in indices]
        try:
            for idx, doc in zip(indices, nlp.pipe(batch_texts, batch_size=64)):
                for ent in doc.ents:
                    label, value = ent.label_, ent.text.strip()
                    if not value:
                        continue
                    if label in ("PER", "PERSON"):
                        results[idx]["persons"].append(value)
                    elif label in ("ORG",):
                        results[idx]["organizations"].append(value)
                    elif label in ("LOC", "GPE", "FAC"):
                        results[idx]["locations"].append(value)
        except Exception:
            # En cas d'erreur batch, fallback post par post
            for i, idx in enumerate(indices):
                try:
                    doc = nlp(batch_texts[i])
                    for ent in doc.ents:
                        label, value = ent.label_, ent.text.strip()
                        if not value:
                            continue
                        if label in ("PER", "PERSON"):
                            results[idx]["persons"].append(value)
                        elif label in ("ORG",):
                            results[idx]["organizations"].append(value)
                        elif label in ("LOC", "GPE", "FAC"):
                            results[idx]["locations"].append(value)
                except Exception:
                    pass

    return results


# ─── Stopwords étendus ────────────────────────────────────────────────────────

_FR_STOPS: set = set()
_EN_STOPS: set = set()

try:
    import nltk
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords
    _FR_STOPS = set(stopwords.words("french"))
    _EN_STOPS = set(stopwords.words("english"))
except Exception:
    pass

if "fr" in _SPACY_MODELS:
    _FR_STOPS |= _SPACY_MODELS["fr"].Defaults.stop_words
if "en" in _SPACY_MODELS:
    _EN_STOPS |= _SPACY_MODELS["en"].Defaults.stop_words

STOPWORDS = _FR_STOPS | _EN_STOPS | {
    "https", "http", "www", "reddit", "post", "edit", "update",
    "deleted", "removed", "comment", "amp", "nbsp", "quot",
}


# ─── Lemmatisation spaCy ──────────────────────────────────────────────────────

def _lemmatize(text: str, lang: str) -> str:
    """Lemmatise un texte avec le modèle spaCy correspondant."""
    if not _SPACY_MODELS:
        return text

    model_key = lang if lang in _SPACY_MODELS else (
        "fr" if "fr" in _SPACY_MODELS else (
            "en" if "en" in _SPACY_MODELS else None
        )
    )
    if model_key is None:
        return text

    try:
        nlp = _SPACY_MODELS[model_key]
        doc = nlp(str(text)[:2000])
        tokens = [
            t.lemma_.lower()
            for t in doc
            if not t.is_stop
            and not t.is_punct
            and not t.is_space
            and len(t.lemma_) > 2
            and t.lemma_.lower() not in STOPWORDS
        ]
        return " ".join(tokens)
    except Exception:
        return text


# ─── Nettoyage de base ────────────────────────────────────────────────────────

def clean_text(text: str, lemmatize: bool = False, lang: str = "fr") -> str:
    """
    Nettoie un texte Reddit.
    FIX — robuste face aux valeurs None/NaN.
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"u/\w+|r/\w+", "", text)
    text = re.sub(r"@\w+|#\w+", "", text)
    text = re.sub(r"[^a-zA-ZÀ-ÿ\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        return ""

    if lemmatize and _SPACY_MODELS:
        return _lemmatize(text, lang)

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
    extract_ner: bool = True,
    lemmatize: bool = True,
) -> Tuple[pd.DataFrame, TfidfVectorizer, spmatrix]:
    """
    Pipeline NLP complet.
    """
    df = df.copy()

    if "text" not in df.columns:
        raise ValueError("Le DataFrame doit contenir une colonne 'text'.")

    df["text"] = df["text"].fillna("").astype(str)

    # 1. Détection de langue
    df["lang"] = df["text"].apply(detect_language)

    # 2. NER batch
    if extract_ner and _SPACY_MODELS:
        try:
            entities_list = extract_entities_batch(
                df["text"].tolist(),
                df["lang"].tolist(),
            )
            df["ner_persons"]   = [e["persons"]       for e in entities_list]
            df["ner_orgs"]      = [e["organizations"] for e in entities_list]
            df["ner_locations"] = [e["locations"]      for e in entities_list]
        except Exception:
            df["ner_persons"]   = [[] for _ in range(len(df))]
            df["ner_orgs"]      = [[] for _ in range(len(df))]
            df["ner_locations"] = [[] for _ in range(len(df))]
    else:
        df["ner_persons"]   = [[] for _ in range(len(df))]
        df["ner_orgs"]      = [[] for _ in range(len(df))]
        df["ner_locations"] = [[] for _ in range(len(df))]

    # 3. Nettoyage + lemmatisation
    df["clean_text"] = df.apply(
        lambda row: clean_text(row["text"], lemmatize=lemmatize, lang=row["lang"]),
        axis=1,
    )

    df["clean_text"] = df["clean_text"].apply(
        lambda t: t if t.strip() else "post vide"
    )

    df = df[df["clean_text"].str.strip().str.len() > 0].reset_index(drop=True)

    if df.empty:
        raise ValueError("Aucun post utilisable après nettoyage NLP.")

    # 4. TF-IDF
    n_docs = len(df)
    min_df = 2 if n_docs >= 10 else 1

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
        min_df=min_df,
    )
    matrix = vectorizer.fit_transform(df["clean_text"])

    return df, vectorizer, matrix


# ─── Utilitaires ──────────────────────────────────────────────────────────────

def get_top_terms(text: str, vectorizer: TfidfVectorizer, top_n: int = 5) -> List[str]:
    """Retourne les top N termes TF-IDF d'un texte donné."""
    vec = vectorizer.transform([clean_text(str(text))])
    indices = vec.toarray()[0].argsort()[-top_n:][::-1]
    return [vectorizer.get_feature_names_out()[i] for i in indices]


def get_all_locations(df: pd.DataFrame) -> List[str]:
    """Extrait tous les lieux détectés par NER dans le DataFrame."""
    if "ner_locations" not in df.columns:
        return []
    locs = []
    for loc_list in df["ner_locations"]:
        if isinstance(loc_list, list):
            locs.extend(loc_list)
    return locs