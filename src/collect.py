"""
collect.py — Collecte Reddit via endpoints JSON publics (sans clé API)
Compatible avec le projet TrendRadar · remplace la version PRAW
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime
from typing import Generator, List, Optional, Set

# Headers réalistes pour éviter le blocage Reddit
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "fr-FR,fr;q=0.9,en-US;q=0.8,en;q=0.7",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}

BASE_URL = "https://www.reddit.com"
REQUEST_DELAY = 1.5   # secondes entre requêtes (respecte ~40 req/min)
MAX_RETRIES = 3       # tentatives avant d'abandonner


# ─── Helpers internes ─────────────────────────────────────────────────────────

def _get_with_fallback(
    url_primary: str,
    url_fallback: str,
    params: dict,
    session: requests.Session,
) -> dict:
    """
    Tente url_primary, puis url_fallback si 404/403.
    Gère le rate-limiting (429) avec backoff.
    Retourne le JSON Reddit ou lève RuntimeError.
    """
    for attempt in range(MAX_RETRIES):
        url = url_primary if attempt == 0 else url_fallback

        # Retire restrict_sr si on passe en fallback global
        current_params = dict(params)
        if url == url_fallback:
            current_params.pop("restrict_sr", None)

        try:
            resp = session.get(url, params=current_params, timeout=15)

            # Rate-limit → attendre et réessayer sur la même URL
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 10))
                time.sleep(wait)
                continue

            # 404 / 403 sur l'URL primaire → basculer sur le fallback
            if resp.status_code in (403, 404) and url == url_primary and url_primary != url_fallback:
                time.sleep(REQUEST_DELAY)
                continue   # prochain tour utilisera url_fallback

            resp.raise_for_status()
            return resp.json()

        except requests.exceptions.Timeout:
            time.sleep(2 ** attempt)
            continue
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Erreur réseau : {e}")

    raise RuntimeError(
        f"Impossible d'obtenir une réponse valide après {MAX_RETRIES} tentatives. "
        f"URL primaire : {url_primary}"
    )


def _build_session() -> requests.Session:
    """Crée une session persistante avec les headers globaux."""
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


# ─── Snapshot batch ───────────────────────────────────────────────────────────

def fetch_posts(
    subreddits: str,
    query: str,
    max_results: int = 200,
    sort: str = "new",
    time_filter: str = "week",
) -> pd.DataFrame:
    """
    Collecte des posts Reddit en mode snapshot via l'API JSON publique.

    Args:
        subreddits : subreddits séparés par '+', ex. "france+programming"
                     ou "all" pour une recherche globale
        query      : mot-clé de recherche
        max_results: nombre max de posts
        sort       : ordre de tri (new, hot, top, relevance)
        time_filter: filtre temporel pour sort="top"

    Returns:
        DataFrame standardisé
    """
    records = []
    after = None
    session = _build_session()

    # URLs : primaire sur les subreddits ciblés, fallback global
    if subreddits.lower() == "all":
        url_primary  = f"{BASE_URL}/search.json"
        url_fallback = url_primary
    else:
        url_primary  = f"{BASE_URL}/r/{subreddits}/search.json"
        url_fallback = f"{BASE_URL}/search.json"   # ← fallback si 404

    while len(records) < max_results:
        batch_size = min(100, max_results - len(records))

        params = {
            "q":            query.strip(),
            "sort":         sort,
            "t":            time_filter,
            "limit":        batch_size,
            "restrict_sr":  "true",   # sera retiré automatiquement sur le fallback
        }
        if after:
            params["after"] = after

        try:
            data = _get_with_fallback(url_primary, url_fallback, params, session)
        except RuntimeError as e:
            raise RuntimeError(f"fetch_posts échoué : {e}")

        children = data.get("data", {}).get("children", [])
        if not children:
            break

        for child in children:
            records.append(_post_to_dict(child.get("data", {})))

        after = data.get("data", {}).get("after")
        if not after:
            break

        time.sleep(REQUEST_DELAY)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    _save_parquet(df)
    return df


def fetch_subreddit_posts(
    subreddit: str,
    sort: str = "new",
    limit: int = 100,
) -> pd.DataFrame:
    """
    Récupère les posts d'un subreddit sans recherche par mot-clé.
    Utile pour les subreddits thématiques (ex: MachineLearning, france).
    """
    session  = _build_session()
    url      = f"{BASE_URL}/r/{subreddit}/{sort}.json"
    fallback = f"{BASE_URL}/{sort}.json"   # front page Reddit si subreddit introuvable
    records  = []
    after    = None

    while len(records) < limit:
        params = {"limit": min(100, limit - len(records))}
        if after:
            params["after"] = after

        try:
            data = _get_with_fallback(url, fallback, params, session)
        except RuntimeError as e:
            raise RuntimeError(f"fetch_subreddit_posts échoué : {e}")

        children = data.get("data", {}).get("children", [])
        if not children:
            break

        for child in children:
            records.append(_post_to_dict(child.get("data", {})))

        after = data.get("data", {}).get("after")
        if not after:
            break

        time.sleep(REQUEST_DELAY)

    df = pd.DataFrame(records) if records else pd.DataFrame()
    if not df.empty:
        _save_parquet(df)
    return df


# ─── Streaming par polling ────────────────────────────────────────────────────

def stream_posts(
    subreddits: str,
    keywords: Optional[List[str]] = None,
    max_posts: int = 100,
    poll_interval: int = 30,
) -> Generator[dict, None, None]:
    """
    Simule un streaming en faisant du polling sur /new.json toutes les N secondes.
    Déduplique les posts via leurs IDs.

    Args:
        subreddits   : subreddits à surveiller, ex. "france+MachineLearning"
        keywords     : filtrer sur ces mots-clés (None = tout accepter)
        max_posts    : s'arrêter après N posts nouveaux
        poll_interval: secondes entre chaque requête

    Yields:
        dict d'un post Reddit normalisé
    """
    seen_ids:  Set[str] = set()
    collected: int      = 0
    session             = _build_session()

    url_primary  = f"{BASE_URL}/r/{subreddits}/new.json"
    url_fallback = f"{BASE_URL}/new.json"

    while collected < max_posts:
        try:
            data     = _get_with_fallback(url_primary, url_fallback, {"limit": 25}, session)
            children = data.get("data", {}).get("children", [])
        except RuntimeError:
            time.sleep(poll_interval)
            continue

        for child in children:
            p       = child.get("data", {})
            post_id = p.get("id", "")

            if post_id in seen_ids:
                continue
            seen_ids.add(post_id)

            post_dict = _post_to_dict(p)

            if keywords:
                text = post_dict["text"].lower()
                if not any(kw.lower() in text for kw in keywords):
                    continue

            yield post_dict
            collected += 1

            if collected >= max_posts:
                return

        time.sleep(poll_interval)


# ─── Cache ────────────────────────────────────────────────────────────────────

def load_cached(path: str = "data/reddit_snapshot.parquet") -> Optional[pd.DataFrame]:
    """Charge le dernier snapshot sauvegardé."""
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None


# ─── Helpers internes ────────────────────────────────────────────────────────

def _save_parquet(df: pd.DataFrame, path: str = "data/reddit_snapshot.parquet") -> None:
    """Sauvegarde le DataFrame en Parquet."""
    os.makedirs("data", exist_ok=True)
    df.to_parquet(path, index=False)


def _post_to_dict(p: dict) -> dict:
    """Normalise un post brut Reddit en dict standardisé."""
    return {
        "id":           p.get("id", ""),
        "title":        p.get("title", ""),
        "text":         (p.get("title", "") + " " + p.get("selftext", "")).strip()[:1000],
        "subreddit":    p.get("subreddit", ""),
        "author":       p.get("author", "[deleted]"),
        "created_at":   datetime.utcfromtimestamp(p.get("created_utc", 0)),
        "score":        p.get("score", 0),
        "upvote_ratio": p.get("upvote_ratio", 0.0),
        "num_comments": p.get("num_comments", 0),
        "url":          "https://reddit.com" + p.get("permalink", ""),
        "flair":        p.get("link_flair_text") or "",
    }