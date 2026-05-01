"""
collect.py — Collecte Reddit via endpoints JSON publics (sans clé API)
Compatible avec le projet TrendRadar · v2 anti-ban
"""

import os
import time
import random
import requests
import pandas as pd
from datetime import datetime
from typing import Generator, List, Optional, Set

# ─── Configuration ────────────────────────────────────────────────────────────

HEADERS = {
    "User-Agent": "TrendRadar/1.0 by u/ton_username",
    "Accept": "application/json",
}

BASE_URL    = "https://api.reddit.com"
MAX_RETRIES = 3
CACHE_PATH  = "data/reddit_snapshot.parquet"


# ─── Session persistante ──────────────────────────────────────────────────────

def _build_session() -> requests.Session:
    """Crée une session persistante avec les headers globaux."""
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


_SESSION = _build_session()


# ─── Helpers internes ─────────────────────────────────────────────────────────

def _safe_get(url: str, params: dict = None, retries: int = MAX_RETRIES) -> Optional[dict]:
    """GET avec retry + backoff exponentiel."""
    for attempt in range(retries):
        try:
            resp = _SESSION.get(url, params=params, timeout=15)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                wait = 2 ** attempt + random.uniform(0.5, 1.5)
                time.sleep(wait)
            elif resp.status_code in (403, 404):
                return None
            else:
                time.sleep(1)
        except Exception:
            time.sleep(1.5)
    return None


def _post_to_dict(post: dict) -> dict:
    """Convertit un post Reddit brut en dict normalisé."""
    d = post.get("data", {})
    created = d.get("created_utc", 0)
    return {
        "id":           d.get("id", ""),
        "title":        d.get("title", ""),
        "selftext":     d.get("selftext", ""),
        "text":         (d.get("title", "") + " " + d.get("selftext", "")).strip(),
        "subreddit":    d.get("subreddit", ""),
        "score":        int(d.get("score", 0)),
        "upvote_ratio": float(d.get("upvote_ratio", 0.5)),
        "num_comments": int(d.get("num_comments", 0)),
        "flair":        d.get("link_flair_text") or "",
        "url":          d.get("url", ""),
        "permalink":    "https://reddit.com" + d.get("permalink", ""),
        "created_at":   pd.Timestamp(created, unit="s", tz="UTC") if created else pd.NaT,
        "author":       d.get("author", "[deleted]"),
        "is_video":     bool(d.get("is_video", False)),
        "over_18":      bool(d.get("over_18", False)),
    }


# ─── Collecte principale ──────────────────────────────────────────────────────

def fetch_posts(
    subreddits: str,
    query: str,
    limit: int = 150,
    sort: str = "relevance",
    time_filter: str = "week",
) -> pd.DataFrame:
    """
    Recherche des posts Reddit via l'API JSON publique.

    Args:
        subreddits : subreddits séparés par '+' (ex: "france+French")
        query      : mots-clés de recherche
        limit      : nombre max de posts à récupérer
        sort       : "relevance" | "new" | "hot" | "top"
        time_filter: "day" | "week" | "month" | "year" (pour sort=top)

    Returns:
        DataFrame normalisé
    """
    posts    = []
    after    = None
    batch    = min(100, limit)
    seen_ids: Set[str] = set()

    sub_path = subreddits if subreddits else "all"
    url      = f"{BASE_URL}/r/{sub_path}/search.json"

    while len(posts) < limit:
        params = {
            "q":          query,
            "sort":       sort,
            "t":          time_filter,
            "limit":      batch,
            "restrict_sr": "true",
        }
        if after:
            params["after"] = after

        data = _safe_get(url, params)
        if not data:
            break

        children = data.get("data", {}).get("children", [])
        if not children:
            break

        for child in children:
            post = _post_to_dict(child)
            if post["id"] not in seen_ids and post["text"].strip():
                seen_ids.add(post["id"])
                posts.append(post)

        after = data.get("data", {}).get("after")
        if not after or len(posts) >= limit:
            break

        time.sleep(random.uniform(0.8, 1.5))

    df = pd.DataFrame(posts[:limit])
    if not df.empty:
        _save_cache(df)
    return df


def fetch_subreddit_posts(
    subreddit: str,
    sort: str = "hot",
    limit: int = 100,
    time_filter: str = "week",
) -> pd.DataFrame:
    """
    Récupère les posts d'un subreddit spécifique sans query de recherche.

    Args:
        subreddit  : nom du subreddit (sans r/)
        sort       : "hot" | "new" | "top" | "rising"
        limit      : nombre max de posts
        time_filter: utilisé si sort="top"

    Returns:
        DataFrame normalisé
    """
    posts    = []
    after    = None
    batch    = min(100, limit)
    seen_ids: Set[str] = set()

    url = f"{BASE_URL}/r/{subreddit}/{sort}.json"

    while len(posts) < limit:
        params: dict = {"limit": batch}
        if sort == "top":
            params["t"] = time_filter
        if after:
            params["after"] = after

        data = _safe_get(url, params)
        if not data:
            break

        children = data.get("data", {}).get("children", [])
        if not children:
            break

        for child in children:
            post = _post_to_dict(child)
            if post["id"] not in seen_ids and post["text"].strip():
                seen_ids.add(post["id"])
                posts.append(post)

        after = data.get("data", {}).get("after")
        if not after or len(posts) >= limit:
            break

        time.sleep(random.uniform(0.8, 1.5))

    df = pd.DataFrame(posts[:limit])
    if not df.empty:
        _save_cache(df)
    return df


# ─── Streaming (polling) ──────────────────────────────────────────────────────

def stream_posts(
    subreddits: str,
    keywords: Optional[List[str]] = None,
    max_posts: int = 200,
    poll_interval: int = 10,
) -> Generator[dict, None, None]:
    """
    Générateur qui poll Reddit toutes les `poll_interval` secondes
    et yield les nouveaux posts.

    Args:
        subreddits    : subreddits séparés par '+'
        keywords      : filtre optionnel (liste de mots-clés)
        max_posts     : nombre total max avant arrêt
        poll_interval : secondes entre chaque requête
    """
    seen_ids: Set[str] = set()
    count    = 0
    sub_path = subreddits if subreddits else "all"
    url      = f"{BASE_URL}/r/{sub_path}/new.json"

    while count < max_posts:
        data = _safe_get(url, {"limit": 100})
        if not data:
            time.sleep(poll_interval)
            continue

        children = data.get("data", {}).get("children", [])
        for child in children:
            post = _post_to_dict(child)
            if post["id"] in seen_ids or not post["text"].strip():
                continue

            # Filtre optionnel par mots-clés
            if keywords:
                text_lower = post["text"].lower()
                if not any(kw.lower() in text_lower for kw in keywords):
                    continue

            seen_ids.add(post["id"])
            yield post
            count += 1
            if count >= max_posts:
                return

        time.sleep(poll_interval)


# ─── Cache ────────────────────────────────────────────────────────────────────

def _save_cache(df: pd.DataFrame) -> None:
    """Sauvegarde le DataFrame en cache Parquet."""
    try:
        os.makedirs("data", exist_ok=True)
        df.to_parquet(CACHE_PATH, index=False)
    except Exception:
        pass


def load_cached() -> Optional[pd.DataFrame]:
    """
    Charge le cache Parquet si disponible.
    Retourne None si aucun cache n'existe.
    """
    if not os.path.exists(CACHE_PATH):
        return None
    try:
        return pd.read_parquet(CACHE_PATH)
    except Exception:
        return None