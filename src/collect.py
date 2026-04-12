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
    "Referer": "https://www.reddit.com/",
}

BASE_URL      = "https://www.reddit.com"
REQUEST_DELAY = 2.0  # secondes entre requêtes (~30 req/min max)
MAX_RETRIES   = 3    # tentatives par URL


# ─── Helpers internes ─────────────────────────────────────────────────────────

def _build_session() -> requests.Session:
    """Crée une session persistante avec les headers globaux."""
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def _get_with_fallback(
    url_primary: str,
    url_fallback: str,
    params: dict,
    session: requests.Session,
) -> dict:
    """
    Tente url_primary d'abord, puis url_fallback si 403/404 ou réponse vide.
    Gère le rate-limiting (429) avec backoff exponentiel.
    """
    urls_to_try = [url_primary]
    if url_fallback and url_fallback != url_primary:
        urls_to_try.append(url_fallback)

    for url in urls_to_try:
        current_params = dict(params)
        if url == url_fallback:
            current_params.pop("restrict_sr", None)

        for retry in range(MAX_RETRIES):
            try:
                resp = session.get(url, params=current_params, timeout=15)

                if resp.status_code == 429:
                    wait = int(resp.headers.get("Retry-After", 15))
                    time.sleep(wait)
                    continue

                if resp.status_code in (403, 404):
                    time.sleep(REQUEST_DELAY)
                    break

                resp.raise_for_status()
                data = resp.json()

                children = data.get("data", {}).get("children", [])
                if not children and url == url_primary and url_primary != url_fallback:
                    break

                return data

            except requests.exceptions.Timeout:
                time.sleep(2 ** retry)
                continue
            except requests.exceptions.RequestException as e:
                raise RuntimeError(f"Erreur réseau : {e}")

    raise RuntimeError(
        f"Impossible d'obtenir une réponse valide après {MAX_RETRIES} tentatives. "
        f"URL primaire : {url_primary}"
    )


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
    """
    records = []
    after   = None
    session = _build_session()

    if subreddits.lower() == "all":
        url_primary  = f"{BASE_URL}/search.json"
        url_fallback = url_primary
    else:
        url_primary  = f"{BASE_URL}/r/{subreddits}/search.json"
        url_fallback = f"{BASE_URL}/search.json"

    while len(records) < max_results:
        batch_size = min(100, max_results - len(records))
        params = {
            "q":           query.strip(),
            "sort":        sort,
            "t":           time_filter,
            "limit":       batch_size,
            "restrict_sr": "true",
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
    """
    session = _build_session()
    records = []
    after   = None

    sort_fallbacks = [sort]
    if sort != "new":
        sort_fallbacks.append("new")

    url_primary  = f"{BASE_URL}/r/{subreddit}/{sort_fallbacks[0]}.json"
    url_fallback = (
        f"{BASE_URL}/r/{subreddit}/{sort_fallbacks[1]}.json"
        if len(sort_fallbacks) > 1
        else f"{BASE_URL}/r/{subreddit}/new.json"
    )

    while len(records) < limit:
        params = {"limit": min(100, limit - len(records))}
        if after:
            params["after"] = after

        try:
            data = _get_with_fallback(url_primary, url_fallback, params, session)
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


# ─── Streaming par polling (SANS CLÉ API) ────────────────────────────────────

def stream_posts(
    subreddits: str,
    keywords: Optional[List[str]] = None,
    max_posts: int = 100,
    poll_interval: int = 10,  # CORRIGÉ : 10s au lieu de 30s
) -> Generator[dict, None, None]:
    """
    Streaming Reddit par polling sur /new.json — SANS clé API.

    Corrections v2 :
      - poll_interval par défaut réduit à 10s (au lieu de 30s)
      - keywords = None accepte TOUS les posts (plus de filtre silencieux)
      - Si keywords fournis, filtre sur la phrase entière (pas mot par mot)
      - Fallback automatique sur 3 niveaux d'URLs

    Stratégie de fallback automatique (3 niveaux) :
      1. /r/sub1+sub2+.../new.json  — multi-subreddit
      2. /r/{premier_subreddit}/new.json — subreddit unique
      3. /new.json — front page Reddit globale

    Args:
        subreddits   : subreddits à surveiller, ex. "france+MachineLearning"
        keywords     : liste de phrases à filtrer (None = tout accepter).
                       Chaque élément est cherché comme phrase complète (OR logique).
                       Ex: ["AI", "machine learning"] accepte tout post contenant l'un ou l'autre.
        max_posts    : nombre max de posts nouveaux à collecter
        poll_interval: secondes entre chaque poll (min recommandé : 10s)

    Yields:
        dict d'un post Reddit normalisé
    """
    seen_ids:           Set[str] = set()
    collected:          int      = 0
    consecutive_errors: int      = 0
    MAX_CONSECUTIVE_ERRORS       = 5

    session = _build_session()

    # ── Construction de la liste de fallback ──────────────────────────────────
    sub_list   = [s.strip() for s in subreddits.split("+") if s.strip()]
    url_multi  = f"{BASE_URL}/r/{'+'.join(sub_list)}/new.json"
    url_single = f"{BASE_URL}/r/{sub_list[0]}/new.json" if sub_list else None
    url_global = f"{BASE_URL}/new.json"

    candidate_urls = [url_multi]
    if url_single and url_single != url_multi:
        candidate_urls.append(url_single)
    candidate_urls.append(url_global)

    active_url_index = 0

    # ── Normalisation des keywords ────────────────────────────────────────────
    # On conserve les phrases entières en minuscules pour le filtre
    normalized_keywords = None
    if keywords:
        normalized_keywords = [kw.lower().strip() for kw in keywords if kw.strip()]
        if not normalized_keywords:
            normalized_keywords = None

    # ── Boucle de polling ─────────────────────────────────────────────────────
    while collected < max_posts:
        url = candidate_urls[active_url_index]

        try:
            resp = session.get(url, params={"limit": 100}, timeout=15)

            # Rate-limit → attendre le délai indiqué par Reddit
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 30))
                time.sleep(wait)
                continue

            # Blocage ou subreddit introuvable → URL suivante
            if resp.status_code in (403, 404):
                if active_url_index < len(candidate_urls) - 1:
                    active_url_index += 1
                time.sleep(REQUEST_DELAY)
                continue

            resp.raise_for_status()
            data     = resp.json()
            children = data.get("data", {}).get("children", [])

            # Réponse vide → URL suivante
            if not children:
                if active_url_index < len(candidate_urls) - 1:
                    active_url_index += 1
                time.sleep(REQUEST_DELAY)
                continue

            # Succès → réinitialiser le compteur d'erreurs
            consecutive_errors = 0

            # ── Traitement des posts reçus ────────────────────────────────────
            new_this_round = 0
            for child in children:
                p       = child.get("data", {})
                post_id = p.get("id", "")

                if not post_id or post_id in seen_ids:
                    continue
                seen_ids.add(post_id)

                post_dict = _post_to_dict(p)

                # CORRIGÉ : filtre sur phrase entière (OR logique entre keywords)
                # Si normalized_keywords est None → on accepte tout
                if normalized_keywords:
                    text = post_dict["text"].lower()
                    if not any(kw in text for kw in normalized_keywords):
                        continue

                yield post_dict
                collected      += 1
                new_this_round += 1

                if collected >= max_posts:
                    return

            # Adapter le délai : doubler si aucun nouveau post
            sleep_time = poll_interval
            time.sleep(sleep_time)

        except requests.exceptions.Timeout:
            consecutive_errors += 1
            time.sleep(min(2 ** consecutive_errors, 60))

        except requests.exceptions.RequestException:
            consecutive_errors += 1
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                if active_url_index < len(candidate_urls) - 1:
                    active_url_index += 1
                consecutive_errors = 0
            time.sleep(min(2 ** consecutive_errors, 60))

        except Exception:
            time.sleep(poll_interval)
            continue


# ─── Cache ────────────────────────────────────────────────────────────────────

def load_cached(path: str = "data/reddit_snapshot.parquet") -> Optional[pd.DataFrame]:
    """Charge le dernier snapshot sauvegardé."""
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None


# ─── Helpers internes ─────────────────────────────────────────────────────────

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