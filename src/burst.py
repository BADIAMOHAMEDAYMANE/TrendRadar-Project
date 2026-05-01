"""
burst.py — Détection de bursts temporels pour TrendRadar
Fenêtres glissantes + score Bt (burst score)

FIXES :
  - Bug Bt = 15_000_000 : score normalisé entre 0 et BT_CAP (100).
    Quand ef=0 (terme nouveau), on retourne BT_NEW_TERM (3.0) au lieu de tf/ε.
  - Bug 6320 alertes sur 150 posts : le nombre de fenêtres est plafonné à
    MAX_WINDOWS. Si les données couvrent plusieurs jours, on augmente
    automatiquement la taille de fenêtre pour rester ≤ MAX_WINDOWS.
  - Bug alertes dupliquées : on agrège les scores par terme (max Bt) plutôt
    que de garder toutes les occurrences fenêtre par fenêtre.
"""

from typing import Dict, List, Optional, Tuple

import pandas as pd

# ─── Constantes ──────────────────────────────────────────────────────────────

EPSILON      = 1e-6
BT_CAP       = 100.0   # score Bt max affiché (évite 15_000_000)
BT_NEW_TERM  = 3.0     # score assigné à un terme jamais vu avant (ef=0)
MAX_WINDOWS  = 200     # nombre max de fenêtres glissantes calculées
MIN_TF       = 2       # un terme doit apparaître ≥ 2 fois dans la fenêtre pour compter


# ─── Score Bt ────────────────────────────────────────────────────────────────

def burst_score(tf: float, ef: float, eps: float = EPSILON) -> float:
    """
    Score de burst Bt = tf / (ef + eps), plafonné à BT_CAP.

    FIX : si ef == 0 (terme nouveau, jamais vu dans les fenêtres passées),
    on retourne BT_NEW_TERM au lieu de tf/eps (qui donnait des millions).
    Un terme qui n'a pas de baseline n'est pas forcément en burst —
    il est simplement nouveau.
    """
    if ef < EPSILON:
        # Terme nouveau : score fixe modéré, pas tf/eps
        return BT_NEW_TERM if tf >= MIN_TF else 0.0
    bt = tf / (ef + eps)
    return min(bt, BT_CAP)


# ─── Détection du type de terme ──────────────────────────────────────────────

def _term_type(term: str) -> str:
    if term.startswith("#"):
        return "hashtag"
    if " " in term:
        return "bigram"
    return "unigram"


# ─── Fenêtre glissante ───────────────────────────────────────────────────────

class SlidingWindowBurst:
    """
    Détecteur de bursts par fenêtre glissante.
    Maintient un historique de fréquences de termes pour calculer la baseline.
    """

    def __init__(self, window_minutes: int = 10, history_windows: int = 6):
        self.window_minutes  = window_minutes
        self.history_windows = history_windows
        self._history: List[Dict[str, int]] = []

    def update(self, term_counts: Dict[str, int]) -> Dict[str, Dict]:
        """
        Met à jour l'historique avec les comptes de la fenêtre courante.
        Retourne {terme: {"bt": score, "tf": tf, "ef": ef}}.
        """
        self._history.append(term_counts)
        if len(self._history) > self.history_windows:
            self._history.pop(0)

        if len(self._history) < 2:
            return {}

        # Baseline = moyenne des fenêtres PASSÉES (hors courante)
        past      = self._history[:-1]
        all_terms = set(t for w in past for t in w)
        baseline: Dict[str, float] = {
            term: sum(w.get(term, 0) for w in past) / len(past)
            for term in all_terms
        }

        current = self._history[-1]
        scores: Dict[str, Dict] = {}
        for term, tf in current.items():
            if tf < MIN_TF:          # ignorer les hapax dans la fenêtre
                continue
            ef = baseline.get(term, 0.0)
            bt = burst_score(float(tf), ef)
            if bt > 0:
                scores[term] = {"bt": bt, "tf": float(tf), "ef": ef}

        return scores


# ─── Détection batch ─────────────────────────────────────────────────────────

def detect_bursts_batch(
    df: pd.DataFrame,
    window_minutes: int = 10,
    threshold: float = 2.5,
    text_col: str = "clean_text",
    time_col: str = "created_at",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Détecte les bursts sur un DataFrame de posts.

    FIX 1 — Nombre de fenêtres plafonné à MAX_WINDOWS :
      Si les données couvrent N jours, la fenêtre est automatiquement
      agrandie pour que N_fenêtres ≤ MAX_WINDOWS.

    FIX 2 — Déduplication par terme :
      On garde le score Bt MAX par terme (pas une ligne par fenêtre × terme).
      Résultat : au plus autant d'alertes que de termes uniques.

    Retourne :
      - alerts_df  : termes dont le score Bt >= threshold (1 ligne / terme)
      - scores_df  : tous les scores (pour scatter TF vs EF)
    """
    if df.empty or text_col not in df.columns:
        return pd.DataFrame(), pd.DataFrame()

    df = df.copy()

    # ── Colonne temporelle ────────────────────────────────────────────────
    if time_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col], utc=True, errors="coerce")
        df = df.sort_values(time_col)
    else:
        df[time_col] = pd.date_range(
            start="2024-01-01", periods=len(df), freq="1min", tz="UTC"
        )

    if df[time_col].isna().all():
        df[time_col] = pd.date_range(
            start="2024-01-01", periods=len(df), freq="1min", tz="UTC"
        )

    t_min = df[time_col].dropna().min()
    t_max = df[time_col].dropna().max()

    if pd.isna(t_min) or pd.isna(t_max):
        return pd.DataFrame(), pd.DataFrame()

    # ── FIX 1 : ajuster la taille de fenêtre pour ≤ MAX_WINDOWS ─────────
    span_minutes = max((t_max - t_min).total_seconds() / 60, 1.0)
    if span_minutes / window_minutes > MAX_WINDOWS:
        window_minutes = int(span_minutes / MAX_WINDOWS) + 1

    window_td = pd.Timedelta(minutes=window_minutes)
    detector  = SlidingWindowBurst(window_minutes=window_minutes)

    # ── Boucle sur les fenêtres ───────────────────────────────────────────
    # best_by_term[term] = meilleure ligne (Bt max) sur toutes les fenêtres
    best_by_term: Dict[str, dict] = {}
    all_scores:   List[dict]      = []

    cursor = t_min
    while cursor <= t_max:
        mask   = (df[time_col] >= cursor) & (df[time_col] < cursor + window_td)
        window = df.loc[mask, text_col].dropna()

        if not window.empty:
            term_counts: Dict[str, int] = {}
            for text in window:
                tokens = str(text).lower().split()
                # unigrams
                for tok in tokens:
                    if len(tok) > 2:
                        term_counts[tok] = term_counts.get(tok, 0) + 1
                # bigrams
                for i in range(len(tokens) - 1):
                    if len(tokens[i]) > 2 and len(tokens[i + 1]) > 2:
                        bg = f"{tokens[i]} {tokens[i+1]}"
                        term_counts[bg] = term_counts.get(bg, 0) + 1
                # hashtags
                for tok in tokens:
                    if tok.startswith("#") and len(tok) > 2:
                        term_counts[tok] = term_counts.get(tok, 0) + 1

            scores = detector.update(term_counts)

            for term, vals in scores.items():
                bt  = vals["bt"]
                tf  = vals["tf"]
                ef  = vals["ef"]
                row = {
                    "window_start": cursor,
                    "window_end":   cursor + window_td,
                    "timestamp":    cursor,
                    "term":         term,
                    "burst_score":  round(bt, 4),
                    "count":        term_counts.get(term, 0),
                    "tf":           round(tf, 4),
                    "ef":           round(ef, 4),
                    "type":         _term_type(term),
                    "is_hashtag":   term.startswith("#"),
                }
                all_scores.append(row)

                # ── FIX 2 : garder seulement le Bt max par terme ────────
                if term not in best_by_term or bt > best_by_term[term]["burst_score"]:
                    best_by_term[term] = row

        cursor += window_td

    # ── Construire scores_df (pour scatter) ──────────────────────────────
    scores_df = pd.DataFrame(all_scores)

    # ── Construire alerts_df : 1 ligne / terme, Bt ≥ threshold ───────────
    alerts_rows = [
        row for row in best_by_term.values()
        if row["burst_score"] >= threshold
    ]
    alerts_df = pd.DataFrame(alerts_rows)

    if not alerts_df.empty:
        alerts_df = (
            alerts_df
            .sort_values("burst_score", ascending=False)
            .reset_index(drop=True)
        )

    return alerts_df, scores_df


# ─── Résumé ──────────────────────────────────────────────────────────────────

def burst_summary(alerts_df: pd.DataFrame) -> Dict:
    """Retourne un résumé des alertes burst."""
    if alerts_df is None or alerts_df.empty:
        return {"total_alerts": 0, "top_term": None, "top_score": 0.0, "hashtag_count": 0}

    top_row       = alerts_df.iloc[0]
    hashtag_count = int(alerts_df.get("is_hashtag", pd.Series([False])).sum())

    return {
        "total_alerts":  len(alerts_df),
        "top_term":      top_row.get("term"),
        "top_score":     round(float(top_row.get("burst_score", 0)), 2),
        "hashtag_count": hashtag_count,
    }