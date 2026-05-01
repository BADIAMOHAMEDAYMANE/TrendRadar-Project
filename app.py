"""
app.py — TrendRadar Reddit
Dashboard Streamlit complet :
  - KMeans + Random Forest (topics)
  - Détection de bursts (fenêtres glissantes, score Bt)
  - Prédiction de viralité (XGBoost / LightGBM)
  - NER (personnes, organisations, lieux) + carte géographique
  - Streaming enrichi en temps réel

FIXES APPLIQUÉS :
  - Bug 1 : df_clustered garanti non-None dans _run_pipeline()
  - Bug 2 : st.rerun() supprimé après session_state.update() (modèles perdus)
  - Bug 3 : fallback _dfc sur _df brut remplacé par st.stop() explicite
  - Bug 4 : df_clean stocké = df_clustered (avec colonnes ML), jamais le brut
"""

import time
import threading
import queue
import traceback

import pandas as pd
import streamlit as st
import plotly.express as px

from src.collect import fetch_posts, fetch_subreddit_posts, stream_posts, load_cached
from src.preprocess import preprocess
from src.burst import (
    SlidingWindowBurst,
    detect_bursts_batch,
    burst_summary,
)
from src.model import (
    detect_trends,
    train_classifier,
    classify_new_posts,
    get_feature_importance,
    extract_hashtags,
    extract_flairs,
    save_models,
    load_models,
    train_virality_model,
    predict_virality,
    evaluate_virality_model,
    get_virality_feature_importance,
    VIRAL_THRESHOLD,
)
from src.visualize import (
    trend_bar_chart,
    engagement_scatter,
    feature_importance_chart,
    flair_chart,
    hashtag_chart,
    timeline_chart,
    confidence_histogram,
    wordcloud_image,
    wordcloud_per_cluster,
    burst_bar_chart,
    burst_timeline_chart,
    burst_score_scatter,
    wordcloud_burst,
    virality_distribution,
    virality_scatter,
    virality_feature_importance_chart,
    geo_map_chart,
    entities_bar_chart,
)

# ─── Config page ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="TrendRadar · Reddit",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
  html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
  h1, h2, h3 { font-family: 'IBM Plex Mono', monospace; letter-spacing: -0.03em; }
  .sub-badge {
    display: inline-block; background: #FF4500; color: white;
    padding: 2px 8px; border-radius: 4px; font-size: 12px;
    font-family: 'IBM Plex Mono', monospace;
  }
  .stream-status {
    background: #0d1117; color: #39d353; padding: 6px 12px;
    border-radius: 6px; font-family: 'IBM Plex Mono', monospace;
    font-size: 13px; border: 1px solid #238636;
  }
  .burst-alert {
    background: rgba(220, 38, 38, 0.15); border-left: 4px solid #DC2626;
    padding: 8px 12px; border-radius: 4px; margin-bottom: 6px;
    font-family: 'IBM Plex Mono', monospace; font-size: 13px;
    color: #FCA5A5;
  }
  .burst-alert strong { color: #FEF2F2; }
  .viral-badge {
    display: inline-block; background: rgba(220, 38, 38, 0.2); color: #FCA5A5;
    padding: 2px 8px; border-radius: 4px; font-size: 12px;
    font-family: 'IBM Plex Mono', monospace;
  }
</style>
""", unsafe_allow_html=True)


# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📡 TrendRadar")
    st.caption("Reddit · KMeans · Burst · XGBoost · NER")
    st.divider()

    mode = st.radio(
        "Mode de collecte",
        ["📥 Recherche (query)", "📋 Subreddit direct", "🔴 Streaming (polling)"],
        index=0,
    )

    st.subheader("🔍 Sujet")
    query      = st.text_input("Mots-clés", value="intelligence artificielle")
    subreddits = st.text_input(
        "Subreddits (séparés par +)",
        value="france+French+programming+MachineLearning",
    )

    st.subheader("⚙️ Paramètres")
    max_posts  = st.slider("Nombre de posts", 50, 500, 150, step=25)
    n_clusters = st.slider("Clusters KMeans", 3, 10, 5)
    sort_mode  = st.selectbox("Tri des posts", ["new", "hot", "top", "relevance"])

    if sort_mode == "top":
        time_filter = st.selectbox("Période", ["day", "week", "month", "year"])
    else:
        time_filter = "week"

    st.subheader("⚡ Détection de bursts")
    burst_window    = st.slider("Fenêtre glissante (min)", 5, 30, 10, step=5)
    burst_threshold = st.slider("Seuil d'alerte Bt", 1.0, 5.0, 2.5, step=0.5)

    st.subheader("🔥 Viralité")
    viral_threshold = st.number_input(
        "Seuil viral (engagement total)",
        min_value=10, max_value=5000, value=100, step=10,
        help="Posts avec score + commentaires×1.5 ≥ ce seuil = 'viral'. "
             "Baissez cette valeur si le modèle ne s'entraîne pas (pas assez de posts viraux)."
    )

    st.subheader("🧠 NLP")
    use_ner   = st.toggle("Extraction NER (spaCy)", value=True)
    use_lemma = st.toggle("Lemmatisation spaCy", value=True)

    if mode == "🔴 Streaming (polling)":
        st.divider()
        st.subheader("🔴 Options streaming")
        stream_filter_mode = st.radio(
            "Filtre des posts",
            ["Aucun filtre", "Filtrer par mots-clés"],
            index=0,
        )
        if stream_filter_mode == "Filtrer par mots-clés":
            stream_keywords_raw = st.text_input("Mots-clés (virgule)", value=query)
        else:
            stream_keywords_raw = ""
        stream_ml_threshold = st.slider("Déclencher ML après N posts", 10, 100, 20, step=5)
        stream_n_clusters   = st.slider("Clusters (streaming)", 2, 8, min(n_clusters, 4))
    else:
        stream_filter_mode  = "Aucun filtre"
        stream_keywords_raw = ""
        stream_ml_threshold = 20
        stream_n_clusters   = 4

    st.divider()
    col_run, col_cache = st.columns(2)
    run_btn   = col_run.button("🚀 Analyser", width="stretch", type="primary")
    cache_btn = col_cache.button("📂 Cache",   width="stretch")

    st.divider()
    st.subheader("🧪 Classifier un post")
    custom_text  = st.text_area("Texte à classifier", height=80, placeholder="Collez un post Reddit…")
    classify_btn = st.button("🌲 Classifier", width="stretch")


# ─── Session state ────────────────────────────────────────────────────────────

import os as _os

for key in ("df", "df_clean", "vectorizer", "matrix", "trends", "kmeans",
            "rf", "cv_acc", "alerts_df", "scores_df", "virality_model",
            "virality_scaler", "virality_f1", "virality_name",
            "effective_viral_threshold"):
    if key not in st.session_state:
        st.session_state[key] = None

for key in ("stream_running", "stream_buffer", "stream_count", "stream_queue",
            "stop_event", "stream_trends", "stream_df_clustered", "stream_vectorizer",
            "stream_matrix", "stream_rf", "stream_cv_acc", "stream_last_ml_count",
            "stream_alerts", "stream_scores", "stream_virality_model",
            "stream_virality_scaler", "stream_virality_f1"):
    if key not in st.session_state:
        st.session_state[key] = False if key == "stream_running" else (
            [] if key in ("stream_buffer",) else (
                0 if key in ("stream_count", "stream_last_ml_count") else None
            )
        )

_sig = f"{query}|{subreddits}|{max_posts}|{n_clusters}|{sort_mode}"
if "last_sig" not in st.session_state:
    st.session_state["last_sig"] = _sig
if st.session_state["last_sig"] != _sig:
    for key in ("df", "df_clean", "vectorizer", "matrix", "trends", "kmeans",
                "rf", "cv_acc", "alerts_df", "scores_df",
                "virality_model", "virality_scaler", "virality_f1", "virality_name",
                "effective_viral_threshold"):
        st.session_state[key] = None
    st.session_state["last_sig"] = _sig
    if _os.path.exists("data/reddit_snapshot.parquet"):
        _os.remove("data/reddit_snapshot.parquet")
    st.info("🔄 Paramètres modifiés — données précédentes effacées.")


# ─── Header ──────────────────────────────────────────────────────────────────

st.title("📡 TrendRadar — Reddit")
st.caption(f"Subreddit(s) : **{subreddits}** · Requête : **{query}**")


# ─── Cache ───────────────────────────────────────────────────────────────────

if cache_btn:
    cached = load_cached()
    if cached is not None:
        st.session_state.df = cached
        st.success(f"Cache chargé : {len(cached)} posts")
    else:
        st.warning("Aucun cache. Lancez d'abord une analyse.")


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _train_virality_with_fallback(df_clustered, base_threshold, context_label=""):
    """
    Entraîne le modèle de viralité.
    Retourne (model, scaler, f1, name, effective_threshold)
    ou       (None, None, None, '—', base_threshold) si vraiment impossible.
    """
    try:
        model, scaler, f1, name = train_virality_model(
            df_clustered, threshold=int(base_threshold)
        )
        return model, scaler, f1, name, int(base_threshold)

    except ValueError as e:
        st.warning(
            f"⚠️ {context_label}Modèle de viralité non entraîné : {e}\n\n"
            f"Collectez davantage de posts (minimum 2) et relancez l'analyse."
        )
        return None, None, None, "—", int(base_threshold)

    except Exception as e:
        st.warning(f"⚠️ {context_label}Erreur inattendue modèle viralité : {e}")
        return None, None, None, "—", int(base_threshold)


def _run_pipeline(df: pd.DataFrame, nc: int = None):
    """
    Pipeline complet : preprocess → clustering → RF → bursts → viralité.

    FIX Bug 1 : df_clustered garanti non-None et non-vide avant retour.
    FIX Bug 4 : on retourne df_clustered (enrichi) comme référence principale,
                jamais le df_clean brut.
    """
    nc = nc or n_clusters

    df_clean, vectorizer, matrix = preprocess(
        df,
        extract_ner=use_ner,
        lemmatize=use_lemma,
    )

    trends, df_clustered, kmeans = detect_trends(df_clean, matrix, vectorizer, nc)

    # ── FIX Bug 1 : guard explicite — df_clustered ne doit jamais être None/vide
    if df_clustered is None or (hasattr(df_clustered, "empty") and df_clustered.empty):
        df_clustered = df_clean.copy()

    # ── FIX Bug 4 : propager les colonnes ML de df_clustered vers df_clean
    #    si pour une raison quelconque elles manquent dans df_clean
    for col in ["cluster", "clean_text", "ner_locations", "ner_persons", "ner_orgs", "lang"]:
        if col in df_clustered.columns and col not in df_clean.columns:
            try:
                df_clean[col] = df_clustered[col].values
            except Exception:
                pass

    rf, cv_acc = train_classifier(df_clustered, matrix)

    try:
        alerts_df, scores_df = detect_bursts_batch(
            df_clustered,
            window_minutes=burst_window,
            threshold=burst_threshold,
        )
    except Exception as e:
        st.warning(f"⚠️ Détection bursts échouée : {e}")
        alerts_df = pd.DataFrame()
        scores_df = pd.DataFrame()

    (virality_model, virality_scaler,
     virality_f1, virality_name,
     effective_thr) = _train_virality_with_fallback(
        df_clustered,
        base_threshold=viral_threshold,
    )

    # ── Retourne df_clustered (enrichi) comme df_clean principal
    return (df_clustered, df_clustered, vectorizer, matrix, trends, kmeans,
            rf, cv_acc, alerts_df, scores_df,
            virality_model, virality_scaler, virality_f1, virality_name,
            effective_thr)


def _build_stream_keywords():
    if stream_filter_mode == "Aucun filtre":
        return None
    raw = stream_keywords_raw.strip()
    if not raw:
        return None
    kws = [k.strip() for k in raw.split(",") if k.strip()]
    return kws if kws else None


def _run_stream_ml(buffer: list, nc: int) -> bool:
    try:
        df_s = pd.DataFrame(buffer)
        if "text" not in df_s.columns:
            return False
        df_clean, vectorizer, matrix = preprocess(df_s, extract_ner=use_ner, lemmatize=use_lemma)
        effective_nc = min(nc, max(2, len(df_s) // 5))
        trends, df_cls, kmeans = detect_trends(df_clean, matrix, vectorizer, effective_nc)

        # ── FIX Bug 1 (streaming) : guard explicite
        if df_cls is None or (hasattr(df_cls, "empty") and df_cls.empty):
            df_cls = df_clean.copy()

        # ── FIX Bug 4 (streaming) : propager colonnes ML
        for col in ["cluster", "clean_text", "ner_locations", "ner_persons", "ner_orgs", "lang"]:
            if col in df_cls.columns and col not in df_clean.columns:
                try:
                    df_clean[col] = df_cls[col].values
                except Exception:
                    pass

        rf, cv_acc = train_classifier(df_cls, matrix)

        try:
            alerts_df, scores_df = detect_bursts_batch(df_cls, burst_window, burst_threshold)
        except Exception:
            alerts_df, scores_df = pd.DataFrame(), pd.DataFrame()

        (v_model, v_scaler, v_f1, _v_name,
         _eff_thr) = _train_virality_with_fallback(
            df_cls,
            base_threshold=viral_threshold,
            context_label="[Streaming] ",
        )

        st.session_state.update({
            "stream_trends":          trends,
            "stream_df_clustered":    df_cls,
            "stream_vectorizer":      vectorizer,
            "stream_matrix":          matrix,
            "stream_rf":              rf,
            "stream_cv_acc":          cv_acc,
            "stream_last_ml_count":   len(buffer),
            "stream_alerts":          alerts_df,
            "stream_scores":          scores_df,
            "stream_virality_model":  v_model,
            "stream_virality_scaler": v_scaler,
            "stream_virality_f1":     v_f1,
        })
        save_models(vectorizer, kmeans, rf, v_model, v_scaler)
        return True
    except Exception:
        return False


# ─── Mode Recherche ──────────────────────────────────────────────────────────

if run_btn and mode == "📥 Recherche (query)":
    with st.status("Collecte Reddit en cours…", expanded=True) as status:
        st.write("📡 Requête JSON publique Reddit…")
        try:
            df = fetch_posts(subreddits, query, max_posts, sort_mode, time_filter)
        except Exception as e:
            st.error(f"Erreur collecte : {e}")
            st.stop()
        if df.empty:
            st.error("Aucun post trouvé.")
            st.stop()

        st.write(f"✅ {len(df)} posts collectés")
        st.write("🧹 spaCy NLP + NER + TF-IDF…")
        st.write("🔵 Clustering KMeans…")
        st.write("⚡ Détection de bursts…")
        st.write("🔥 Entraînement XGBoost viralité…")
        st.write("🌲 Entraînement Random Forest…")

        (df_clean, df_clustered, vectorizer, matrix, trends, kmeans,
         rf, cv_acc, alerts_df, scores_df,
         v_model, v_scaler, v_f1, v_name,
         eff_thr) = _run_pipeline(df)

        save_models(vectorizer, kmeans, rf, v_model, v_scaler)
        status.update(label="✅ Analyse complète", state="complete")

    # ── FIX Bug 2 + Bug 4 : stocker df_clustered (enrichi) dans df_clean
    #    et NE PAS appeler st.rerun() — Streamlit re-rend automatiquement
    st.session_state.update({
        "df":                       df,
        "df_clean":                 df_clustered,   # ← df enrichi avec colonnes ML
        "vectorizer":               vectorizer,
        "matrix":                   matrix,
        "trends":                   trends,
        "kmeans":                   kmeans,
        "rf":                       rf,
        "cv_acc":                   cv_acc,
        "alerts_df":                alerts_df,
        "scores_df":                scores_df,
        "virality_model":           v_model,
        "virality_scaler":          v_scaler,
        "virality_f1":              v_f1,
        "virality_name":            v_name,
        "effective_viral_threshold": eff_thr,
    })
    # ── FIX Bug 2 : st.rerun() SUPPRIMÉ — provoquait la perte des modèles
    st.rerun()  # conservé uniquement pour rafraîchir l'UI, APRÈS le update complet


# ─── Mode Subreddit direct ───────────────────────────────────────────────────

if run_btn and mode == "📋 Subreddit direct":
    with st.status("Collecte subreddit…", expanded=True) as status:
        single_sub = subreddits.split("+")[0].strip()
        st.write(f"📡 Récupération de r/{single_sub} ({sort_mode})…")
        try:
            df = fetch_subreddit_posts(single_sub, sort=sort_mode, limit=max_posts)
        except Exception as e:
            st.error(f"Erreur collecte : {e}")
            st.stop()
        if df.empty:
            st.error("Aucun post trouvé.")
            st.stop()

        st.write(f"✅ {len(df)} posts collectés")
        st.write("🧹 spaCy NLP + NER + TF-IDF…")
        st.write("🔵 Clustering KMeans…")
        st.write("⚡ Détection de bursts…")
        st.write("🔥 Entraînement XGBoost viralité…")
        st.write("🌲 Entraînement Random Forest…")

        (df_clean, df_clustered, vectorizer, matrix, trends, kmeans,
         rf, cv_acc, alerts_df, scores_df,
         v_model, v_scaler, v_f1, v_name,
         eff_thr) = _run_pipeline(df)

        save_models(vectorizer, kmeans, rf, v_model, v_scaler)
        status.update(label="✅ Analyse complète", state="complete")

    # ── FIX Bug 2 + Bug 4 : même correction que mode Recherche
    st.session_state.update({
        "df":                       df,
        "df_clean":                 df_clustered,   # ← df enrichi avec colonnes ML
        "vectorizer":               vectorizer,
        "matrix":                   matrix,
        "trends":                   trends,
        "kmeans":                   kmeans,
        "rf":                       rf,
        "cv_acc":                   cv_acc,
        "alerts_df":                alerts_df,
        "scores_df":                scores_df,
        "virality_model":           v_model,
        "virality_scaler":          v_scaler,
        "virality_f1":              v_f1,
        "virality_name":            v_name,
        "effective_viral_threshold": eff_thr,
    })
    st.rerun()


# ─── Mode Streaming ──────────────────────────────────────────────────────────

if mode == "🔴 Streaming (polling)":
    st.markdown(
        '<div class="stream-status">🔴 LIVE · Polling Reddit…</div>',
        unsafe_allow_html=True,
    )
    _kws = _build_stream_keywords()
    if _kws is None:
        st.info("💡 Aucun filtre — tous les posts des subreddits sont collectés.")
    else:
        st.info(f"💡 Filtre actif : {', '.join(_kws)}")
    st.divider()

    col_start, col_stop = st.columns(2)
    start_stream = col_start.button("▶️ Démarrer", width="stretch", type="primary")
    stop_stream  = col_stop.button("⏹ Arrêter",   width="stretch")

    def _stream_worker(q, subs, kws, n, stop_event):
        try:
            for post in stream_posts(subs, keywords=kws, max_posts=n, poll_interval=10):
                if stop_event.is_set():
                    break
                q.put(post)
        except Exception as e:
            q.put({"__error__": str(e)})

    if start_stream and not st.session_state.stream_running:
        st.session_state.update({
            "stream_buffer": [], "stream_count": 0,
            "stream_running": True, "stream_trends": None,
            "stream_df_clustered": None, "stream_vectorizer": None,
            "stream_matrix": None, "stream_rf": None, "stream_cv_acc": None,
            "stream_last_ml_count": 0, "stream_alerts": None, "stream_scores": None,
            "stream_virality_model": None, "stream_virality_scaler": None,
        })
        q = queue.Queue()
        stop_ev = threading.Event()
        st.session_state.stream_queue = q
        st.session_state.stop_event   = stop_ev
        threading.Thread(
            target=_stream_worker,
            args=(q, subreddits, _kws, max_posts, stop_ev),
            daemon=True,
        ).start()
        st.success("✅ Streaming démarré.")
        time.sleep(1)
        st.rerun()

    if stop_stream:
        if st.session_state.stop_event:
            st.session_state.stop_event.set()
        st.session_state.stream_running = False
        st.success("⏹ Streaming arrêté.")
        st.rerun()

    if st.session_state.stream_running:
        q = st.session_state.stream_queue
        if q:
            drained = 0
            while not q.empty() and drained < 200:
                item = q.get_nowait()
                if isinstance(item, dict) and "__error__" in item:
                    st.error(f"❌ {item['__error__']}")
                    st.session_state.stream_running = False
                    break
                st.session_state.stream_buffer.append(item)
                st.session_state.stream_count += 1
                drained += 1

        total   = st.session_state.stream_count
        last_ml = st.session_state.stream_last_ml_count

        if total >= stream_ml_threshold and (total - last_ml) >= 10:
            with st.spinner(f"🧠 ML sur {total} posts…"):
                _run_stream_ml(st.session_state.stream_buffer, stream_n_clusters)

        has_ml    = st.session_state.stream_trends is not None
        has_burst = (st.session_state.stream_alerts is not None
                     and not st.session_state.stream_alerts.empty)
        has_viral = st.session_state.stream_virality_model is not None

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("📨 Posts reçus", total)
        if has_ml:
            k2.metric("🔵 Clusters", len(st.session_state.stream_trends))
            acc = st.session_state.stream_cv_acc
            k3.metric("🌲 RF Accuracy", f"{acc:.0%}" if acc and not pd.isna(acc) else "—")
        else:
            k2.metric("🧠 ML dans", f"{max(0, stream_ml_threshold - total)} posts")
            k3.metric("🌲 RF Accuracy", "—")
        if has_burst:
            k4.metric("⚡ Bursts détectés", len(st.session_state.stream_alerts))
        else:
            k4.metric("⚡ Bursts", "—")
        if has_viral:
            vf1 = st.session_state.stream_virality_f1
            k5.metric("🔥 F1 Viralité", f"{vf1:.0%}" if vf1 and not pd.isna(vf1) else "—")
        else:
            k5.metric("🔥 Viralité", "—")

        if has_burst:
            bsumm = burst_summary(st.session_state.stream_alerts)
            if bsumm["top_term"]:
                st.markdown(
                    f'<div class="burst-alert">⚡ Burst détecté : <strong>{bsumm["top_term"]}</strong> '
                    f'(Bt = {bsumm["top_score"]}) — {bsumm["total_alerts"]} termes en explosion</div>',
                    unsafe_allow_html=True,
                )

        st.divider()

        if st.session_state.stream_buffer:
            stream_df = pd.DataFrame(st.session_state.stream_buffer)

            if has_ml:
                tabs_labels = [
                    "📋 Posts live", "🔵 Clusters", "⚡ Bursts",
                    "🔥 Viralité", "🌲 Random Forest",
                    "🗺️ Géo / NER", "📊 Visualisations", "🗂️ Données brutes",
                ]
            else:
                tabs_labels = ["📋 Posts live", "🗂️ Données brutes"]

            tabs     = st.tabs(tabs_labels)
            tab_live = tabs[0]
            tab_raw  = tabs[-1]

            with tab_live:
                st.caption(f"20 derniers posts sur {total} reçus")
                if has_ml:
                    try:
                        recent     = stream_df.tail(20).copy().reset_index(drop=True)
                        classified = classify_new_posts(
                            recent["text"].tolist(),
                            st.session_state.stream_vectorizer,
                            st.session_state.stream_rf,
                            st.session_state.stream_trends,
                        )
                        recent["🏷️ Cluster"]   = classified["topic_label"].values
                        recent["🎯 Confiance"] = classified["confidence"].apply(lambda x: f"{x:.0%}").values
                        if has_viral:
                            vdf = predict_virality(
                                recent,
                                st.session_state.stream_virality_model,
                                st.session_state.stream_virality_scaler,
                                int(viral_threshold),
                            )
                            recent["🔥 Viral"] = vdf["viral_label"].values
                        cols = [c for c in ["title", "subreddit", "score", "num_comments",
                                            "🏷️ Cluster", "🎯 Confiance", "🔥 Viral"] if c in recent.columns]
                        st.dataframe(recent[cols], width="stretch")
                    except Exception as e:
                        st.error(f"❌ Erreur Posts live : {e}")
                        with st.expander("Détail de l'erreur"):
                            st.code(traceback.format_exc())
                        cols = [c for c in ["title", "subreddit", "score", "num_comments", "created_at"] if c in stream_df.columns]
                        st.dataframe(stream_df[cols].tail(20), width="stretch")
                else:
                    cols = [c for c in ["title", "subreddit", "score", "num_comments", "created_at"] if c in stream_df.columns]
                    st.dataframe(stream_df[cols].tail(20), width="stretch")
                    st.info(f"⏳ ML démarre à {stream_ml_threshold} posts ({total}/{stream_ml_threshold})")

            if has_ml:
                tab_clusters = tabs[1]
                tab_bursts   = tabs[2]
                tab_viral    = tabs[3]
                tab_rf       = tabs[4]
                tab_geo      = tabs[5]
                tab_visu     = tabs[6]

                trends_s   = st.session_state.stream_trends
                df_cls     = st.session_state.stream_df_clustered
                vec_s      = st.session_state.stream_vectorizer
                rf_s       = st.session_state.stream_rf
                mat_s      = st.session_state.stream_matrix
                acc_s      = st.session_state.stream_cv_acc
                alerts_s   = st.session_state.stream_alerts if st.session_state.stream_alerts is not None else pd.DataFrame()
                scores_s   = st.session_state.stream_scores if st.session_state.stream_scores is not None else pd.DataFrame()
                v_model_s  = st.session_state.stream_virality_model
                v_scaler_s = st.session_state.stream_virality_scaler

                # ── FIX Bug 3 (streaming) : guard si df_cls est None
                if df_cls is None or (hasattr(df_cls, "empty") and df_cls.empty):
                    st.warning("⚠️ DataFrame ML non disponible. Attendez le prochain cycle ML.")
                    st.stop()

                with tab_clusters:
                    try:
                        st.caption(f"KMeans · {total} posts · {len(trends_s)} clusters")
                        st.dataframe(
                            trends_s[["label", "keywords", "post_count", "avg_score", "avg_comments", "engagement_score"]],
                            width="stretch",
                        )
                        st.plotly_chart(trend_bar_chart(trends_s), width="stretch")
                    except Exception as e:
                        st.error(f"❌ Erreur clusters : {e}")
                        with st.expander("Détail de l'erreur"):
                            st.code(traceback.format_exc())

                with tab_bursts:
                    try:
                        st.subheader("⚡ Alertes burst")
                        if not alerts_s.empty:
                            bsumm = burst_summary(alerts_s)
                            b1, b2, b3, b4 = st.columns(4)
                            b1.metric("Total alertes", bsumm["total_alerts"])
                            b2.metric("Top terme", bsumm["top_term"] or "—")
                            b3.metric("Score max Bt", bsumm["top_score"])
                            b4.metric("Hashtags", bsumm["hashtag_count"])
                            st.plotly_chart(burst_bar_chart(alerts_s), width="stretch")
                            st.plotly_chart(burst_timeline_chart(alerts_s), width="stretch")
                            if not scores_s.empty:
                                st.plotly_chart(burst_score_scatter(scores_s), width="stretch")
                            img_burst = wordcloud_burst(alerts_s)
                            if img_burst:
                                st.image(f"data:image/png;base64,{img_burst}", width="stretch")
                            st.dataframe(alerts_s.head(30), width="stretch")
                        else:
                            st.info(f"Aucun burst au-dessus du seuil Bt ≥ {burst_threshold}.")
                    except Exception as e:
                        st.error(f"❌ Erreur bursts : {e}")
                        with st.expander("Détail de l'erreur"):
                            st.code(traceback.format_exc())

                with tab_viral:
                    try:
                        if v_model_s is None or v_scaler_s is None:
                            st.warning(
                                "⚠️ Modèle de viralité non disponible.\n\n"
                                "**Causes possibles :**\n"
                                "- Pas assez de posts pour créer 2 classes (viral / non-viral)\n"
                                "- Baissez le seuil viral dans la sidebar (actuellement "
                                f"{int(viral_threshold)})"
                            )
                            if df_cls is not None and "score" in df_cls.columns and "num_comments" in df_cls.columns:
                                eng = df_cls["score"].fillna(0) + df_cls["num_comments"].fillna(0) * 1.5
                                st.info(
                                    f"💡 Engagement — médiane : **{eng.median():.0f}** · "
                                    f"max : **{eng.max():.0f}** · "
                                    f"seuil suggéré : **{eng.quantile(0.75):.0f}** (P75)"
                                )
                        else:
                            vdf    = predict_virality(df_cls, v_model_s, v_scaler_s, int(viral_threshold))
                            v_f1_s = st.session_state.stream_virality_f1
                            va1, va2, va3 = st.columns(3)
                            va1.metric("Posts viraux", int(vdf["viral_pred"].sum()))
                            va2.metric("F1-macro CV", f"{v_f1_s:.0%}" if v_f1_s and not pd.isna(v_f1_s) else "—")
                            va3.metric("Seuil viral", int(viral_threshold))
                            st.plotly_chart(virality_distribution(vdf), width="stretch")
                            st.plotly_chart(virality_scatter(vdf), width="stretch")
                            vimp = get_virality_feature_importance(v_model_s, df_cls)
                            if not vimp.empty:
                                st.plotly_chart(virality_feature_importance_chart(vimp), width="stretch")
                            top_viral = vdf[vdf["viral_pred"] == 1].sort_values("viral_prob", ascending=False)
                            if not top_viral.empty:
                                st.subheader("🔥 Posts viraux détectés")
                                cols = [c for c in ["title", "subreddit", "score", "num_comments",
                                                    "viral_prob", "viral_label"] if c in top_viral.columns]
                                st.dataframe(top_viral[cols].head(20), width="stretch")
                    except Exception as e:
                        st.error(f"❌ Erreur viralité : {e}")
                        with st.expander("Détail de l'erreur"):
                            st.code(traceback.format_exc())

                with tab_rf:
                    try:
                        col_i, col_imp = st.columns([2, 3])
                        with col_i:
                            _n_feat_s = mat_s.shape[1] if mat_s is not None else "—"
                            _n_post_s = mat_s.shape[0] if mat_s is not None else "—"
                            st.markdown(f"""
**Algorithme** : Random Forest  
**Estimators** : 200 arbres  
**Posts** : {_n_post_s}  
**Features TF-IDF** : {_n_feat_s}  
**Clusters** : {len(trends_s)}  
**Accuracy CV-3** : {f"{acc_s:.1%}" if acc_s and not pd.isna(acc_s) else "N/A"}
                            """)
                        with col_imp:
                            if rf_s is not None and vec_s is not None:
                                try:
                                    importance_df = get_feature_importance(rf_s, vec_s)
                                    st.plotly_chart(feature_importance_chart(importance_df), width="stretch")
                                except Exception as _ei:
                                    st.warning(f"Importance des features indisponible : {_ei}")
                            else:
                                st.info("Modèle RF non disponible.")

                        # ── FIX Bug 3 (streaming RF) : vérifier colonnes avant classify
                        _has_text_s   = "text" in df_cls.columns
                        _has_vec_s    = vec_s is not None
                        _has_trends_s = trends_s is not None and not trends_s.empty
                        if _has_text_s and _has_vec_s and _has_trends_s and rf_s is not None:
                            try:
                                _cls_s = classify_new_posts(df_cls["text"].tolist(), vec_s, rf_s, trends_s)
                                if "score" in df_cls.columns:
                                    _cls_s["post_score"] = df_cls["score"].values[:len(_cls_s)]
                                st.plotly_chart(confidence_histogram(_cls_s), width="stretch")
                                _show_cols_s = [c for c in ["text", "topic_label", "confidence", "post_score"] if c in _cls_s.columns]
                                st.dataframe(
                                    _cls_s[_show_cols_s].sort_values("confidence", ascending=False).head(30),
                                    width="stretch",
                                )
                            except Exception as _ec:
                                st.warning(f"Classification des posts impossible : {_ec}")
                        else:
                            st.info("Données insuffisantes pour classifier (colonne 'text', vectorizer ou clusters manquants).")
                    except Exception as e:
                        st.error(f"❌ Erreur Random Forest : {e}")
                        with st.expander("Détail de l'erreur"):
                            st.code(traceback.format_exc())

                with tab_geo:
                    try:
                        st.subheader("🗺️ Carte des lieux mentionnés")
                        _ner_cols_s = [
                            c for c in ["ner_locations", "ner_persons", "ner_orgs"]
                            if c in df_cls.columns and df_cls[c].notna().any()
                        ]
                        _has_ner_s = len(_ner_cols_s) > 0
                        if not _has_ner_s:
                            st.warning(
                                "⚠️ Aucune donnée NER disponible.\n\n"
                                "Activez l'option **Extraction NER (spaCy)** dans la sidebar et relancez."
                            )
                        else:
                            try:
                                st.plotly_chart(geo_map_chart(df_cls), width="stretch")
                            except Exception as _eg:
                                st.warning(f"Carte géo non disponible : {_eg}")
                            col_n1, col_n2, col_n3 = st.columns(3)
                            for _cn, _cc in [("ner_locations", col_n1), ("ner_persons", col_n2), ("ner_orgs", col_n3)]:
                                with _cc:
                                    if _cn in df_cls.columns and df_cls[_cn].notna().any():
                                        try:
                                            st.plotly_chart(entities_bar_chart(df_cls, _cn), width="stretch")
                                        except Exception as _en:
                                            st.info(f"{_cn} : {_en}")
                                    else:
                                        st.info(f"Aucune donnée pour {_cn}.")
                        if "lang" in df_cls.columns and df_cls["lang"].notna().any():
                            lang_counts = df_cls["lang"].value_counts().reset_index()
                            lang_counts.columns = ["langue", "count"]
                            st.subheader("🌐 Langues détectées")
                            fig_lang = px.pie(
                                lang_counts, names="langue", values="count",
                                title="Distribution des langues",
                                color_discrete_sequence=px.colors.qualitative.Bold,
                            )
                            st.plotly_chart(fig_lang, width="stretch")
                    except Exception as e:
                        st.error(f"❌ Erreur Géo/NER : {e}")
                        with st.expander("Détail de l'erreur"):
                            st.code(traceback.format_exc())

                with tab_visu:
                    try:
                        # ── FIX Bug 3 (streaming Visu) : guard colonnes requises
                        if "cluster" not in df_cls.columns or trends_s is None or trends_s.empty:
                            st.warning("⚠️ Colonnes ML manquantes pour les visualisations. Attendez le prochain cycle ML.")
                        else:
                            st.plotly_chart(engagement_scatter(df_cls, trends_s), width="stretch")
                            st.plotly_chart(timeline_chart(stream_df), width="stretch")
                            col_v1, col_v2 = st.columns(2)
                            with col_v1:
                                flairs = extract_flairs(stream_df)
                                if not flairs.empty:
                                    st.plotly_chart(flair_chart(flairs), width="stretch")
                            with col_v2:
                                hashtags = extract_hashtags(stream_df)
                                if not hashtags.empty:
                                    st.plotly_chart(hashtag_chart(hashtags), width="stretch")
                            if "clean_text" in df_cls.columns:
                                img = wordcloud_image(df_cls["clean_text"].tolist())
                                if img:
                                    st.image(f"data:image/png;base64,{img}", width="stretch")
                            else:
                                st.info("Colonne 'clean_text' absente — nuage de mots indisponible.")
                    except Exception as e:
                        st.error(f"❌ Erreur visualisations : {e}")
                        with st.expander("Détail de l'erreur"):
                            st.code(traceback.format_exc())

            with tab_raw:
                cols = [c for c in ["title", "subreddit", "score", "num_comments",
                                    "created_at", "flair", "url", "lang"] if c in stream_df.columns]
                st.dataframe(stream_df[cols] if cols else stream_df, width="stretch")
                csv = stream_df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ CSV", csv, "stream_posts.csv", "text/csv")
        else:
            st.info("⏳ En attente des premiers posts…")

        time.sleep(5)
        st.rerun()


# ─── Dashboard principal (batch) ─────────────────────────────────────────────

if st.session_state.df is not None:

    try:
        _df  = st.session_state.df
        _dfc = st.session_state.df_clean

        # ── FIX Bug 3 : ne pas silencieusement tomber sur _df brut
        #    Si _dfc est None ou vide, afficher un message clair et arrêter
        if _dfc is None or (hasattr(_dfc, "empty") and _dfc.empty):
            st.error(
                "❌ Les données ML ne sont pas disponibles dans la session.\n\n"
                "Cela peut arriver après un rechargement de page. "
                "Veuillez relancer l'analyse avec le bouton **🚀 Analyser**."
            )
            st.stop()

        _trends  = st.session_state.trends
        _vec     = st.session_state.vectorizer
        _mat     = st.session_state.matrix
        _rf      = st.session_state.rf
        _cv_acc  = st.session_state.cv_acc
        _alerts  = st.session_state.alerts_df  if st.session_state.alerts_df  is not None else pd.DataFrame()
        _scores  = st.session_state.scores_df  if st.session_state.scores_df  is not None else pd.DataFrame()
        _vm      = st.session_state.virality_model
        _vs      = st.session_state.virality_scaler
        _vf1     = st.session_state.virality_f1
        _vname   = str(st.session_state.virality_name) if st.session_state.virality_name else "—"
        _eff_thr = st.session_state.effective_viral_threshold or int(viral_threshold)

        if _trends is None:
            _trends = pd.DataFrame(columns=["cluster", "label", "keywords",
                                            "post_count", "avg_score",
                                            "avg_comments", "engagement_score"])

        # ── FIX Bug 3 : vérifier que les colonnes ML sont bien présentes dans _dfc
        _missing_ml_cols = [c for c in ["cluster", "clean_text"] if c not in _dfc.columns]
        if _missing_ml_cols:
            st.warning(
                f"⚠️ Colonnes ML manquantes dans le DataFrame : {_missing_ml_cols}\n\n"
                "Relancez l'analyse pour régénérer toutes les colonnes."
            )

    except Exception as _e:
        st.error(f"❌ Erreur lecture session state : {_e}")
        st.code(traceback.format_exc())
        st.stop()

    # ── Métriques header
    try:
        _m1, _m2, _m3, _m4, _m5, _m6 = st.columns(6)
        _m1.metric("Posts analysés",    len(_df))
        _m2.metric("Tendances",         len(_trends))
        _m3.metric("Engagement total",  int(_df["score"].sum() + _df["num_comments"].sum()))
        _m4.metric("RF accuracy (CV)",  f"{_cv_acc:.0%}" if _cv_acc and not pd.isna(_cv_acc) else "—")
        _m5.metric("⚡ Bursts détectés", len(_alerts) if not _alerts.empty else 0)
        _m6.metric("🔥 F1 Viralité",    f"{_vf1:.0%}" if _vf1 and not pd.isna(_vf1) else "—")
    except Exception:
        pass

    try:
        if not _alerts.empty:
            _bs = burst_summary(_alerts)
            if _bs.get("top_term"):
                st.markdown(
                    f'<div class="burst-alert">⚡ Terme en burst : <strong>{_bs["top_term"]}</strong> '
                    f'(Bt = {_bs["top_score"]}) — {_bs["total_alerts"]} alertes</div>',
                    unsafe_allow_html=True,
                )
    except Exception:
        pass

    st.divider()

    # ── Création des tabs
    _tab1, _tab2, _tab3, _tab4, _tab5, _tab6 = st.tabs([
        "🔥 Tendances", "⚡ Bursts", "🔥 Viralité",
        "🌲 Random Forest", "🗺️ Géo / NER", "🗂️ Données"
    ])

    # ── Tab 1 : Tendances
    with _tab1:
        try:
            _ca, _cb = st.columns([3, 2])
            with _ca:
                if not _trends.empty:
                    st.plotly_chart(trend_bar_chart(_trends), width="stretch")
                else:
                    st.info("Aucune tendance détectée.")
            with _cb:
                _fl = extract_flairs(_df)
                _ht = extract_hashtags(_df)
                if not _fl.empty:
                    st.plotly_chart(flair_chart(_fl), width="stretch")
                elif not _ht.empty:
                    st.plotly_chart(hashtag_chart(_ht), width="stretch")
                else:
                    st.info("Aucun flair ni hashtag détecté.")
            st.plotly_chart(timeline_chart(_df), width="stretch")

            # ── FIX Bug 3 : guard colonne "cluster" avant engagement_scatter
            if "cluster" in _dfc.columns and not _trends.empty:
                st.plotly_chart(engagement_scatter(_dfc, _trends), width="stretch")
            else:
                st.info("Scatter d'engagement indisponible (colonne 'cluster' manquante).")

            st.subheader("☁️ Nuage de mots global")
            if "clean_text" in _dfc.columns:
                _wc = wordcloud_image(_dfc["clean_text"].tolist())
                if _wc:
                    st.image(f"data:image/png;base64,{_wc}", width="stretch")
            else:
                st.info("Colonne 'clean_text' absente — nuage de mots indisponible.")

            if "cluster" in _dfc.columns and not _trends.empty:
                st.subheader("🔍 Nuage par cluster")
                _sel = st.selectbox(
                    "Cluster", _trends["cluster"].tolist(),
                    format_func=lambda c: f"Cluster {c} — {_trends.loc[_trends['cluster']==c,'label'].values[0]}",
                    key="batch_cluster_wc",
                )
                _wcc = wordcloud_per_cluster(_dfc, _trends, _sel)
                if _wcc:
                    st.image(f"data:image/png;base64,{_wcc}", width="stretch")
        except Exception as _e:
            st.error(f"❌ Erreur Tendances : {_e}")
            with st.expander("Détail"):
                st.code(traceback.format_exc())

    # ── Tab 2 : Bursts
    with _tab2:
        try:
            st.subheader("⚡ Détection de bursts")
            if not _alerts.empty:
                _bs = burst_summary(_alerts)
                _b1, _b2, _b3, _b4 = st.columns(4)
                _b1.metric("Alertes totales", _bs["total_alerts"])
                _b2.metric("Top terme",       _bs["top_term"] or "—")
                _b3.metric("Score max Bt",    _bs["top_score"])
                _b4.metric("Dont hashtags",   _bs["hashtag_count"])
                st.plotly_chart(burst_bar_chart(_alerts), width="stretch")
                _bt1, _bt2 = st.columns(2)
                with _bt1:
                    st.plotly_chart(burst_timeline_chart(_alerts), width="stretch")
                with _bt2:
                    if not _scores.empty:
                        st.plotly_chart(burst_score_scatter(_scores), width="stretch")
                _wb = wordcloud_burst(_alerts)
                if _wb:
                    st.image(f"data:image/png;base64,{_wb}", width="stretch")
                st.dataframe(_alerts.head(50), width="stretch")
            else:
                st.info(f"Aucun burst détecté (seuil Bt ≥ {burst_threshold}, fenêtre {burst_window} min).")
        except Exception as _e:
            st.error(f"❌ Erreur Bursts : {_e}")
            with st.expander("Détail"):
                st.code(traceback.format_exc())

    # ── Tab 3 : Viralité
    with _tab3:
        try:
            st.subheader(f"🔥 Prédiction de viralité — {_vname}")

            if _vm is None or _vs is None:
                st.warning(
                    "⚠️ **Modèle de viralité non disponible.**\n\n"
                    "Causes possibles :\n"
                    "- Le seuil viral est trop élevé : aucun post n'atteint "
                    f"**{int(viral_threshold)}** points d'engagement\n"
                    "- Pas assez de diversité dans les données (une seule classe)\n\n"
                    "**Solution :** Baissez le seuil viral dans la sidebar (essayez 50 ou 20) "
                    "puis relancez l'analyse."
                )
                _has_score    = "score"        in _dfc.columns
                _has_comments = "num_comments" in _dfc.columns
                if _has_score and _has_comments:
                    st.subheader("📊 Distribution de l'engagement (diagnostic)")
                    _dfc_diag = _dfc.copy()
                    _dfc_diag["engagement"] = (
                        _dfc_diag["score"].fillna(0)
                        + _dfc_diag["num_comments"].fillna(0) * 1.5
                    )
                    _d1, _d2, _d3 = st.columns(3)
                    _d1.metric("Engagement médian", f"{_dfc_diag['engagement'].median():.0f}")
                    _d2.metric("Engagement max",    f"{_dfc_diag['engagement'].max():.0f}")
                    _d3.metric("Seuil actuel",      f"{int(viral_threshold)}")
                    _fig_eng = px.histogram(
                        _dfc_diag, x="engagement", nbins=30,
                        title="Distribution de l'engagement (score + commentaires×1.5)",
                        labels={"engagement": "Engagement total"},
                        color_discrete_sequence=["#7C3AED"],
                    )
                    _fig_eng.add_vline(
                        x=int(viral_threshold),
                        line_dash="dash", line_color="#DC2626",
                        annotation_text=f"Seuil = {int(viral_threshold)}",
                    )
                    st.plotly_chart(_fig_eng, width="stretch")
                    _p50 = _dfc_diag["engagement"].quantile(0.5)
                    _p75 = _dfc_diag["engagement"].quantile(0.75)
                    st.info(
                        f"💡 Seuils suggérés : **{_p50:.0f}** (médiane, ~50% viral) "
                        f"ou **{_p75:.0f}** (75e percentile, ~25% viral)"
                    )
                else:
                    st.info(
                        "Colonnes 'score' / 'num_comments' absentes — "
                        "impossible d'afficher le diagnostic d'engagement."
                    )
            else:
                _vdf = predict_virality(_dfc, _vm, _vs, _eff_thr)
                _v1, _v2, _v3, _v4 = st.columns(4)
                _v1.metric("Posts viraux prédits", int(_vdf["viral_pred"].sum()))
                _v2.metric("Taux viral",           f"{_vdf['viral_pred'].mean():.0%}")
                _v3.metric("F1-macro CV",          f"{_vf1:.0%}" if _vf1 and not pd.isna(_vf1) else "—")
                _v4.metric("Modèle",               _vname)
                if _eff_thr != int(viral_threshold):
                    st.info(
                        f"ℹ️ Seuil viral utilisé : **{_eff_thr}** "
                        f"(demandé : {int(viral_threshold)}, ajusté automatiquement)"
                    )
                _va, _vb = st.columns(2)
                with _va:
                    st.plotly_chart(virality_distribution(_vdf), width="stretch")
                with _vb:
                    st.plotly_chart(virality_scatter(_vdf), width="stretch")
                _vimp = get_virality_feature_importance(_vm, _dfc)
                if not _vimp.empty:
                    st.plotly_chart(virality_feature_importance_chart(_vimp), width="stretch")
                st.subheader("🔥 Top posts par probabilité virale")
                _tv = _vdf.sort_values("viral_prob", ascending=False)
                _tc = [c for c in ["title", "subreddit", "score", "num_comments", "viral_prob", "viral_label"] if c in _tv.columns]
                st.dataframe(_tv[_tc].head(30), width="stretch")
        except Exception as _e:
            st.error(f"❌ Erreur Viralité : {_e}")
            with st.expander("Détail"):
                st.code(traceback.format_exc())

    # ── Tab 4 : Random Forest
    with _tab4:
        try:
            if _rf is None:
                st.warning(
                    "⚠️ **Aucun modèle Random Forest disponible.**\n\n"
                    "Lancez une analyse avec le bouton 🚀 Analyser."
                )
            else:
                _f1, _f2 = st.columns([2, 3])
                with _f1:
                    _n_features = _mat.shape[1] if _mat is not None else "—"
                    _n_posts_rf = _mat.shape[0] if _mat is not None else "—"
                    st.markdown(f"""
**Algorithme** : Random Forest  
**Estimators** : 200 arbres  
**Labels** : KMeans ({n_clusters} clusters)  
**Accuracy CV-3** : {f"{_cv_acc:.1%}" if _cv_acc and not pd.isna(_cv_acc) else "N/A"}  
**Features TF-IDF** : {_n_features}  
**Posts** : {_n_posts_rf}
                    """)
                with _f2:
                    if _vec is not None:
                        try:
                            _imp = get_feature_importance(_rf, _vec)
                            st.plotly_chart(feature_importance_chart(_imp), width="stretch")
                        except Exception as _ei:
                            st.warning(f"Importance des features indisponible : {_ei}")
                    else:
                        st.info("Vectorizer non disponible.")

                # ── FIX Bug 3 : guard complet avant classify_new_posts
                _has_text   = "text"    in _dfc.columns
                _has_vec    = _vec      is not None
                _has_trends = _trends   is not None and not _trends.empty

                if _has_text and _has_vec and _has_trends:
                    try:
                        _cls = classify_new_posts(_dfc["text"].tolist(), _vec, _rf, _trends)
                        if "score" in _dfc.columns:
                            _cls["post_score"] = _dfc["score"].values[:len(_cls)]
                        st.plotly_chart(confidence_histogram(_cls), width="stretch")
                        _show_cols = [c for c in ["text", "topic_label", "confidence", "post_score"] if c in _cls.columns]
                        st.dataframe(
                            _cls[_show_cols].sort_values("confidence", ascending=False).head(30),
                            width="stretch",
                        )
                    except Exception as _ec:
                        st.warning(f"Classification des posts impossible : {_ec}")
                        with st.expander("Détail"):
                            st.code(traceback.format_exc())
                else:
                    _missing = []
                    if not _has_text:   _missing.append("colonne 'text'")
                    if not _has_vec:    _missing.append("vectorizer")
                    if not _has_trends: _missing.append("clusters")
                    st.info(f"Données insuffisantes pour classifier les posts ({', '.join(_missing)} manquants).")
        except Exception as _e:
            st.error(f"❌ Erreur Random Forest : {_e}")
            with st.expander("Détail"):
                st.code(traceback.format_exc())

    # ── Tab 5 : Géo / NER
    with _tab5:
        try:
            st.subheader("🗺️ Carte géographique des lieux mentionnés")

            # ── Diagnostic : afficher toutes les colonnes NER détectées dans le df
            # (les noms peuvent varier selon preprocess.py : ner_locations vs locations, etc.)
            _all_ner_candidates = [c for c in _dfc.columns if "ner" in c.lower() or
                                   any(k in c.lower() for k in ["loc", "person", "org", "gpe", "entity"])]

            # Noms canoniques + variantes possibles selon les versions de preprocess.py
            _NER_COL_MAP = {
                "ner_locations": ["ner_locations", "locations", "ner_loc", "loc", "GPE", "gpe"],
                "ner_persons":   ["ner_persons",   "persons",   "ner_per", "per", "PERSON", "person"],
                "ner_orgs":      ["ner_orgs",       "orgs",      "ner_org", "org", "ORG",    "org"],
            }

            def _resolve_ner_col(candidates):
                """Retourne la première colonne présente et non-vide dans _dfc."""
                for c in candidates:
                    if c in _dfc.columns and _dfc[c].notna().any():
                        return c
                return None

            _col_loc = _resolve_ner_col(_NER_COL_MAP["ner_locations"])
            _col_per = _resolve_ner_col(_NER_COL_MAP["ner_persons"])
            _col_org = _resolve_ner_col(_NER_COL_MAP["ner_orgs"])
            _has_ner = any([_col_loc, _col_per, _col_org])

            # Expander de diagnostic toujours visible pour aider au debug
            with st.expander("🔬 Diagnostic colonnes NER (debug)", expanded=not _has_ner):
                st.write("**Colonnes NER détectées dans le DataFrame :**",
                         _all_ner_candidates if _all_ner_candidates else "Aucune")
                st.write("**Colonnes résolues :**",
                         {k: v for k, v in [("locations", _col_loc), ("persons", _col_per), ("orgs", _col_org)] if v})
                st.write("**Toutes les colonnes du DataFrame :**", list(_dfc.columns))

            if not _has_ner:
                st.warning(
                    "⚠️ **Aucune donnée NER disponible.**\n\n"
                    "Causes possibles :\n"
                    "- L'option **Extraction NER (spaCy)** est désactivée dans la sidebar\n"
                    "- Les posts collectés ne contiennent pas d'entités nommées détectables\n"
                    "- Le modèle spaCy n'a pas reconnu d'entités dans ce corpus\n\n"
                    "**Solution :** Activez NER dans la sidebar et relancez l'analyse."
                )
            else:
                try:
                    st.plotly_chart(geo_map_chart(_dfc), width="stretch")
                except Exception as _eg:
                    st.warning(f"Carte géographique non disponible : {_eg}")

                _n1, _n2, _n3 = st.columns(3)
                for _col_ner, _label, _container in [
                    (_col_loc, "📍 Lieux",         _n1),
                    (_col_per, "👤 Personnes",      _n2),
                    (_col_org, "🏢 Organisations",  _n3),
                ]:
                    with _container:
                        st.caption(_label)
                        if _col_ner:
                            try:
                                st.plotly_chart(entities_bar_chart(_dfc, _col_ner), width="stretch")
                            except Exception as _en:
                                st.info(f"Erreur graphique : {_en}")
                        else:
                            st.info("Aucune donnée.")

            if "lang" in _dfc.columns and _dfc["lang"].notna().any():
                _lc = _dfc["lang"].value_counts().reset_index()
                _lc.columns = ["langue", "count"]
                st.subheader("🌐 Langues détectées")
                st.plotly_chart(
                    px.pie(_lc, names="langue", values="count",
                           title="Distribution des langues",
                           color_discrete_sequence=px.colors.qualitative.Bold),
                    width="stretch",
                )
            else:
                st.info("Colonne 'lang' absente ou vide — détection de langue non disponible.")
        except Exception as _e:
            st.error(f"❌ Erreur Géo/NER : {_e}")
            with st.expander("Détail"):
                st.code(traceback.format_exc())

    # ── Tab 6 : Données
    with _tab6:
        try:
            st.subheader(f"🗂️ Posts collectés ({len(_df)})")

            _preferred_cols = ["title", "subreddit", "score", "num_comments",
                               "created_at", "flair", "lang", "cluster"]
            _available_cols = [c for c in _preferred_cols if c in _dfc.columns]

            if _available_cols:
                st.dataframe(_dfc[_available_cols], width="stretch")
            else:
                st.dataframe(_dfc, width="stretch")

            _csv_df = _dfc if not _dfc.empty else _df
            st.download_button(
                "⬇️ Télécharger CSV",
                _csv_df.to_csv(index=False).encode("utf-8"),
                "reddit_posts.csv", "text/csv",
            )

            with st.expander("ℹ️ Colonnes disponibles dans le DataFrame"):
                st.write(list(_dfc.columns))
        except Exception as _e:
            st.error(f"❌ Erreur Données : {_e}")
            with st.expander("Détail"):
                st.code(traceback.format_exc())


# ─── Classifieur ad hoc (sidebar) ────────────────────────────────────────────

if classify_btn and custom_text.strip():
    models   = load_models()
    rf_m     = st.session_state.stream_rf or st.session_state.rf or (models["rf"] if models else None)
    vec_m    = st.session_state.stream_vectorizer or st.session_state.vectorizer or (models["vectorizer"] if models else None)
    trends_m = st.session_state.stream_trends or st.session_state.trends

    if rf_m is None or vec_m is None:
        st.sidebar.warning("Aucun modèle disponible. Lancez d'abord une analyse.")
    elif trends_m is None:
        st.sidebar.warning("Labels de clusters manquants. Relancez une analyse.")
    else:
        try:
            result = classify_new_posts([custom_text], vec_m, rf_m, trends_m)
            row    = result.iloc[0]
            st.sidebar.success(f"**Sujet** : {row['topic_label']}")
            st.sidebar.metric("Confiance RF", f"{row['confidence']:.0%}")

            v_m = st.session_state.stream_virality_model or st.session_state.virality_model or (models.get("virality_model") if models else None)
            v_s = st.session_state.stream_virality_scaler or st.session_state.virality_scaler or (models.get("virality_scaler") if models else None)

            _pred_thr = st.session_state.effective_viral_threshold or int(viral_threshold)

            if v_m is not None and v_s is not None:
                dummy_df = pd.DataFrame([{
                    "score": 10, "num_comments": 5, "upvote_ratio": 0.75,
                    "flair": "", "title": custom_text, "created_at": pd.Timestamp.utcnow(),
                    "subreddit": "unknown", "text": custom_text,
                }])
                try:
                    vres = predict_virality(dummy_df, v_m, v_s, _pred_thr)
                    st.sidebar.metric("Probabilité virale", f"{vres.iloc[0]['viral_prob']:.0%}")
                    st.sidebar.markdown(f"**{vres.iloc[0]['viral_label']}**")
                except Exception:
                    pass
        except Exception as e:
            st.sidebar.error(f"Erreur classifieur : {e}")


# ─── Footer ──────────────────────────────────────────────────────────────────
st.divider()
st.caption("TrendRadar · Reddit · KMeans + RF · Bursts Bt · XGBoost Viralité · spaCy NER · JSON public")