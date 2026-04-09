"""
app.py — TrendRadar Reddit
Dashboard Streamlit : détection de tendances (KMeans) + classification (Random Forest)
"""

import time
import threading
import queue

import pandas as pd
import streamlit as st

from src.collect import fetch_posts, fetch_subreddit_posts, stream_posts, load_cached
from src.preprocess import preprocess
from src.model import (
    detect_trends,
    train_classifier,
    classify_new_posts,
    get_feature_importance,
    extract_hashtags,
    extract_flairs,
    save_models,
    load_models,
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

  /* Badge subreddit */
  .sub-badge {
    display: inline-block; background: #FF4500; color: white;
    padding: 2px 8px; border-radius: 4px; font-size: 12px;
    font-family: 'IBM Plex Mono', monospace;
  }

  /* Ligne de statut streaming */
  .stream-status {
    background: #0d1117; color: #39d353; padding: 6px 12px;
    border-radius: 6px; font-family: 'IBM Plex Mono', monospace;
    font-size: 13px; border: 1px solid #238636;
  }
</style>
""", unsafe_allow_html=True)


# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 📡 TrendRadar")
    st.caption("Reddit · JSON public · KMeans + Random Forest")
    st.divider()

    mode = st.radio(
        "Mode de collecte",
        ["📥 Recherche (query)", "📋 Subreddit direct", "🔴 Streaming (polling)"],
        index=0,
    )

    st.subheader("🔍 Sujet")
    query = st.text_input("Mots-clés", value="intelligence artificielle")
    subreddits = st.text_input(
        "Subreddits (séparés par +)",
        value="france+French+programming+MachineLearning",
    )

    st.subheader("⚙️ Paramètres")
    max_posts = st.slider("Nombre de posts", 50, 500, 150, step=25)
    n_clusters = st.slider("Clusters KMeans", 3, 10, 5)
    sort_mode = st.selectbox("Tri des posts", ["new", "hot", "top", "relevance"])

    if sort_mode == "top":
        time_filter = st.selectbox("Période", ["day", "week", "month", "year"])
    else:
        time_filter = "week"

    st.divider()

    col_run, col_cache = st.columns(2)
    run_btn = col_run.button("🚀 Analyser", use_container_width=True, type="primary")
    cache_btn = col_cache.button("📂 Charger cache", use_container_width=True)

    st.divider()
    st.subheader("🧪 Classifier un post")
    custom_text = st.text_area("Texte à classifier", height=80, placeholder="Collez un post Reddit…")
    classify_btn = st.button("🌲 Classifier", use_container_width=True)


# ─── State ───────────────────────────────────────────────────────────────────

import os as _os

for key in ("df", "df_clean", "vectorizer", "matrix", "trends", "kmeans", "rf", "cv_acc"):
    if key not in st.session_state:
        st.session_state[key] = None

# ─── Reset automatique si les paramètres changent ────────────────────────────
_current_sig = f"{query}|{subreddits}|{max_posts}|{n_clusters}|{sort_mode}"

if "last_sig" not in st.session_state:
    st.session_state["last_sig"] = _current_sig

if st.session_state["last_sig"] != _current_sig:
    for key in ("df", "df_clean", "vectorizer", "matrix", "trends", "kmeans", "rf", "cv_acc"):
        st.session_state[key] = None
    st.session_state["last_sig"] = _current_sig
    _cache_path = "data/reddit_snapshot.parquet"
    if _os.path.exists(_cache_path):
        _os.remove(_cache_path)
    st.info("🔄 Paramètres modifiés — données précédentes effacées.")


# ─── Header ──────────────────────────────────────────────────────────────────

st.title("📡 TrendRadar — Reddit")
st.caption(f"Subreddit(s) : **{subreddits}** · Requête : **{query}**")


# ─── Chargement du cache ─────────────────────────────────────────────────────

if cache_btn:
    cached = load_cached()
    if cached is not None:
        st.session_state.df = cached
        st.success(f"Cache chargé : {len(cached)} posts")
    else:
        st.warning("Aucun cache disponible. Lancez d'abord une analyse.")


# ─── Helper : pipeline complet ───────────────────────────────────────────────

def _run_pipeline(df: pd.DataFrame):
    """Prétraitement → KMeans → Random Forest. Retourne tous les objets utiles."""
    df_clean, vectorizer, matrix = preprocess(df)
    # detect_trends retourne (trends, df_with_cluster_col, kmeans)
    trends, df_clustered, kmeans = detect_trends(df_clean, matrix, vectorizer, n_clusters)
    rf, cv_acc = train_classifier(df_clustered, matrix)
    return df_clean, df_clustered, vectorizer, matrix, trends, kmeans, rf, cv_acc


# ─── Mode Recherche ──────────────────────────────────────────────────────────

if run_btn and mode == "📥 Recherche (query)":
    with st.status("Collecte Reddit en cours…", expanded=True) as status:
        st.write("📡 Requête JSON publique Reddit…")
        try:
            df = fetch_posts(subreddits, query, max_posts, sort_mode, time_filter)
        except Exception as e:
            st.error(f"Erreur : {e}")
            st.stop()

        if df.empty:
            st.error("Aucun post trouvé. Essayez d'autres mots-clés ou subreddits.")
            st.stop()

        st.write(f"✅ {len(df)} posts collectés")
        st.write("🧹 Nettoyage NLP + TF-IDF…")
        st.write("🔵 Clustering KMeans…")
        st.write("🌲 Entraînement Random Forest…")

        df_clean, df_clustered, vectorizer, matrix, trends, kmeans, rf, cv_acc = _run_pipeline(df)
        save_models(vectorizer, kmeans, rf)
        status.update(label="✅ Analyse complète", state="complete")

    st.session_state.update({
        "df": df, "df_clean": df_clustered, "vectorizer": vectorizer,
        "matrix": matrix, "trends": trends, "kmeans": kmeans,
        "rf": rf, "cv_acc": cv_acc,
    })


# ─── Mode Subreddit direct ───────────────────────────────────────────────────

if run_btn and mode == "📋 Subreddit direct":
    with st.status("Collecte subreddit…", expanded=True) as status:
        single_sub = subreddits.split("+")[0].strip()
        st.write(f"📡 Récupération de r/{single_sub} ({sort_mode})…")
        try:
            df = fetch_subreddit_posts(single_sub, sort=sort_mode, limit=max_posts)
        except Exception as e:
            st.error(f"Erreur : {e}")
            st.stop()

        if df.empty:
            st.error("Aucun post trouvé.")
            st.stop()

        st.write(f"✅ {len(df)} posts collectés")
        st.write("🧹 Nettoyage NLP + TF-IDF…")
        st.write("🔵 Clustering KMeans…")
        st.write("🌲 Entraînement Random Forest…")

        df_clean, df_clustered, vectorizer, matrix, trends, kmeans, rf, cv_acc = _run_pipeline(df)
        save_models(vectorizer, kmeans, rf)
        status.update(label="✅ Analyse complète", state="complete")

    st.session_state.update({
        "df": df, "df_clean": df_clustered, "vectorizer": vectorizer,
        "matrix": matrix, "trends": trends, "kmeans": kmeans,
        "rf": rf, "cv_acc": cv_acc,
    })


# ─── Mode Streaming (polling) ────────────────────────────────────────────────

if run_btn and mode == "🔴 Streaming (polling)":
    st.markdown('<div class="stream-status">🔴 LIVE · Polling Reddit toutes les 30s…</div>', unsafe_allow_html=True)
    st.info("💡 Le streaming utilise du polling sur /new.json. Aucune clé API requise.")

    models = load_models()
    if models is None:
        st.warning("⚠️ Aucun modèle entraîné. Lancez d'abord une analyse (Recherche ou Subreddit direct).")
        st.stop()

    vectorizer = models["vectorizer"]
    rf = models["rf"]
    trends_live = (
        st.session_state.trends
        if st.session_state.trends is not None
        else pd.DataFrame(
            [{"cluster": i, "label": f"Cluster {i}"} for i in range(10)]
        )
    )

    stream_placeholder = st.empty()
    stop_btn = st.button("⏹ Arrêter le streaming")

    post_queue: queue.Queue = queue.Queue()
    stream_buffer = []
    collected_count = [0]

    def _stream_worker():
        for post in stream_posts(subreddits, keywords=query.split() if query else None, max_posts=max_posts):
            post_queue.put(post)
            collected_count[0] += 1

    thread = threading.Thread(target=_stream_worker, daemon=True)
    thread.start()

    while not stop_btn:
        while not post_queue.empty():
            stream_buffer.append(post_queue.get())

        if stream_buffer:
            stream_df = pd.DataFrame(stream_buffer[-50:])
            classified = classify_new_posts(
                stream_df["text"].tolist(), vectorizer, rf, trends_live
            )
            with stream_placeholder.container():
                st.metric("Posts reçus", collected_count[0])
                st.dataframe(
                    classified[["text", "topic_label", "confidence"]].tail(10),
                    use_container_width=True,
                )

        time.sleep(2)
        if stop_btn or not thread.is_alive():
            break

    st.success("Streaming arrêté.")


# ─── Dashboard principal ─────────────────────────────────────────────────────

if st.session_state.df is not None:
    df         = st.session_state.df
    df_clean   = st.session_state.df_clean   # contient la colonne 'cluster'
    trends     = st.session_state.trends
    vectorizer = st.session_state.vectorizer
    matrix     = st.session_state.matrix
    rf         = st.session_state.rf
    cv_acc     = st.session_state.cv_acc

    # ── KPIs ──────────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Posts analysés",   len(df))
    k2.metric("Tendances",        len(trends))
    k3.metric("Engagement total", int(df["score"].sum() + df["num_comments"].sum()))
    k4.metric("Upvote ratio moy", f"{df['upvote_ratio'].mean():.0%}" if "upvote_ratio" in df.columns else "—")
    if cv_acc and not pd.isna(cv_acc):
        k5.metric("RF accuracy (CV)", f"{cv_acc:.0%}")
    else:
        k5.metric("RF accuracy (CV)", "—")

    st.divider()

    # ── Onglets ───────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔥 Tendances", "🌲 Random Forest", "📊 Exploration", "🗂️ Données"
    ])

    # ── Tab 1 : Tendances ─────────────────────────────────────────────────────
    with tab1:
        col_a, col_b = st.columns([3, 2])
        with col_a:
            st.plotly_chart(trend_bar_chart(trends), use_container_width=True)
        with col_b:
            flairs = extract_flairs(df)
            hashtags = extract_hashtags(df)
            if not flairs.empty:
                st.plotly_chart(flair_chart(flairs), use_container_width=True)
            elif not hashtags.empty:
                st.plotly_chart(hashtag_chart(hashtags), use_container_width=True)
            else:
                st.info("Aucun flair ni hashtag détecté dans ce corpus.")

        st.plotly_chart(timeline_chart(df), use_container_width=True)

        st.subheader("☁️ Nuage de mots global")
        img = wordcloud_image(df_clean["clean_text"].tolist())
        if img:
            # FIX : use_container_width remplace use_column_width (déprécié)
            st.image(f"data:image/png;base64,{img}", use_container_width=True)

        st.subheader("🔍 Nuage par cluster")
        selected_cluster = st.selectbox(
            "Choisir un cluster",
            trends["cluster"].tolist(),
            format_func=lambda c: f"Cluster {c} — {trends.loc[trends['cluster']==c, 'label'].values[0]}",
        )
        img_c = wordcloud_per_cluster(df_clean, trends, selected_cluster)
        if img_c:
            # FIX : use_container_width remplace use_column_width (déprécié)
            st.image(f"data:image/png;base64,{img_c}", use_container_width=True)

    # ── Tab 2 : Random Forest ─────────────────────────────────────────────────
    with tab2:
        if rf is not None:
            col_f1, col_f2 = st.columns([2, 3])
            with col_f1:
                st.subheader("📋 Résumé du modèle")
                st.markdown(f"""
- **Algorithme** : Random Forest  
- **Estimators** : 200 arbres  
- **Labels** : issus de KMeans ({n_clusters} clusters)  
- **Accuracy CV-3** : {f"{cv_acc:.1%}" if cv_acc and not pd.isna(cv_acc) else "N/A"}
- **Features TF-IDF** : {matrix.shape[1]}  
- **Posts d'entraînement** : {matrix.shape[0]}
                """)

            with col_f2:
                importance_df = get_feature_importance(rf, vectorizer)
                st.plotly_chart(feature_importance_chart(importance_df), use_container_width=True)

            st.subheader("🎯 Classement des posts par confiance")
            classified_all = classify_new_posts(
                df_clean["text"].tolist(), vectorizer, rf, trends
            )
            classified_all["post_score"] = df_clean["score"].values
            st.plotly_chart(confidence_histogram(classified_all), use_container_width=True)

            st.dataframe(
                classified_all[["text", "topic_label", "confidence", "post_score"]]
                .sort_values("confidence", ascending=False)
                .head(30),
                use_container_width=True,
            )
        else:
            st.info("Lancez une analyse pour entraîner le Random Forest.")

    # ── Tab 3 : Exploration ───────────────────────────────────────────────────
    with tab3:
        st.plotly_chart(engagement_scatter(df_clean, trends), use_container_width=True)

        st.subheader("📊 Statistiques par cluster")
        st.dataframe(
            trends[["label", "keywords", "post_count", "avg_score", "avg_comments", "engagement_score"]],
            use_container_width=True,
        )

    # ── Tab 4 : Données brutes ────────────────────────────────────────────────
    with tab4:
        st.subheader(f"🗂️ Posts collectés ({len(df)})")
        cols_show = [c for c in ["title", "subreddit", "score", "num_comments", "created_at", "flair"] if c in df.columns]
        st.dataframe(df[cols_show], use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Télécharger CSV", csv, "reddit_posts.csv", "text/csv")


# ─── Classifieur ad hoc (sidebar) ────────────────────────────────────────────

if classify_btn and custom_text.strip():
    models = load_models()
    if models is None or st.session_state.rf is None:
        st.sidebar.warning("Aucun modèle entraîné. Lancez une analyse d'abord.")
    else:
        rf_m = st.session_state.rf or models["rf"]
        vec_m = st.session_state.vectorizer or models["vectorizer"]
        trends_m = st.session_state.trends

        if trends_m is not None:
            result = classify_new_posts([custom_text], vec_m, rf_m, trends_m)
            row = result.iloc[0]
            st.sidebar.success(f"**Sujet** : {row['topic_label']}")
            st.sidebar.metric("Confiance RF", f"{row['confidence']:.0%}")
        else:
            st.sidebar.warning("Relancez une analyse pour avoir les labels de clusters.")


# ─── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.caption("TrendRadar · Reddit · KMeans + Random Forest · JSON public · Sans clé API")