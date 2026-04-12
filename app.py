"""
app.py — TrendRadar Reddit
Dashboard Streamlit : détection de tendances (KMeans) + classification (Random Forest)
Mode streaming enrichi : clustering + RF + visualisations en temps réel
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

    # ── Options streaming ─────────────────────────────────────────────────────
    if mode == "🔴 Streaming (polling)":
        st.divider()
        st.subheader("🔴 Options streaming")

        stream_filter_mode = st.radio(
            "Filtre des posts",
            ["Aucun filtre (tout accepter)", "Filtrer par mots-clés"],
            index=0,
        )
        if stream_filter_mode == "Filtrer par mots-clés":
            stream_keywords_raw = st.text_input(
                "Mots-clés (séparés par virgule)",
                value=query,
                help="Ex: AI, machine learning, LLM",
            )
        else:
            stream_keywords_raw = ""

        st.divider()
        st.subheader("🧠 ML en streaming")
        stream_ml_threshold = st.slider(
            "Déclencher ML après N posts",
            min_value=10, max_value=100, value=20, step=5,
            help="KMeans + RF se lancent automatiquement quand ce seuil est atteint.",
        )
        stream_n_clusters = st.slider(
            "Clusters KMeans (streaming)",
            min_value=2, max_value=8, value=min(n_clusters, 4),
        )
    else:
        stream_filter_mode  = "Aucun filtre (tout accepter)"
        stream_keywords_raw = ""
        stream_ml_threshold = 20
        stream_n_clusters   = 4

    st.divider()

    col_run, col_cache = st.columns(2)
    run_btn   = col_run.button("🚀 Analyser", use_container_width=True, type="primary")
    cache_btn = col_cache.button("📂 Charger cache", use_container_width=True)

    st.divider()
    st.subheader("🧪 Classifier un post")
    custom_text  = st.text_area("Texte à classifier", height=80, placeholder="Collez un post Reddit…")
    classify_btn = st.button("🌲 Classifier", use_container_width=True)


# ─── State ───────────────────────────────────────────────────────────────────

import os as _os

for key in ("df", "df_clean", "vectorizer", "matrix", "trends", "kmeans", "rf", "cv_acc"):
    if key not in st.session_state:
        st.session_state[key] = None

if "stream_running"       not in st.session_state: st.session_state.stream_running       = False
if "stream_buffer"        not in st.session_state: st.session_state.stream_buffer        = []
if "stream_count"         not in st.session_state: st.session_state.stream_count         = 0
if "stream_queue"         not in st.session_state: st.session_state.stream_queue         = None
if "stop_event"           not in st.session_state: st.session_state.stop_event           = None
if "stream_trends"        not in st.session_state: st.session_state.stream_trends        = None
if "stream_df_clustered"  not in st.session_state: st.session_state.stream_df_clustered  = None
if "stream_vectorizer"    not in st.session_state: st.session_state.stream_vectorizer    = None
if "stream_matrix"        not in st.session_state: st.session_state.stream_matrix        = None
if "stream_rf"            not in st.session_state: st.session_state.stream_rf            = None
if "stream_cv_acc"        not in st.session_state: st.session_state.stream_cv_acc        = None
if "stream_last_ml_count" not in st.session_state: st.session_state.stream_last_ml_count = 0


# ─── Reset si paramètres changent ────────────────────────────────────────────

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


# ─── Cache ───────────────────────────────────────────────────────────────────

if cache_btn:
    cached = load_cached()
    if cached is not None:
        st.session_state.df = cached
        st.success(f"Cache chargé : {len(cached)} posts")
    else:
        st.warning("Aucun cache disponible. Lancez d'abord une analyse.")


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _run_pipeline(df: pd.DataFrame, nc: int = None):
    nc = nc or n_clusters
    df_clean, vectorizer, matrix = preprocess(df)
    trends, df_clustered, kmeans = detect_trends(df_clean, matrix, vectorizer, nc)
    rf, cv_acc = train_classifier(df_clustered, matrix)
    return df_clean, df_clustered, vectorizer, matrix, trends, kmeans, rf, cv_acc


def _build_stream_keywords():
    if stream_filter_mode == "Aucun filtre (tout accepter)":
        return None
    raw = stream_keywords_raw.strip()
    if not raw:
        return None
    kws = [k.strip() for k in raw.split(",") if k.strip()]
    return kws if kws else None


def _run_stream_ml(buffer: list, nc: int) -> bool:
    """Lance KMeans + RF sur le buffer streaming, met à jour session_state."""
    try:
        df_stream = pd.DataFrame(buffer)
        if "text" not in df_stream.columns:
            return False

        df_clean, vectorizer, matrix = preprocess(df_stream)
        effective_nc = min(nc, max(2, len(df_stream) // 5))
        trends, df_clustered, kmeans = detect_trends(df_clean, matrix, vectorizer, effective_nc)
        rf, cv_acc = train_classifier(df_clustered, matrix)

        st.session_state.stream_trends        = trends
        st.session_state.stream_df_clustered  = df_clustered
        st.session_state.stream_vectorizer    = vectorizer
        st.session_state.stream_matrix        = matrix
        st.session_state.stream_rf            = rf
        st.session_state.stream_cv_acc        = cv_acc
        st.session_state.stream_last_ml_count = len(buffer)
        save_models(vectorizer, kmeans, rf)
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


# ─── Mode Streaming ──────────────────────────────────────────────────────────

if mode == "🔴 Streaming (polling)":

    st.markdown(
        '<div class="stream-status">🔴 LIVE · Polling Reddit toutes les 10s…</div>',
        unsafe_allow_html=True,
    )

    _kws_preview = _build_stream_keywords()
    if _kws_preview is None:
        st.info("💡 Mode : **aucun filtre** — tous les posts des subreddits sont collectés.")
    else:
        st.info(f"💡 Mode : **filtre actif** — mots-clés : {', '.join(_kws_preview)}")

    st.divider()

    col_start, col_stop = st.columns(2)
    start_stream = col_start.button("▶️ Démarrer", use_container_width=True, type="primary")
    stop_stream  = col_stop.button("⏹ Arrêter",   use_container_width=True)

    # ── Worker thread ─────────────────────────────────────────────────────────
    def _stream_worker(q, subs, kws, n, stop_event):
        try:
            for post in stream_posts(subs, keywords=kws, max_posts=n, poll_interval=10):
                if stop_event.is_set():
                    break
                q.put(post)
        except Exception as e:
            q.put({"__error__": str(e)})

    # ── Démarrage ─────────────────────────────────────────────────────────────
    if start_stream and not st.session_state.stream_running:
        st.session_state.stream_buffer        = []
        st.session_state.stream_count         = 0
        st.session_state.stream_running       = True
        st.session_state.stream_trends        = None
        st.session_state.stream_df_clustered  = None
        st.session_state.stream_vectorizer    = None
        st.session_state.stream_matrix        = None
        st.session_state.stream_rf            = None
        st.session_state.stream_cv_acc        = None
        st.session_state.stream_last_ml_count = 0

        q          = queue.Queue()
        stop_event = threading.Event()
        st.session_state.stream_queue = q
        st.session_state.stop_event   = stop_event

        kws = _build_stream_keywords()
        threading.Thread(
            target=_stream_worker,
            args=(q, subreddits, kws, max_posts, stop_event),
            daemon=True,
        ).start()

        msg = "✅ Streaming démarré — collecte de TOUS les posts." if kws is None \
              else f"✅ Streaming démarré — filtre : {', '.join(kws)}"
        st.success(msg)
        time.sleep(1)
        st.rerun()

    # ── Arrêt ─────────────────────────────────────────────────────────────────
    if stop_stream:
        if st.session_state.stop_event is not None:
            st.session_state.stop_event.set()
        st.session_state.stream_running = False
        st.success("⏹ Streaming arrêté.")
        st.rerun()

    # ── Boucle d'affichage live ───────────────────────────────────────────────
    if st.session_state.stream_running:

        # Drainer la queue → buffer
        q = st.session_state.stream_queue
        if q is not None:
            drained = 0
            while not q.empty() and drained < 200:
                item = q.get_nowait()
                if isinstance(item, dict) and "__error__" in item:
                    st.error(f"❌ Erreur streaming : {item['__error__']}")
                    st.session_state.stream_running = False
                    break
                st.session_state.stream_buffer.append(item)
                st.session_state.stream_count += 1
                drained += 1

        total    = st.session_state.stream_count
        last_ml  = st.session_state.stream_last_ml_count

        # ── Déclenchement automatique ML ──────────────────────────────────────
        if total >= stream_ml_threshold and (total - last_ml) >= 10:
            with st.spinner(f"🧠 Analyse ML sur {total} posts…"):
                _run_stream_ml(st.session_state.stream_buffer, stream_n_clusters)

        has_ml = st.session_state.stream_trends is not None

        # ── KPIs ──────────────────────────────────────────────────────────────
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("📨 Posts reçus", total)

        if has_ml:
            k2.metric("🔵 Clusters", len(st.session_state.stream_trends))
            acc = st.session_state.stream_cv_acc
            k3.metric("🌲 RF Accuracy (CV)", f"{acc:.0%}" if acc and not pd.isna(acc) else "—")
            buf_df = pd.DataFrame(st.session_state.stream_buffer)
            k4.metric("💬 Engagement total",
                      int(buf_df["score"].sum() + buf_df["num_comments"].sum())
                      if "score" in buf_df.columns else "—")
        else:
            k2.metric("🧠 ML démarre dans", f"{max(0, stream_ml_threshold - total)} posts")
            k3.metric("🌲 RF Accuracy", "—")
            k4.metric("💬 Engagement", "—")

        st.divider()

        # ── Tabs ──────────────────────────────────────────────────────────────
        if st.session_state.stream_buffer:
            stream_df = pd.DataFrame(st.session_state.stream_buffer)

            if has_ml:
                tab_live, tab_clusters, tab_rf, tab_visu, tab_raw = st.tabs([
                    "📋 Posts live", "🔵 Clusters", "🌲 Random Forest",
                    "📊 Visualisations", "🗂️ Données brutes",
                ])
            else:
                tab_live, tab_raw = st.tabs(["📋 Posts live", "🗂️ Données brutes"])

            # ── Posts live ────────────────────────────────────────────────────
            with tab_live:
                st.caption(f"20 derniers posts sur {total} reçus")
                if has_ml:
                    try:
                        recent = stream_df.tail(20).copy().reset_index(drop=True)
                        classified = classify_new_posts(
                            recent["text"].tolist(),
                            st.session_state.stream_vectorizer,
                            st.session_state.stream_rf,
                            st.session_state.stream_trends,
                        )
                        recent["🏷️ Cluster"]   = classified["topic_label"].values
                        recent["🎯 Confiance"] = classified["confidence"].apply(lambda x: f"{x:.0%}").values
                        cols = [c for c in ["title", "subreddit", "score", "num_comments", "🏷️ Cluster", "🎯 Confiance"] if c in recent.columns]
                        st.dataframe(recent[cols], use_container_width=True)
                    except Exception:
                        cols = [c for c in ["title", "subreddit", "score", "num_comments", "created_at"] if c in stream_df.columns]
                        st.dataframe(stream_df[cols].tail(20), use_container_width=True)
                else:
                    cols = [c for c in ["title", "subreddit", "score", "num_comments", "created_at"] if c in stream_df.columns]
                    st.dataframe(stream_df[cols].tail(20), use_container_width=True)
                    st.info(f"⏳ L'analyse ML démarre automatiquement à {stream_ml_threshold} posts ({total}/{stream_ml_threshold})")

            # ── Clusters ──────────────────────────────────────────────────────
            if has_ml:
                trends_s = st.session_state.stream_trends
                df_cls   = st.session_state.stream_df_clustered
                vec_s    = st.session_state.stream_vectorizer
                rf_s     = st.session_state.stream_rf
                mat_s    = st.session_state.stream_matrix
                acc_s    = st.session_state.stream_cv_acc

                with tab_clusters:
                    st.caption(f"KMeans · {total} posts · {len(trends_s)} clusters")

                    st.subheader("📊 Résumé des clusters")
                    st.dataframe(
                        trends_s[["label", "keywords", "post_count", "avg_score", "avg_comments", "engagement_score"]],
                        use_container_width=True,
                    )
                    st.divider()

                    st.subheader("🔥 Tendances par cluster")
                    st.plotly_chart(trend_bar_chart(trends_s), use_container_width=True)
                    st.divider()

                    st.subheader("🔍 Posts d'un cluster")
                    selected_c = st.selectbox(
                        "Choisir un cluster",
                        trends_s["cluster"].tolist(),
                        format_func=lambda c: f"Cluster {c} — {trends_s.loc[trends_s['cluster']==c, 'label'].values[0]}",
                        key="stream_cluster_select",
                    )
                    cluster_posts = df_cls[df_cls["cluster"] == selected_c][
                        ["title", "subreddit", "score", "num_comments"]
                    ].head(15)
                    st.dataframe(cluster_posts, use_container_width=True)

                # ── Random Forest ─────────────────────────────────────────────
                with tab_rf:
                    col_info, col_imp = st.columns([2, 3])
                    with col_info:
                        st.subheader("📋 Résumé du modèle")
                        st.markdown(f"""
- **Algorithme** : Random Forest
- **Estimators** : 200 arbres
- **Posts d'entraînement** : {total}
- **Features TF-IDF** : {mat_s.shape[1]}
- **Clusters** : {len(trends_s)}
- **Accuracy CV-3** : {f"{acc_s:.1%}" if acc_s and not pd.isna(acc_s) else "N/A"}
                        """)
                    with col_imp:
                        st.subheader("🏆 Features importantes")
                        importance_df = get_feature_importance(rf_s, vec_s)
                        st.plotly_chart(feature_importance_chart(importance_df), use_container_width=True)

                    st.divider()
                    st.subheader("🎯 Distribution de confiance")
                    try:
                        classified_all = classify_new_posts(df_cls["text"].tolist(), vec_s, rf_s, trends_s)
                        classified_all["post_score"] = df_cls["score"].values
                        st.plotly_chart(confidence_histogram(classified_all), use_container_width=True)
                        st.dataframe(
                            classified_all[["text", "topic_label", "confidence", "post_score"]]
                            .sort_values("confidence", ascending=False)
                            .head(25),
                            use_container_width=True,
                        )
                    except Exception as e:
                        st.warning(f"Distribution indisponible : {e}")

                # ── Visualisations ────────────────────────────────────────────
                with tab_visu:
                    st.subheader("📊 Engagement par cluster")
                    try:
                        st.plotly_chart(engagement_scatter(df_cls, trends_s), use_container_width=True)
                    except Exception:
                        st.info("Pas assez de données pour le scatter.")

                    st.divider()
                    st.subheader("📅 Timeline des posts")
                    try:
                        st.plotly_chart(timeline_chart(stream_df), use_container_width=True)
                    except Exception:
                        st.info("Timeline indisponible.")

                    st.divider()
                    col_v1, col_v2 = st.columns(2)
                    with col_v1:
                        flairs = extract_flairs(stream_df)
                        if not flairs.empty:
                            st.subheader("🏷️ Flairs")
                            st.plotly_chart(flair_chart(flairs), use_container_width=True)
                    with col_v2:
                        hashtags = extract_hashtags(stream_df)
                        if not hashtags.empty:
                            st.subheader("#️⃣ Hashtags")
                            st.plotly_chart(hashtag_chart(hashtags), use_container_width=True)

                    st.divider()
                    st.subheader("☁️ Nuage de mots global")
                    try:
                        img = wordcloud_image(df_cls["clean_text"].tolist())
                        if img:
                            st.image(f"data:image/png;base64,{img}", use_container_width=True)
                    except Exception:
                        st.info("Nuage de mots indisponible.")

                    st.divider()
                    st.subheader("🔍 Nuage par cluster")
                    selected_c2 = st.selectbox(
                        "Cluster",
                        trends_s["cluster"].tolist(),
                        format_func=lambda c: f"Cluster {c} — {trends_s.loc[trends_s['cluster']==c, 'label'].values[0]}",
                        key="stream_wc_cluster",
                    )
                    try:
                        img_c = wordcloud_per_cluster(df_cls, trends_s, selected_c2)
                        if img_c:
                            st.image(f"data:image/png;base64,{img_c}", use_container_width=True)
                    except Exception:
                        st.info("Nuage par cluster indisponible.")

            # ── Données brutes ────────────────────────────────────────────────
            with tab_raw:
                st.caption(f"Tous les posts reçus ({total})")
                cols = [c for c in ["title", "subreddit", "score", "num_comments", "created_at", "flair", "url"] if c in stream_df.columns]
                st.dataframe(stream_df[cols], use_container_width=True)
                csv = stream_df.to_csv(index=False).encode("utf-8")
                st.download_button("⬇️ Télécharger CSV complet", csv, "stream_posts.csv", "text/csv")

        else:
            st.info("⏳ En attente des premiers posts…")

        # Auto-refresh toutes les 5 secondes
        time.sleep(5)
        st.rerun()


# ─── Dashboard principal (batch) ─────────────────────────────────────────────

if st.session_state.df is not None:
    df         = st.session_state.df
    df_clean   = st.session_state.df_clean
    trends     = st.session_state.trends
    vectorizer = st.session_state.vectorizer
    matrix     = st.session_state.matrix
    rf         = st.session_state.rf
    cv_acc     = st.session_state.cv_acc

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

    tab1, tab2, tab3, tab4 = st.tabs([
        "🔥 Tendances", "🌲 Random Forest", "📊 Exploration", "🗂️ Données"
    ])

    with tab1:
        col_a, col_b = st.columns([3, 2])
        with col_a:
            st.plotly_chart(trend_bar_chart(trends), use_container_width=True)
        with col_b:
            flairs   = extract_flairs(df)
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
            st.image(f"data:image/png;base64,{img}", use_container_width=True)

        st.subheader("🔍 Nuage par cluster")
        selected_cluster = st.selectbox(
            "Choisir un cluster",
            trends["cluster"].tolist(),
            format_func=lambda c: f"Cluster {c} — {trends.loc[trends['cluster']==c, 'label'].values[0]}",
        )
        img_c = wordcloud_per_cluster(df_clean, trends, selected_cluster)
        if img_c:
            st.image(f"data:image/png;base64,{img_c}", use_container_width=True)

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
            classified_all = classify_new_posts(df_clean["text"].tolist(), vectorizer, rf, trends)
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

    with tab3:
        st.plotly_chart(engagement_scatter(df_clean, trends), use_container_width=True)
        st.subheader("📊 Statistiques par cluster")
        st.dataframe(
            trends[["label", "keywords", "post_count", "avg_score", "avg_comments", "engagement_score"]],
            use_container_width=True,
        )

    with tab4:
        st.subheader(f"🗂️ Posts collectés ({len(df)})")
        cols_show = [c for c in ["title", "subreddit", "score", "num_comments", "created_at", "flair"] if c in df.columns]
        st.dataframe(df[cols_show], use_container_width=True)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Télécharger CSV", csv, "reddit_posts.csv", "text/csv")


# ─── Classifieur ad hoc (sidebar) ────────────────────────────────────────────

if classify_btn and custom_text.strip():
    models   = load_models()
    rf_m     = st.session_state.stream_rf or st.session_state.rf or (models["rf"] if models else None)
    vec_m    = st.session_state.stream_vectorizer or st.session_state.vectorizer or (models["vectorizer"] if models else None)
    trends_m = st.session_state.stream_trends or st.session_state.trends

    if rf_m is None or vec_m is None:
        st.sidebar.warning("Aucun modèle disponible. Lancez une analyse ou attendez le seuil ML en streaming.")
    elif trends_m is None:
        st.sidebar.warning("Labels de clusters manquants. Relancez une analyse.")
    else:
        result = classify_new_posts([custom_text], vec_m, rf_m, trends_m)
        row    = result.iloc[0]
        st.sidebar.success(f"**Sujet** : {row['topic_label']}")
        st.sidebar.metric("Confiance RF", f"{row['confidence']:.0%}")


# ─── Footer ──────────────────────────────────────────────────────────────────
st.divider()
st.caption("TrendRadar · Reddit · KMeans + Random Forest · JSON public · Sans clé API")