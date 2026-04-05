import streamlit as st
import pandas as pd
import time
from src.collect import collect_tweets
from src.preprocess import preprocess
from src.model import detect_trends, extract_hashtags
from src.visualize import trend_bar_chart, hashtag_chart, wordcloud_image

st.set_page_config(page_title="TrendRadar", page_icon="📡", layout="wide")
st.title("📡 TrendRadar — Détection de tendances Twitter")

# Sidebar
with st.sidebar:
    st.header("⚙️ Paramètres")
    query = st.text_input("Sujet à surveiller", value="intelligence artificielle")
    max_tweets = st.slider("Nombre de tweets", 50, 500, 200)
    n_clusters = st.slider("Nombre de tendances", 3, 10, 5)
    auto_refresh = st.toggle("Actualisation auto (5 min)")
    run = st.button("🚀 Analyser", use_container_width=True)

# Main
if run:
    with st.spinner("Collecte des tweets..."):
        df = collect_tweets(query, max_tweets)
    
    if df.empty:
        st.error("Aucun tweet trouvé. Vérifiez votre clé API ou le sujet.")
        st.stop()
    
    with st.spinner("Analyse en cours..."):
        df_clean, vectorizer, matrix = preprocess(df)
        trends = detect_trends(df_clean, matrix, vectorizer, n_clusters)
        hashtags = extract_hashtags(df)
    
    # Métriques
    col1, col2, col3 = st.columns(3)
    col1.metric("Tweets analysés", len(df))
    col2.metric("Tendances détectées", len(trends))
    col3.metric("Engagement total", int(df["likes"].sum() + df["retweets"].sum()))
    
    st.divider()
    
    # Graphiques
    col_left, col_right = st.columns([3, 2])
    with col_left:
        st.plotly_chart(trend_bar_chart(trends), use_container_width=True)
    with col_right:
        st.plotly_chart(hashtag_chart(hashtags), use_container_width=True)
    
    # Word cloud
    st.subheader("☁️ Nuage de mots")
    img_b64 = wordcloud_image(df_clean["clean_text"].tolist())
    st.image(f"data:image/png;base64,{img_b64}", use_column_width=True)
    
    # Tableau des tendances
    st.subheader("📊 Détail des tendances")
    st.dataframe(trends, use_container_width=True)

# Auto-refresh
if auto_refresh:
    time.sleep(300)
    st.rerun()