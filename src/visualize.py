"""
visualize.py — Graphiques Plotly et WordCloud pour TrendRadar Reddit
"""

import base64
from io import BytesIO

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud


# ─── Palette cohérente ───────────────────────────────────────────────────────

PALETTE = px.colors.sequential.Plasma
BG = "rgba(0,0,0,0)"        # transparent pour s'adapter au thème Streamlit
FONT = "IBM Plex Mono"


def _base_layout(fig: go.Figure, title: str) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, family=FONT)),
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        font=dict(family=FONT),
        margin=dict(l=10, r=10, t=45, b=10),
    )
    return fig


# ─── Tendances ───────────────────────────────────────────────────────────────

def trend_bar_chart(trends_df: pd.DataFrame) -> go.Figure:
    """
    Barres horizontales : engagement par cluster.
    Couleur = nombre de posts.
    """
    fig = px.bar(
        trends_df,
        x="engagement_score",
        y="label",
        orientation="h",
        color="post_count",
        color_continuous_scale=PALETTE,
        text="post_count",
        labels={
            "engagement_score": "Score d'engagement",
            "label": "Sujet",
            "post_count": "Posts",
        },
    )
    fig.update_traces(texttemplate="%{text} posts", textposition="outside")
    fig.update_layout(yaxis=dict(autorange="reversed"), height=420)
    return _base_layout(fig, "🔥 Tendances détectées")


def engagement_scatter(df: pd.DataFrame, trends_df: pd.DataFrame) -> go.Figure:
    """
    Scatter plot score vs commentaires, coloré par cluster.
    Révèle les posts viraux dans chaque thème.
    """
    df = df.copy()
    label_map = dict(zip(trends_df["cluster"], trends_df["label"]))
    df["topic"] = df["cluster"].map(label_map)

    fig = px.scatter(
        df,
        x="score",
        y="num_comments",
        color="topic",
        hover_data=["title", "subreddit"] if "title" in df.columns else None,
        size="score",
        size_max=30,
        opacity=0.7,
        color_discrete_sequence=px.colors.qualitative.Bold,
        labels={"score": "Score Reddit", "num_comments": "Commentaires"},
    )
    fig.update_layout(height=400)
    return _base_layout(fig, "💬 Score vs Commentaires par sujet")


def feature_importance_chart(importance_df: pd.DataFrame) -> go.Figure:
    """Barres horizontales des features les plus importantes du Random Forest."""
    fig = px.bar(
        importance_df.sort_values("importance"),
        x="importance",
        y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale="Teal",
        labels={"importance": "Importance", "feature": "Terme"},
    )
    fig.update_layout(height=420, showlegend=False)
    return _base_layout(fig, "🌲 Termes clés — Random Forest")


def flair_chart(flairs_df: pd.DataFrame) -> go.Figure:
    """Treemap des flairs Reddit."""
    if flairs_df.empty:
        return go.Figure().add_annotation(text="Aucun flair disponible", showarrow=False)

    fig = px.treemap(
        flairs_df,
        path=["flair"],
        values="count",
        color="count",
        color_continuous_scale="Blues",
    )
    return _base_layout(fig, "🏷️ Flairs Reddit")


def hashtag_chart(hashtags_df: pd.DataFrame) -> go.Figure:
    """Treemap des hashtags."""
    if hashtags_df.empty:
        return go.Figure().add_annotation(text="Aucun hashtag détecté", showarrow=False)

    fig = px.treemap(
        hashtags_df,
        path=["hashtag"],
        values="count",
        color="count",
        color_continuous_scale="Oranges",
    )
    return _base_layout(fig, "# Hashtags")


def timeline_chart(df: pd.DataFrame) -> go.Figure:
    """
    Volume de posts par heure sur la période collectée.
    """
    if "created_at" not in df.columns:
        return go.Figure()

    ts = df.set_index("created_at").resample("1h").size().reset_index()
    ts.columns = ["heure", "posts"]

    fig = px.area(
        ts, x="heure", y="posts",
        color_discrete_sequence=["#7C3AED"],
        labels={"heure": "Heure", "posts": "Posts"},
    )
    fig.update_traces(fillcolor="rgba(124,58,237,0.15)", line_color="#7C3AED")
    fig.update_layout(height=220)
    return _base_layout(fig, "📈 Volume de posts dans le temps")


def confidence_histogram(classified_df: pd.DataFrame) -> go.Figure:
    """Distribution des scores de confiance du Random Forest."""
    fig = px.histogram(
        classified_df,
        x="confidence",
        color="topic_label",
        nbins=20,
        barmode="overlay",
        opacity=0.7,
        labels={"confidence": "Confiance RF", "topic_label": "Sujet"},
    )
    fig.update_layout(height=300)
    return _base_layout(fig, "🎯 Confiance du classifieur")


# ─── WordCloud ───────────────────────────────────────────────────────────────

def wordcloud_image(
    texts: list[str],
    colormap: str = "plasma",
    width: int = 900,
    height: int = 380,
) -> str:
    """
    Génère un WordCloud et retourne l'image encodée en base64.
    À utiliser avec : st.image(f"data:image/png;base64,{img_b64}")
    """
    combined = " ".join(texts)
    if not combined.strip():
        return ""

    wc = WordCloud(
        width=width,
        height=height,
        background_color="white",
        colormap=colormap,
        max_words=120,
        collocations=True,
        prefer_horizontal=0.8,
    ).generate(combined)

    buf = BytesIO()
    wc.to_image().save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def wordcloud_per_cluster(
    df: pd.DataFrame,
    trends_df: pd.DataFrame,
    cluster_id: int,
) -> str:
    """WordCloud pour un cluster spécifique."""
    texts = df[df["cluster"] == cluster_id]["clean_text"].tolist()
    return wordcloud_image(texts, colormap="viridis")