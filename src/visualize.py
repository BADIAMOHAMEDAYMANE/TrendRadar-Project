"""
visualize.py — Graphiques Plotly, WordCloud et carte géographique pour TrendRadar Reddit
"""

import base64
from io import BytesIO
from collections import Counter
from typing import List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud

# ─── Palette cohérente ───────────────────────────────────────────────────────

PALETTE = px.colors.sequential.Plasma
BG      = "rgba(0,0,0,0)"
FONT    = "IBM Plex Mono"

BURST_COLORS = {
    "unigram":  "#7C3AED",
    "bigram":   "#059669",
    "hashtag":  "#DC2626",
}


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


# ─── Burst ───────────────────────────────────────────────────────────────────

def burst_bar_chart(alerts_df: pd.DataFrame, top_n: int = 20) -> go.Figure:
    if alerts_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="Aucun burst détecté", showarrow=False,
                           xref="paper", yref="paper", x=0.5, y=0.5)
        return _base_layout(fig, "⚡ Termes en burst")

    df = alerts_df.head(top_n).copy()
    df["color"] = df["type"].map(BURST_COLORS).fillna("#64748B")

    fig = go.Figure()
    for term_type, color in BURST_COLORS.items():
        sub = df[df["type"] == term_type]
        if sub.empty:
            continue
        fig.add_trace(go.Bar(
            x=sub["burst_score"],
            y=sub["term"],
            orientation="h",
            name=term_type,
            marker_color=color,
            text=sub["burst_score"].round(2),
            textposition="outside",
        ))

    fig.update_layout(
        barmode="stack",
        yaxis=dict(autorange="reversed"),
        height=max(300, top_n * 22),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        xaxis_title="Score de burst Bt",
    )
    return _base_layout(fig, "⚡ Termes en burst (Bt = (TF − EF) / √EF)")


def burst_timeline_chart(alerts_df: pd.DataFrame) -> go.Figure:
    if alerts_df.empty or "timestamp" not in alerts_df.columns:
        return go.Figure().add_annotation(text="Aucun historique burst", showarrow=False)

    df = alerts_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df_grouped = df.groupby(pd.Grouper(key="timestamp", freq="5min")).agg(
        nb_alerts=("term", "count"),
        top_score=("burst_score", "max"),
    ).reset_index()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_grouped["timestamp"],
        y=df_grouped["nb_alerts"],
        mode="lines+markers",
        name="Nb alertes",
        line=dict(color="#DC2626", width=2),
        marker=dict(size=6),
    ))
    fig.add_trace(go.Scatter(
        x=df_grouped["timestamp"],
        y=df_grouped["top_score"],
        mode="lines",
        name="Score max",
        line=dict(color="#7C3AED", width=1.5, dash="dot"),
        yaxis="y2",
    ))
    fig.update_layout(
        height=280,
        yaxis2=dict(overlaying="y", side="right", title="Score max"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return _base_layout(fig, "⚡ Chronologie des bursts")


def burst_score_scatter(scores_df: pd.DataFrame) -> go.Figure:
    if scores_df.empty:
        return go.Figure().add_annotation(text="Pas de données burst", showarrow=False)

    fig = px.scatter(
        scores_df,
        x="ef",
        y="tf",
        size="burst_score",
        size_max=30,
        color="burst_score",
        color_continuous_scale="Reds",
        hover_data=["term"],
        labels={"ef": "Fréquence attendue (EF)", "tf": "Fréquence observée (TF)"},
    )
    max_val = max(scores_df["ef"].max(), scores_df["tf"].max()) * 1.1
    fig.add_shape(
        type="line", x0=0, y0=0, x1=max_val, y1=max_val,
        line=dict(color="gray", width=1, dash="dash"),
    )
    fig.update_layout(height=380)
    return _base_layout(fig, "📊 TF vs EF — Positions des termes")


# ─── Viralité ────────────────────────────────────────────────────────────────

def virality_distribution(df: pd.DataFrame) -> go.Figure:
    if "viral_prob" not in df.columns:
        return go.Figure().add_annotation(text="Probabilités virales non calculées", showarrow=False)

    fig = px.histogram(
        df,
        x="viral_prob",
        color="viral_label" if "viral_label" in df.columns else None,
        nbins=25,
        barmode="overlay",
        opacity=0.75,
        color_discrete_map={"🔥 Viral": "#DC2626", "Normal": "#64748B"},
        labels={"viral_prob": "Probabilité virale", "viral_label": "Classe"},
    )
    fig.add_vline(x=0.5, line_dash="dash", line_color="orange",
                  annotation_text="Seuil 0.5")
    fig.update_layout(height=300)
    return _base_layout(fig, "🔥 Distribution des probabilités virales")


def virality_scatter(df: pd.DataFrame) -> go.Figure:
    if "viral_prob" not in df.columns:
        return go.Figure()

    fig = px.scatter(
        df,
        x="score",
        y="num_comments",
        color="viral_prob",
        color_continuous_scale="RdYlGn",
        size="viral_prob",
        size_max=20,
        hover_data=["title"] if "title" in df.columns else None,
        labels={
            "score": "Score Reddit",
            "num_comments": "Commentaires",
            "viral_prob": "P(viral)",
        },
    )
    fig.update_layout(height=380)
    return _base_layout(fig, "🔥 Posts par probabilité virale")


def virality_feature_importance_chart(importance_df: pd.DataFrame) -> go.Figure:
    if importance_df.empty:
        return go.Figure().add_annotation(text="Aucune importance disponible", showarrow=False)
    fig = px.bar(
        importance_df.sort_values("importance"),
        x="importance",
        y="feature",
        orientation="h",
        color="importance",
        color_continuous_scale="OrRd",
        labels={"importance": "Importance", "feature": "Feature"},
    )
    fig.update_layout(height=400, showlegend=False)
    return _base_layout(fig, "🏆 Features viralité — XGBoost/LightGBM")


# ─── Carte géographique (NER lieux) ──────────────────────────────────────────

_GEO_CACHE = {
    "france": (46.227638, 2.213749), "paris": (48.8566, 2.3522),
    "lyon": (45.7640, 4.8357), "marseille": (43.2965, 5.3698),
    "bordeaux": (44.8378, -0.5792), "toulouse": (43.6047, 1.4442),
    "nice": (43.7102, 7.2620), "nantes": (47.2184, -1.5536),
    "strasbourg": (48.5734, 7.7521), "lille": (50.6292, 3.0573),
    "montpellier": (43.6108, 3.8767), "rennes": (48.1147, -1.6794),
    "europe": (50.0, 10.0), "belgique": (50.8503, 4.3517),
    "suisse": (46.8182, 8.2275), "allemagne": (51.1657, 10.4515),
    "espagne": (40.4637, -3.7492), "italie": (41.8719, 12.5674),
    "royaume-uni": (55.3781, -3.4360), "london": (51.5074, -0.1278),
    "berlin": (52.5200, 13.4050), "madrid": (40.4168, -3.7038),
    "rome": (41.9028, 12.4964), "amsterdam": (52.3676, 4.9041),
    "états-unis": (37.0902, -95.7129), "usa": (37.0902, -95.7129),
    "canada": (56.1304, -106.3468), "new york": (40.7128, -74.0060),
    "california": (36.7783, -119.4179), "texas": (31.9686, -99.9018),
    "chine": (35.8617, 104.1954), "japon": (36.2048, 138.2529),
    "inde": (20.5937, 78.9629), "russie": (61.5240, 105.3188),
    "brésil": (14.2350, -51.9253), "afrique": (8.7832, 34.5085),
    "maroc": (31.7917, -7.0926), "algérie": (28.0339, 1.6596),
    "tunisie": (33.8869, 9.5375),
}


def _geocode(location: str) -> Optional[tuple]:
    return _GEO_CACHE.get(location.lower().strip())


def geo_map_chart(df: pd.DataFrame) -> go.Figure:
    if "ner_locations" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(
            text="NER désactivé — activez extract_ner=True dans preprocess()",
            showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5,
        )
        return _base_layout(fig, "🗺️ Carte géographique des lieux mentionnés")

    loc_counter: Counter = Counter()
    for loc_list in df["ner_locations"]:
        if isinstance(loc_list, list):
            for loc in loc_list:
                loc_counter[loc.strip()] += 1

    if not loc_counter:
        fig = go.Figure()
        fig.add_annotation(
            text="Aucun lieu détecté par NER dans ce corpus",
            showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5,
        )
        return _base_layout(fig, "🗺️ Carte géographique des lieux mentionnés")

    rows = []
    for loc, count in loc_counter.most_common(50):
        coords = _geocode(loc)
        if coords:
            rows.append({"location": loc, "count": count, "lat": coords[0], "lon": coords[1]})

    if not rows:
        fig = go.Figure()
        top_locs = ", ".join([f"{l} ({c})" for l, c in loc_counter.most_common(10)])
        fig.add_annotation(
            text=f"Lieux détectés (non géocodés) : {top_locs}",
            showarrow=False, xref="paper", yref="paper", x=0.5, y=0.5,
            font=dict(size=12),
        )
        return _base_layout(fig, "🗺️ Lieux mentionnés (non géocodés)")

    geo_df = pd.DataFrame(rows)
    fig = px.scatter_geo(
        geo_df,
        lat="lat",
        lon="lon",
        size="count",
        hover_name="location",
        hover_data={"count": True, "lat": False, "lon": False},
        size_max=40,
        color="count",
        color_continuous_scale="Viridis",
        projection="natural earth",
        labels={"count": "Mentions"},
    )
    fig.update_layout(
        height=450,
        geo=dict(
            showframe=False,
            showcoastlines=True,
            coastlinecolor="rgba(100,100,100,0.4)",
            showland=True,
            landcolor="rgba(230,230,230,0.4)",
            showocean=True,
            oceancolor="rgba(180,210,240,0.3)",
        ),
    )
    return _base_layout(fig, "🗺️ Carte géographique des lieux mentionnés")


def entities_bar_chart(df: pd.DataFrame, entity_type: str = "ner_locations") -> go.Figure:
    title_map = {
        "ner_locations": "📍 Lieux les plus mentionnés",
        "ner_persons":   "👤 Personnes les plus mentionnées",
        "ner_orgs":      "🏢 Organisations les plus mentionnées",
    }
    title = title_map.get(entity_type, "Entités")

    if entity_type not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="NER non disponible", showarrow=False)
        return _base_layout(fig, title)

    counter: Counter = Counter()
    for lst in df[entity_type]:
        if isinstance(lst, list):
            counter.update(lst)

    if not counter:
        fig = go.Figure()
        fig.add_annotation(text="Aucune entité détectée", showarrow=False)
        return _base_layout(fig, title)

    top = pd.DataFrame(counter.most_common(15), columns=["entity", "count"])
    fig = px.bar(
        top,
        x="count", y="entity",
        orientation="h",
        color="count",
        color_continuous_scale="Blues",
        labels={"count": "Occurrences", "entity": ""},
    )
    fig.update_layout(yaxis=dict(autorange="reversed"), height=380, showlegend=False)
    return _base_layout(fig, title)


# ─── WordCloud ───────────────────────────────────────────────────────────────

def wordcloud_image(
    texts: List[str],
    colormap: str = "plasma",
    width: int = 900,
    height: int = 380,
) -> str:
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
    texts = df[df["cluster"] == cluster_id]["clean_text"].tolist()
    return wordcloud_image(texts, colormap="viridis")


def wordcloud_burst(alerts_df: pd.DataFrame) -> str:
    if alerts_df.empty:
        return ""
    text_weighted = " ".join(
        " ".join([row["term"].replace(" ", "_")] * max(1, int(row["burst_score"])))
        for _, row in alerts_df.iterrows()
    )
    if not text_weighted.strip():
        return ""
    wc = WordCloud(
        width=900, height=300,
        background_color="white",
        colormap="Reds",
        max_words=80,
    ).generate(text_weighted)
    buf = BytesIO()
    wc.to_image().save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode() 