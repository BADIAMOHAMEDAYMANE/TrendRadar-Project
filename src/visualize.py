import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from wordcloud import WordCloud
import base64
from io import BytesIO

def trend_bar_chart(trends_df: pd.DataFrame):
    fig = px.bar(
        trends_df,
        x="score",
        y="keywords",
        orientation="h",
        color="tweet_count",
        color_continuous_scale="Viridis",
        labels={"score": "Score d'engagement", "keywords": "Tendance", "tweet_count": "Tweets"},
        title="🔥 Tendances émergentes",
    )
    fig.update_layout(yaxis=dict(autorange="reversed"), height=400)
    return fig

def hashtag_chart(hashtags_df: pd.DataFrame):
    fig = px.treemap(
        hashtags_df,
        path=["hashtag"],
        values="count",
        title="Hashtags les plus utilisés",
        color="count",
        color_continuous_scale="Blues",
    )
    return fig

def wordcloud_image(texts: list[str]) -> str:
    combined = " ".join(texts)
    wc = WordCloud(width=800, height=400, background_color="white", colormap="plasma").generate(combined)
    buf = BytesIO()
    wc.to_image().save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()