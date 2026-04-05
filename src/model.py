import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
import re

def detect_trends(df: pd.DataFrame, matrix, vectorizer, n_clusters: int = 5) -> pd.DataFrame:
    n_clusters = min(n_clusters, len(df))
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df = df.copy()
    df["cluster"] = kmeans.fit_predict(matrix)
    
    feature_names = vectorizer.get_feature_names_out()
    trends = []
    
    for cluster_id in range(n_clusters):
        center = kmeans.cluster_centers_[cluster_id]
        top_indices = center.argsort()[-10:][::-1]
        keywords = [feature_names[i] for i in top_indices]
        
        cluster_df = df[df["cluster"] == cluster_id]
        trends.append({
            "cluster": cluster_id,
            "keywords": ", ".join(keywords[:5]),
            "tweet_count": len(cluster_df),
            "avg_likes": round(cluster_df["likes"].mean(), 1),
            "avg_retweets": round(cluster_df["retweets"].mean(), 1),
            "score": round(cluster_df["likes"].sum() + cluster_df["retweets"].sum() * 2, 0),
        })
    
    return pd.DataFrame(trends).sort_values("score", ascending=False).reset_index(drop=True)

def extract_hashtags(df: pd.DataFrame) -> pd.DataFrame:
    all_tags = []
    for text in df["text"]:
        all_tags.extend(re.findall(r"#(\w+)", text.lower()))
    counts = Counter(all_tags)
    return pd.DataFrame(counts.most_common(20), columns=["hashtag", "count"])