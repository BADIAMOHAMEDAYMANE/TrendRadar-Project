import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download("stopwords", quiet=True)
STOPWORDS = set(stopwords.words("french")) | set(stopwords.words("english"))

def clean_text(text: str) -> str:
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+|#", "", text)
    text = re.sub(r"[^a-zA-ZÀ-ÿ\s]", " ", text)
    return " ".join(w.lower() for w in text.split() if w.lower() not in STOPWORDS and len(w) > 2)

def preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, TfidfVectorizer, any]:
    df = df.copy()
    df["clean_text"] = df["text"].apply(clean_text)
    df = df[df["clean_text"].str.strip() != ""]
    
    vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(df["clean_text"])
    
    return df, vectorizer, matrix