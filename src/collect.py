import pandas as pd
import random
from datetime import datetime, timedelta

TWEETS_DATA = [
    "L'intelligence artificielle transforme le marché du travail #IA #emploi",
    "ChatGPT révolutionne la façon de coder au quotidien #dev #AI #OpenAI",
    "Le machine learning appliqué à la santé sauve des vies #santé #ML",
    "GPT-5 vient d'être annoncé, les performances sont incroyables #OpenAI",
    "Débat sur la régulation de l'IA en Europe #EUAIAct #tech #politique",
    "Python reste le langage numéro 1 en data science #python #datascience",
    "Les LLMs changent complètement notre rapport à la recherche #IA #Google",
    "Anthropic publie Claude 4, un nouveau bond en avant #Claude #Anthropic",
    "La cybersécurité devient priorité absolue pour les entreprises #cybersec",
    "Le cloud computing domine les infrastructures modernes #AWS #Azure",
    "Les startups IA européennes lèvent des milliards #startup #VC #deeptech",
    "Mistral AI, la pépite française de l'intelligence artificielle #Mistral",
    "Data privacy : le RGPD insuffisant face aux nouvelles IA #RGPD #privacy",
    "Quantum computing : IBM annonce un processeur révolutionnaire #quantum",
    "Les voitures autonomes arrivent enfin en Europe #tesla #autonomous #tech",
    "TikTok et l'IA générative changent la création de contenu #socialmedia",
    "Le No-code et l'IA démocratisent le développement logiciel #nocode",
    "Meta lance un nouveau modèle open source très puissant #Meta #LLaMA",
    "L'IA dans l'éducation : révolution ou menace ? #edtech #IA #école",
    "Google Gemini face à ChatGPT : la guerre des LLMs continue #Google #AI",
]

def collect_tweets(query: str, max_results: int = 100) -> pd.DataFrame:
    random.seed(42)
    records = []
    keywords = query.lower().split()
    tag = keywords[0] if keywords else "tech"

    for i in range(min(max_results, 80)):
        base = random.choice(TWEETS_DATA)
        extra_tags = random.sample(["#IA", "#tech", "#innovation", "#digital", "#futur"], 2)
        records.append({
            "id": i,
            "text": f"{base} #{tag} {' '.join(extra_tags)}",
            "created_at": datetime.now() - timedelta(minutes=random.randint(1, 2880)),
            "likes": random.randint(0, 1000),
            "retweets": random.randint(0, 400),
        })

    df = pd.DataFrame(records)
    df.to_parquet("data/tweets_demo.parquet")
    return df