import requests
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
from datetime import datetime

def fetch_and_analyze_sentiment(api_key, data_dir):
    print("Loading FinBERT model...")
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    print("Model loaded.")
    print("Fetching news from MarketAux API...")
    url = f"https://api.marketaux.com/v1/news/all?api_token={api_key}&language=en"
    try:
        response = requests.get(url)
        response.raise_for_status()
        news_data = response.json()
        articles = news_data.get('data', [])
        print(f"Found {len(articles)} new articles.")
        sentiments = []
        for entry in articles:
            title = entry.get('title')
            published_time_str = entry.get('published_at')
            published_time = datetime.fromisoformat(published_time_str)
            inputs = tokenizer(title, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            sentiment_score = probs[0][0].item() - probs[0][1].item()
            sentiments.append({'timestamp_utc': published_time, 'headline': title, 'sentiment_score': sentiment_score})
            print(f"Analyzed '{title[:50]}...': Score={sentiment_score:.2f}")
        if sentiments:
            df = pd.DataFrame(sentiments)
            df.set_index('timestamp_utc', inplace=True)
            df.sort_index(inplace=True)
            os.makedirs(data_dir, exist_ok=True)
            filepath = os.path.join(data_dir, "news_sentiment.parquet")
            df.to_parquet(filepath)
            print(f"Created new sentiment file at {filepath}")
        else:
            print("No new articles to analyze.")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching news from MarketAux: {e}")

if __name__ == '__main__':
    MARKETAUX_API_KEY = 'HR084l0Ck2EF04063RFy71YzcSjcAH7vpPtjGr1d'
    DATA_DIRECTORY = 'data/raw/sentiment'
    fetch_and_analyze_sentiment(MARKETAUX_API_KEY, DATA_DIRECTORY)
