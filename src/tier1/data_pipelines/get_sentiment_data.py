import feedparser
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
from datetime import datetime

def fetch_and_analyze_sentiment(rss_url, data_dir):
    """
    Fetches news headlines from an RSS feed, analyzes their sentiment using
    a financial NLP model, and saves the results.

    Args:
        rss_url (str): The URL of the RSS feed.
        data_dir (str): The directory to save the sentiment data.
    """
    # 1. Load Pre-trained Financial NLP Model (FinBERT)
    print("Loading FinBERT model...")
    tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
    model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
    print("Model loaded.")

    # 2. Fetch News from RSS Feed
    print(f"Fetching news from {rss_url}...")
    feed = feedparser.parse(rss_url)
    print(f"Found {len(feed.entries)} new articles.")

    # 3. Analyze Sentiment of Each Article
    sentiments = []
    for entry in feed.entries:
        title = entry.title
        published_time = datetime.strptime(entry.published, '%a, %d %b %Y %H:%M:%S %z')

        # Tokenize and predict sentiment
        inputs = tokenizer(title, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)

        # The model outputs logits for positive, negative, neutral.
        # We use softmax to get probabilities.
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # We can create a single sentiment score: positive_prob - negative_prob
        # This gives a score from -1 (very negative) to +1 (very positive)
        sentiment_score = probs[0][0].item() - probs[0][1].item()

        sentiments.append({
            'timestamp_utc': published_time,
            'headline': title,
            'sentiment_score': sentiment_score
        })
        print(f"Analyzed '{title[:50]}...': Score={sentiment_score:.2f}")

    # 4. Save to DataFrame and Parquet file
    if sentiments:
        df = pd.DataFrame(sentiments)
        df.set_index('timestamp_utc', inplace=True)
        df.sort_index(inplace=True)

        os.makedirs(data_dir, exist_ok=True)
        filename = "news_sentiment.parquet"
        filepath = os.path.join(data_dir, filename)

        # Append to existing file or create new one
        if os.path.exists(filepath):
            existing_df = pd.read_parquet(filepath)
            combined_df = pd.concat([existing_df, df])
            # Remove duplicates, keeping the first instance of a headline
            combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
            combined_df.to_parquet(filepath)
            print(f"Appended {len(df)} new sentiment records to {filepath}")
        else:
            df.to_parquet(filepath)
            print(f"Created new sentiment file at {filepath}")
    else:
        print("No new articles to analyze.")


if __name__ == '__main__':
    # Switched to Reuters Business News RSS feed for more relevant financial headlines
    RSS_FEED_URL = "https://www.reuters.com/arc/outboundfeeds/rss/category/businessNews"
    DATA_DIRECTORY = 'data/raw/sentiment'

    fetch_and_analyze_sentiment(RSS_FEED_URL, DATA_DIRECTORY)
