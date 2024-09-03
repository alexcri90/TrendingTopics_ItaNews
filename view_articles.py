import json
import csv
import os
from datetime import datetime

def load_articles(file_path):
    """
    Load articles from either JSON or CSV file.
    """
    _, ext = os.path.splitext(file_path)
    if ext.lower() == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif ext.lower() == '.csv':
        with open(file_path, 'r', encoding='utf-8') as f:
            return list(csv.DictReader(f))
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def display_article_summary(article, index):
    """
    Display a summary of a single article.
    """
    print(f"\nArticle {index + 1}:")
    print(f"Title: {article.get('title', 'N/A')}")
    print(f"Source: {article.get('source_name', 'N/A')}")
    print(f"Published: {article.get('publishedAt', 'N/A')}")
    print(f"Description: {article.get('description', 'N/A')[:100]}...")  # First 100 characters of description

def main():
    # Get the most recent file in the data directory
    data_dir = 'data'
    files = [f for f in os.listdir(data_dir) if f.startswith('articles_') and (f.endswith('.json') or f.endswith('.csv'))]
    if not files:
        print("No article files found.")
        return

    latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(data_dir, f)))
    file_path = os.path.join(data_dir, latest_file)

    print(f"Viewing articles from: {file_path}")

    try:
        articles = load_articles(file_path)
        print(f"\nTotal articles: {len(articles)}")

        # Display summaries of the first 5 articles
        for i, article in enumerate(articles[:5]):
            display_article_summary(article, i)

        # Option to view more
        while True:
            choice = input("\nDo you want to see 5 more articles? (y/n): ").lower()
            if choice != 'y':
                break
            for i, article in enumerate(articles[i+1:i+6], start=i+1):
                display_article_summary(article, i)
            if i+1 >= len(articles):
                print("No more articles to display.")
                break

    except Exception as e:
        print(f"Error loading articles: {e}")

if __name__ == "__main__":
    main()