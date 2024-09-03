import os
import requests
from collections import Counter
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
API_KEY = os.getenv('NEWS_API_KEY')
EVERYTHING_URL = 'https://newsapi.org/v2/everything'
SOURCES_URL = 'https://newsapi.org/v2/top-headlines/sources'

def get_italian_sources():
    """
    Fetch all available Italian news sources.
    """
    params = {
        'apiKey': API_KEY,
        'language': 'it',
        'country': 'it'
    }
    
    try:
        response = requests.get(SOURCES_URL, params=params)
        response.raise_for_status()
        
        all_sources = response.json()['sources']
        return ','.join(source['id'] for source in all_sources)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching sources: {e}")
        return None

def fetch_articles(sources):
    """
    Fetch recent articles from Italian sources to extract authors.
    """
    params = {
        'apiKey': API_KEY,
        'sources': sources,
        'language': 'it',
        'pageSize': 100  # Maximum allowed for free tier
    }
    
    try:
        response = requests.get(EVERYTHING_URL, params=params)
        response.raise_for_status()
        return response.json().get('articles', [])
    except requests.exceptions.RequestException as e:
        print(f"Error fetching articles: {e}")
        return []

def extract_authors(articles):
    """
    Extract and count authors from the articles.
    """
    authors = [article.get('author') for article in articles if article.get('author')]
    return Counter(authors)

def main():
    print("Fetching Italian news sources...")
    sources = get_italian_sources()
    if not sources:
        print("Could not fetch Italian sources. Exiting.")
        return

    print("Fetching recent articles...")
    articles = fetch_articles(sources)
    if not articles:
        print("No articles fetched. Exiting.")
        return

    print("\nAvailable authors (with article count):")
    author_counts = extract_authors(articles)
    for author, count in author_counts.most_common():
        print(f"- {author}: {count} article(s)")

if __name__ == "__main__":
    main()