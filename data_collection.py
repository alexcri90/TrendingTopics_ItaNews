import os
import json
import requests
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
API_KEY = os.getenv('NEWSDATA_API_KEY')
if not API_KEY:
    raise ValueError("API_KEY not found. Make sure it's set in your .env file.")

BASE_URL = 'https://newsdata.io/api/1/news'
MAX_CREDITS = 30

def load_source_ids():
    """Load source IDs from the list_sources file."""
    with open('Utilities/list_sources', 'r') as f:
        return [line.strip() for line in f if line.strip()]

def fetch_news(domains):
    """Fetch news articles from NewsData.io API for the specified domains."""
    all_articles = []
    credits_used = 0
    next_page = None

    # Join all domains with commas
    domains_param = ','.join(domains)

    params = {
        'apikey': API_KEY,
        'language': 'it',
        'country': 'it',
        'domain': domains_param,
    }

    while True:
        if next_page:
            params['page'] = next_page

        try:
            print(f"Fetching news articles (Credit {credits_used + 1})...")
            print(f"Request URL: {BASE_URL}")
            print(f"Request params: {params}")
            response = requests.get(BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()

            if data['status'] == 'success':
                articles = data['results']
                all_articles.extend(articles)
                print(f"Fetched {len(articles)} articles. Total: {len(all_articles)}")
                credits_used += 1

                next_page = data.get('nextPage')
                if not next_page:
                    print("No more pages available.")
                    break
            else:
                print(f"Error in API response: {data.get('results', {}).get('message', 'Unknown error')}")
                break

        except requests.exceptions.RequestException as e:
            print(f"Error fetching articles: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response content: {e.response.text}")
            break

        if credits_used >= MAX_CREDITS:
            print(f"Reached maximum credits limit ({MAX_CREDITS}).")
            break

    print(f"Total credits used: {credits_used}")
    return all_articles

def save_articles(articles):
    """Save articles to a JSON file only if there are articles to save."""
    if not articles:
        print("No articles to save. Skipping file creation.")
        return

    today = datetime(2024, 10, 24).strftime("%Y-%m-%d")
    # today = datetime.now().strftime("%Y-%m-%d")
    filename = f"data/articles_{today}.json"
    
    os.makedirs("data", exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(articles, f, ensure_ascii=False, indent=4)
    
    print(f"Saved {len(articles)} articles to {filename}")

def main():
    print("Starting data collection process")
    source_ids = load_source_ids()
    articles = fetch_news(source_ids)
    save_articles(articles)
    print("Data collection process completed")

if __name__ == "__main__":
    domains = load_source_ids()  # Rename this function or create a new one to load domains
    print(f"Using domains: {domains}")  # Print all domains for verification
    articles = fetch_news(domains)
    save_articles(articles)