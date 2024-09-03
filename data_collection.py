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

def fetch_news(language='it', category=None, country=None, max_credits=12):
    """
    Fetch news articles from NewsData.io API, using exactly 12 credits.
    """
    params = {
        'apikey': API_KEY,
        'language': language,
    }
    
    if category:
        params['category'] = category
    if country:
        params['country'] = country

    all_articles = []
    credits_used = 0
    next_page = None

    while credits_used < max_credits:
        if next_page:
            params['page'] = next_page

        try:
            print(f"Fetching news articles (Credit {credits_used + 1}/{max_credits})...")
            print(f"Request URL: {BASE_URL}")
            print(f"Request params: {params}")
            response = requests.get(BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()

            if data['status'] == 'success':
                all_articles.extend(data['results'])
                print(f"Fetched {len(data['results'])} articles. Total: {len(all_articles)}")
                
                credits_used += 1
                next_page = data.get('nextPage')
                if not next_page:
                    break
            else:
                print(f"Error in API response: {data.get('results', {}).get('message', 'Unknown error')}")
                break

        except requests.exceptions.RequestException as e:
            print(f"Error fetching articles: {e}")
            print(f"Response content: {response.text}")
            break

    print(f"Total credits used: {credits_used}")
    return all_articles

def process_articles(articles):
    """
    Process and flatten the article data.
    """
    processed_articles = []
    for article in articles:
        processed_article = {
            'article_id': article.get('article_id'),
            'title': article.get('title'),
            'description': article.get('description'),
            'content': article.get('content'),
            'link': article.get('link'),
            'pubDate': article.get('pubDate'),
            'source_id': article.get('source_id'),
            'source_url': article.get('source_url'),
            'creator': article.get('creator'),
            'category': article.get('category'),
            'language': article.get('language'),
            'country': article.get('country')
        }
        processed_articles.append(processed_article)
    return processed_articles

def save_articles(articles):
    """
    Save fetched articles to a JSON file.
    """
    date_str = datetime.now().strftime('%Y-%m-%d')
    os.makedirs('data', exist_ok=True)
    
    filename = f"data/articles_{date_str}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(articles, f, ensure_ascii=False, indent=4)
    
    print(f"Saved {len(articles)} articles to {filename}")

def main(language='it', category=None, country=None):
    print("Starting data collection process")
    print(f"API Key being used: {API_KEY[:5]}...{API_KEY[-5:]}")  # Print first and last 5 characters of API key
    
    articles = fetch_news(language, category, country, max_credits=12)
    if articles:
        print(f"Fetched a total of {len(articles)} articles")
        
        processed_articles = process_articles(articles)
        save_articles(processed_articles)
    else:
        print("No articles fetched.")

    print("Data collection process completed")

if __name__ == "__main__":
    main(language='it', country='it')