import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
API_KEY = os.getenv('NEWS_API_KEY')
SOURCES_URL = 'https://newsapi.org/v2/top-headlines/sources'

def get_italian_sources():
    """
    Fetch and display available Italian news sources.
    """
    params = {
        'apiKey': API_KEY,
        'language': 'it',
        'country': 'it'
    }
    
    try:
        response = requests.get(SOURCES_URL, params=params)
        response.raise_for_status()
        
        sources = response.json()['sources']
        
        print(f"Found {len(sources)} Italian news sources:")
        for source in sources:
            print(f"- ID: {source['id']}")
            print(f"  Name: {source['name']}")
            print(f"  Description: {source['description'][:100]}...")  # First 100 characters of description
            print()

    except requests.exceptions.RequestException as e:
        print(f"Error fetching sources: {e}")

if __name__ == "__main__":
    get_italian_sources()