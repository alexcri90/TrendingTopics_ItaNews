import os
import json
import requests
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Constants
API_KEY = os.getenv('NEWSDATA_API_KEY')
SOURCES_URL = 'https://newsdata.io/api/1/sources'

def get_italian_sources():
    """
    Fetch and display available Italian news sources.
    """
    params = {
        'apikey': API_KEY,
        'language': 'it',
        'country': 'it'
    }
    
    try:
        response = requests.get(SOURCES_URL, params=params)
        response.raise_for_status()
        
        data = response.json()
        
        if data['status'] == 'success':
            sources = data['results']
            print(f"Found {len(sources)} Italian news sources:")
            for source in sources:
                print(f"- ID: {source['id']}")
                print(f"  Name: {source['name']}")
                print(f"  URL: {source['url']}")
                print()
            
            # Save sources to a JSON file
            save_sources(sources)
        else:
            print(f"Error in API response: {data.get('results', {}).get('message', 'Unknown error')}")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching sources: {e}")

def save_sources(sources):
    """
    Save the list of sources to a JSON file.
    """
    # Create a 'data' directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Generate filename with current date
    date_str = datetime.now().strftime('%Y-%m-%d')
    filename = f'data/italian_sources_{date_str}.json'
    
    # Save sources to JSON file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(sources, f, ensure_ascii=False, indent=4)
    
    print(f"Saved {len(sources)} sources to {filename}")

if __name__ == "__main__":
    get_italian_sources()