import logging
from typing import List, Dict, Optional
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import string

logger = logging.getLogger(__name__)

def download_nltk_data():
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('punkt', quiet=True)
        return True
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")
        return False

def simple_tokenize(text):
    # Simple tokenization by splitting on whitespace and punctuation
    return re.findall(r'\b\w+\b', text.lower())

def remove_source_ending(text: str) -> str:
    # Pattern to match "proviene da" followed by any source name
    pattern = r'\s*proviene da.*$'
    return re.sub(pattern, '', text, flags=re.IGNORECASE)

def preprocess_text(text: str, use_nltk: bool = True) -> List[str]:
    if not isinstance(text, str):
        logger.warning(f"Expected string, got {type(text)}. Converting to string.")
        text = str(text)
    
    # Remove the source ending
    text = remove_source_ending(text)
    
    text = text.lower()
    
    if use_nltk:
        try:
            tokens = word_tokenize(text)
        except Exception as e:
            logger.error(f"NLTK tokenization failed: {e}")
            tokens = text.split()
    else:
        tokens = text.split()
    
    stop_words = set(stopwords.words('italian'))
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    
    return tokens

def preprocess_articles(articles: List[Dict]) -> List[List[str]]:
    preprocessed_articles = []
    
    for i, article in enumerate(articles):
        try:
            title = article.get('title', '')
            description = article.get('description', '')
            
            if not isinstance(title, str):
                logger.warning(f"Article {i}: title is not a string. Type: {type(title)}")
                title = str(title) if title is not None else ''
            
            if not isinstance(description, str):
                logger.warning(f"Article {i}: description is not a string. Type: {type(description)}")
                description = str(description) if description is not None else ''
            
            text = f"{title} {description}".strip()
            
            if not text:
                logger.warning(f"Article {i}: Empty text after combining title and description")
                continue
            
            tokens = preprocess_text(text, use_nltk=True)
            
            if tokens:
                preprocessed_articles.append(tokens)
            else:
                logger.warning(f"Article {i}: No tokens after preprocessing")
        
        except Exception as e:
            logger.error(f"Error processing article {i}: {e}")
    
    return preprocessed_articles

if __name__ == "__main__":
    download_nltk_data()