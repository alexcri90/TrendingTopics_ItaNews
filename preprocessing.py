import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

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

def preprocess_text(text, use_nltk=True):
    if use_nltk:
        try:
            # Tokenize using NLTK's word_tokenize
            tokens = word_tokenize(text.lower())
        except LookupError:
            print("NLTK tokenizer not available. Using simple tokenization.")
            tokens = simple_tokenize(text)
    else:
        tokens = simple_tokenize(text)
    
    # Remove stopwords and non-alphabetic tokens
    stop_words = set(stopwords.words('italian'))
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    
    return tokens

def preprocess_articles(articles):
    use_nltk = download_nltk_data()
    
    processed_articles = []
    for article in articles:
        title = article.get('title', '')
        description = article.get('description', '')
        
        # Combine title and description
        combined_text = f"{title} {description}".strip()
        
        if combined_text:
            processed_text = preprocess_text(combined_text, use_nltk)
            processed_article = article.copy()
            processed_article['processed_text'] = processed_text
            processed_articles.append(processed_article)
    
    print(f"Preprocessed {len(processed_articles)} articles")
    return processed_articles