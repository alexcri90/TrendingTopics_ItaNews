import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    # Tokenize
    tokens = word_tokenize(text.lower())
    
    # Remove stopwords and non-alphabetic tokens
    stop_words = set(stopwords.words('italian'))
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    
    # Stem
    stemmer = SnowballStemmer('italian')
    tokens = [stemmer.stem(token) for token in tokens]
    
    return tokens

def preprocess_articles(date):
    filename = f"data/articles_{date}.csv"
    df = pd.read_csv(filename)
    
    df['processed_text'] = df['content'].fillna('').apply(preprocess_text)
    
    processed_filename = f"data/processed_articles_{date}.csv"
    df.to_csv(processed_filename, index=False)
    print(f"Saved processed articles to {processed_filename}")