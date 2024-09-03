import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

# Explicitly download Italian punkt data
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt', quiet=True)

# Load Italian tokenizer
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
punkt_param = PunktParameters()
punkt_param.abbrev_types = set(['dr', 'vs', 'mr', 'mrs', 'prof', 'inc', 'fig'])
italian_tokenizer = PunktSentenceTokenizer(punkt_param)

def preprocess_text(text):
    # Tokenize
    sentences = italian_tokenizer.tokenize(text.lower())
    tokens = [word for sent in sentences for word in word_tokenize(sent)]
    
    # Remove stopwords and non-alphabetic tokens
    stop_words = set(stopwords.words('italian'))
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    
    # Stem
    stemmer = SnowballStemmer('italian')
    tokens = [stemmer.stem(token) for token in tokens]
    
    return tokens

def preprocess_articles(articles):
    processed_articles = []
    for article in articles:
        content = article.get('content', '')
        if content:
            processed_text = preprocess_text(content)
            processed_article = article.copy()
            processed_article['processed_text'] = processed_text
            processed_articles.append(processed_article)
    
    print(f"Preprocessed {len(processed_articles)} articles")
    return processed_articles