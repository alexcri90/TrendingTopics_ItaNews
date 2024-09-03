import json
from gensim import corpora
from gensim.models import LdaModel

def perform_topic_modeling(date, num_topics=10):
    filename = f"data/articles_{date}.json"
    
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            articles = json.load(f)
    except FileNotFoundError:
        print(f"Error: File {filename} not found.")
        return None, None
    except json.JSONDecodeError:
        print(f"Error: File {filename} is not a valid JSON file.")
        return None, None

    # Extract processed text from articles
    processed_texts = [article.get('processed_text', []) for article in articles]

    # Create dictionary
    dictionary = corpora.Dictionary(processed_texts)
    
    # Create corpus
    corpus = [dictionary.doc2bow(text) for text in processed_texts]
    
    # Train LDA model
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=100)
    
    return lda_model, dictionary