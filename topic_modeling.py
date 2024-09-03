import pandas as pd
from gensim import corpora
from gensim.models import LdaModel

def perform_topic_modeling(date, num_topics=10):
    filename = f"data/processed_articles_{date}.csv"
    df = pd.read_csv(filename)
    
    # Create dictionary
    dictionary = corpora.Dictionary(df['processed_text'].apply(eval))
    
    # Create corpus
    corpus = [dictionary.doc2bow(text) for text in df['processed_text'].apply(eval)]
    
    # Train LDA model
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=100)
    
    return lda_model, dictionary