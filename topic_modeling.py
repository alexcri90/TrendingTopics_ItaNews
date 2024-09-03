from gensim import corpora
from gensim.models import LdaModel

def perform_topic_modeling(preprocessed_articles, num_topics=10):
    # Create dictionary
    texts = [article['processed_text'] for article in preprocessed_articles]
    dictionary = corpora.Dictionary(texts)
    
    # Create corpus
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    # Train LDA model
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=100)
    
    return lda_model, dictionary