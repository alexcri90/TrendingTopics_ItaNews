import numpy as np
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel
from topic_modeling import perform_topic_modeling
import logging

logger = logging.getLogger(__name__)

def compute_coherence_values(texts, start=3, limit=12, step=1):
    coherence_values = []
    model_list = []
    
    # Create dictionary from the original texts
    dictionary = corpora.Dictionary(texts)
    
    # Create corpus
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    for num_topics in range(start, limit + 1, step):
        try:
            model, feature_names, X = perform_topic_modeling(texts, num_topics=num_topics)
            model_list.append(model)
            
            # Ensure the model has the correct format for coherence calculation
            topics = model.components_
            topic_words = [[feature_names[i] for i in topic.argsort()[:-10 - 1:-1]] for topic in topics]
            
            coherencemodel = CoherenceModel(topics=topic_words, texts=texts, dictionary=dictionary, coherence='c_v')
            coherence_value = coherencemodel.get_coherence()
            coherence_values.append(coherence_value)
            
            logger.info(f"Num Topics: {num_topics}, Coherence Score: {coherence_value}")
        except Exception as e:
            logger.error(f"Error computing coherence for {num_topics} topics: {str(e)}")
            coherence_values.append(None)

    return model_list, coherence_values

def find_optimal_number_of_topics(texts):
    model_list, coherence_values = compute_coherence_values(texts)
    
    # Filter out None values
    valid_coherence_values = [v for v in coherence_values if v is not None]
    
    if not valid_coherence_values:
        logger.warning("No valid coherence values found. Defaulting to 5 topics.")
        return 5
    
    optimal_num_topics = coherence_values.index(max(valid_coherence_values)) + 3  # +3 because we start from 3
    return optimal_num_topics
