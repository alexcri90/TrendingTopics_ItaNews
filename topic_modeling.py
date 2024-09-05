from gensim import corpora
from gensim.models import LdaModel
from gensim.models import CoherenceModel
import matplotlib.pyplot as plt
import streamlit as st

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=1):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=42)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

def perform_topic_modeling(preprocessed_articles):
    dictionary = corpora.Dictionary(preprocessed_articles)
    corpus = [dictionary.doc2bow(text) for text in preprocessed_articles]

    # Find optimal number of topics
    limit = 15
    start = 2
    step = 1
    model_list, coherence_values = compute_coherence_values(dictionary, corpus, preprocessed_articles, limit, start, step)

    # Select the model with the highest coherence score
    optimal_model = model_list[coherence_values.index(max(coherence_values))]

    return optimal_model, dictionary, corpus