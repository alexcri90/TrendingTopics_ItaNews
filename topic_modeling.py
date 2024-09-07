import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
from gensim.models.coherencemodel import CoherenceModel

def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=1):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = perform_topic_modeling(texts, num_topics=num_topics)[0]
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values

def perform_topic_modeling(preprocessed_articles, num_topics=10):
    # Use CountVectorizer for document-term matrix creation
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
    X = vectorizer.fit_transform([' '.join(doc) for doc in preprocessed_articles])

    # Use scikit-learn's LatentDirichletAllocation
    lda_model = LatentDirichletAllocation(n_components=num_topics, random_state=42, n_jobs=-1)
    lda_model.fit(X)

    # Get feature names (words)
    feature_names = vectorizer.get_feature_names_out()

    return lda_model, feature_names, X

class LDA(torch.nn.Module):
    def __init__(self, n_components, vocab_size, random_state, device):
        super(LDA, self).__init__()
        self.n_components = n_components
        self.vocab_size = vocab_size
        self.random_state = random_state
        self.device = device
        self.beta = torch.nn.Parameter(torch.randn(n_components, vocab_size, device=device))
        self.theta = None
        self.components_ = None

    def forward(self, X):
        batch_size = X.shape[0]
        self.theta = torch.softmax(torch.randn(batch_size, self.n_components, device=self.device), dim=1)
        for _ in range(10):  # You can adjust the number of iterations
            phi = torch.softmax(self.beta, dim=1)
            self.theta = torch.softmax(X @ phi.T, dim=1)
        self.components_ = phi.T
        return X @ phi.T

    def fit(self, X):
        self(X)
        return self