import streamlit as st
import json
from datetime import datetime, timedelta
from preprocessing import preprocess_articles
from visualization import visualize_topics_sklearn
import nltk
import os
import logging
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from topic_modeling import perform_topic_modeling

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

USE_GPU = torch.cuda.is_available()

def load_articles(date):
    """
    Load articles from a JSON file for a specific date.
    
    Args:
    date (str): Date string in the format 'YYYY-MM-DD'
    
    Returns:
    list: List of article dictionaries
    """
    try:
        with open(f"data/articles_{date}.json", 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"No data file found for date: {date}")
        return None

def download_nltk_data():
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        st.warning("Downloading necessary NLTK data...")
        nltk.download('stopwords')
        nltk.download('punkt')

    # Check if the download was successful
    try:
        nltk.data.find('corpora/stopwords')
        nltk.data.find('tokenizers/punkt')
        st.success("NLTK data downloaded successfully!")
    except LookupError:
        st.error("Failed to download NLTK data. Please try manual download.")
        st.code("""
import nltk
nltk.download('stopwords')
nltk.download('punkt')
        """)

class ArticleDataset(Dataset):
    def __init__(self, articles, tokenizer, max_length=512):
        self.articles = articles
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.articles)

    def __getitem__(self, idx):
        return self.tokenizer(self.articles[idx], truncation=True, padding='max_length', max_length=self.max_length, return_tensors='pt')

def encode_articles(articles, model, tokenizer, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    dataset = ArticleDataset(articles, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    embeddings = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].squeeze(1).to(device)
            attention_mask = batch['attention_mask'].squeeze(1).to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            embeddings.append(outputs.last_hidden_state[:, 0, :].cpu())

    return torch.cat(embeddings, dim=0)

def main():
    st.title('Italian News Topic Modeler')
    
    # Date selection
    today = datetime.now().date()
    selected_date = st.date_input(
        "Select date for analysis",
        value=today,
        max_value=today,
        min_value=today - timedelta(days=30)
    )
    
    date_str = selected_date.strftime('%Y-%m-%d')
    st.write(f"Analyzing data for: {date_str}")
    
    # Load data
    articles = load_articles(date_str)
    
    if articles:
        st.write(f"Loaded {len(articles)} articles")
        logger.info(f"Loaded {len(articles)} articles")
        
        try:
            # Preprocess articles
            preprocessed_articles = preprocess_articles(articles)
            logger.info(f"Preprocessed {len(preprocessed_articles)} articles")
            
            if not preprocessed_articles:
                st.error("No articles remained after preprocessing. Check your data.")
                logger.error("No articles remained after preprocessing")
                return
            
            # Perform topic modeling
            with st.spinner("Performing topic modeling..."):
                lda_model, feature_names, X = perform_topic_modeling(preprocessed_articles)
            
            if lda_model and feature_names is not None and X is not None:
                st.write(f"Performed topic modeling. Number of topics: {lda_model.n_components}")
                logger.info(f"Performed topic modeling. Number of topics: {lda_model.n_components}")
                
                # Create and display visualization
                visualize_topics_sklearn(lda_model, X, feature_names, articles)
            else:
                st.error("Topic modeling failed. Please check your data structure.")
                logger.error("Topic modeling failed")
            
        except Exception as e:
            logger.exception("An error occurred during processing")
            st.error(f"An error occurred during processing: {str(e)}")
            st.error("Check the logs for more details.")
    else:
        st.write(f"No data available for analysis on {date_str}.")
        logger.warning(f"No data available for analysis on {date_str}")
        st.write("Please make sure you have a JSON file named "
                 f"'articles_{date_str}.json' in the 'data/' directory.")

    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

if __name__ == "__main__":
    main()