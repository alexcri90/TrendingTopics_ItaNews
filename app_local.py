import streamlit as st
import json
from datetime import datetime, timedelta
from preprocessing import preprocess_articles
from topic_modeling import perform_topic_modeling
from visualization import visualize_topics
import nltk
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    except LookupError:
        st.warning("Downloading necessary NLTK data...")
        nltk.download('stopwords')

    # Check if the download was successful
    try:
        nltk.data.find('corpora/stopwords')
        st.success("NLTK data downloaded successfully!")
    except LookupError:
        st.error("Failed to download NLTK data. Please try manual download.")
        st.code("""
import nltk
nltk.download('stopwords')
        """)

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
            with st.spinner("Performing topic modeling and finding optimal number of topics..."):
                lda_model, dictionary, corpus = perform_topic_modeling(preprocessed_articles)
            
            if lda_model and dictionary and corpus:
                st.write(f"Performed topic modeling on titles and descriptions. Number of topics: {lda_model.num_topics}")
                logger.info(f"Performed topic modeling. Number of topics: {lda_model.num_topics}")
                
                # Create and display visualization
                visualize_topics(lda_model, corpus, dictionary)
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

if __name__ == "__main__":
    main()