import streamlit as st
import json
from datetime import datetime, timedelta
from preprocessing import preprocess_articles
from topic_modeling import perform_topic_modeling
from visualization import create_topic_visualization
import nltk
import ssl

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
        # Create an unverified SSL context to avoid download issues
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context

        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        
        # Explicitly download Italian punkt data
        import nltk.data
        nltk.data.find('tokenizers/punkt/italian.pickle')
    except LookupError:
        st.warning("Downloading necessary NLTK data...")
        nltk.download('punkt')
        nltk.download('stopwords')
    except Exception as e:
        st.error(f"An error occurred while downloading NLTK data: {str(e)}")
        st.error("Please try running the following commands in your Python environment:")
        st.code("import nltk\nnltk.download('punkt')\nnltk.download('stopwords')")

def main():
    st.title('Italian News Topic Modeler')
    
    # Ensure NLTK data is available
    download_nltk_data()
    
    # Get today's date
    today = datetime.now().date()
    
    # Date selection
    selected_date = st.date_input(
        "Select date for analysis",
        value=today,
        max_value=today,
        min_value=today - timedelta(days=30)  # Allowing selection up to 30 days in the past
    )
    
    date_str = selected_date.strftime('%Y-%m-%d')
    st.write(f"Analyzing data for: {date_str}")
    
    # Load data
    articles = load_articles(date_str)
    
    if articles:
        st.write(f"Loaded {len(articles)} articles")
        
        try:
            # Preprocess articles
            preprocessed_articles = preprocess_articles(articles)
            st.write(f"Preprocessed {len(preprocessed_articles)} articles")
            
            # Perform topic modeling
            lda_model, dictionary = perform_topic_modeling(preprocessed_articles)
            st.write(f"Performed topic modeling. Number of topics: {lda_model.num_topics}")
            
            # Create visualization
            fig = create_topic_visualization(lda_model, dictionary)
            
            # Display visualization
            st.plotly_chart(fig)
        except Exception as e:
            st.error(f"An error occurred during processing: {str(e)}")
            st.error("If the error persists, please try manually downloading NLTK data:")
            st.code("""
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('popular')
            """)
    else:
        st.write(f"No data available for analysis on {date_str}.")
        st.write("Please make sure you have a JSON file named "
                 f"'articles_{date_str}.json' in the 'data/' directory.")

if __name__ == "__main__":
    main()