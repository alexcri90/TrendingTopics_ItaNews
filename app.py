import streamlit as st
from datetime import datetime, timedelta
from data_collection import main as collect_data
from preprocessing import preprocess_articles
from topic_modeling import perform_topic_modeling
from visualization import create_topic_visualization

def main():
    st.title('Italian News Topic Modeler')
    
    # Get yesterday's date
    yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    # Collect data
    collect_data()
    
    # Preprocess articles
    preprocess_articles(yesterday)
    
    # Perform topic modeling
    lda_model, dictionary = perform_topic_modeling(yesterday)
    
    # Create visualization
    fig = create_topic_visualization(lda_model, dictionary)
    
    # Display visualization
    st.plotly_chart(fig)

if __name__ == "__main__":
    main()