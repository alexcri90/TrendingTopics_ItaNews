import streamlit as st
import plotly.graph_objs as go
import numpy as np
from sklearn.manifold import TSNE
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pandas as pd

def visualize_topics_sklearn(lda_model, X, feature_names, articles):
    st.header("Topic Visualization")

    # Topic-Term Heatmap
    topic_term_heatmap(lda_model, feature_names)

    # Intertopic Distance Map
    intertopic_distance_map(lda_model)

    # Word Clouds
    topic_word_clouds(lda_model, feature_names)

    # Topic Trends Over Time
    topic_trends_over_time(lda_model, X, articles)

def topic_term_heatmap(lda_model, feature_names):
    st.subheader("Topic-Term Heatmap")
    
    # Get top 10 words for each topic
    n_top_words = 10
    topic_words = []
    for topic_idx, topic in enumerate(lda_model.components_):
        top_words_idx = topic.argsort()[:-n_top_words - 1:-1]
        top_words = [feature_names[i] for i in top_words_idx]
        topic_words.append(top_words)
    
    # Create heatmap data
    heatmap_data = []
    for topic_idx, words in enumerate(topic_words):
        for word in words:
            weight = lda_model.components_[topic_idx, list(feature_names).index(word)]
            heatmap_data.append([f"Topic {topic_idx+1}", word, weight])

    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=[d[2] for d in heatmap_data],
        x=[d[1] for d in heatmap_data],
        y=[d[0] for d in heatmap_data],
        colorscale='Viridis'
    ))

    fig.update_layout(
        title='Topic-Term Heatmap',
        xaxis_title='Words',
        yaxis_title='Topics',
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

def intertopic_distance_map(lda_model):
    st.subheader("Intertopic Distance Map")
    
    # Apply t-SNE to topic-term matrix
    n_topics = lda_model.n_components
    perplexity = min(30, n_topics - 1)  # Adjust perplexity based on number of topics
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    topic_coord = tsne.fit_transform(lda_model.components_)
    
    # Create scatter plot
    fig = go.Figure(data=go.Scatter(
        x=topic_coord[:, 0],
        y=topic_coord[:, 1],
        mode='markers+text',
        marker=dict(size=10, color=list(range(n_topics)), colorscale='Viridis', showscale=True),
        text=[f"Topic {i+1}" for i in range(n_topics)],
        textposition="top center"
    ))

    fig.update_layout(
        title='Intertopic Distance Map',
        xaxis_title='t-SNE dimension 1',
        yaxis_title='t-SNE dimension 2',
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)

def topic_word_clouds(lda_model, feature_names):
    st.subheader("Topic Word Clouds")
    
    # Create word cloud for each topic
    for topic_idx, topic in enumerate(lda_model.components_):
        word_freq = dict(zip(feature_names, topic))
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(f'Topic {topic_idx + 1}')
        
        st.pyplot(fig)

def topic_trends_over_time(lda_model, X, articles):
    st.subheader("Topic Trends Over Time")
    
    # Get topic distribution for each document
    doc_topic_dist = lda_model.transform(X)
    
    # Create DataFrame with topic distributions and dates
    df = pd.DataFrame(doc_topic_dist, columns=[f"Topic {i+1}" for i in range(lda_model.n_components)])
    df['date'] = [article.get('pubDate', article.get('publishedAt')) for article in articles]
    df['date'] = pd.to_datetime(df['date'])
    
    # Group by date and calculate mean topic distribution
    daily_topic_dist = df.groupby('date').mean()
    
    # Create line plot
    fig = go.Figure()
    for topic in daily_topic_dist.columns:
        fig.add_trace(go.Scatter(x=daily_topic_dist.index, y=daily_topic_dist[topic], mode='lines', name=topic))

    fig.update_layout(
        title='Topic Trends Over Time',
        xaxis_title='Date',
        yaxis_title='Topic Prevalence',
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)