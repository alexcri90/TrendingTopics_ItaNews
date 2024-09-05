import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud

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

def create_topic_document_map(lda_model, X, articles):
    # Perform t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42)
    doc_topic_dist = lda_model.transform(X)
    tsne_output = tsne.fit_transform(doc_topic_dist)
    
    # Create a DataFrame for plotting
    df = pd.DataFrame({
        'x': tsne_output[:, 0],
        'y': tsne_output[:, 1],
        'topic': doc_topic_dist.argmax(axis=1),
        'title': [a['title'] for a in articles]
    })
    
    # Create the scatter plot
    fig = px.scatter(df, x='x', y='y', color='topic', hover_data=['title'],
                     title='Topic-Document Map')
    st.plotly_chart(fig)

def create_topic_trends(lda_model, articles):
    # Check if 'date' is available in the articles
    if 'date' not in articles[0]:
        st.warning("Date information is not available. Unable to create topic trends visualization.")
        return

    dates = [a['date'] for a in articles]
    doc_topic_dist = lda_model.transform([a['content'] for a in articles])
    
    df = pd.DataFrame({
        'date': dates,
        **{f'Topic {i}': doc_topic_dist[:, i] for i in range(lda_model.n_components)}
    })
    df = df.groupby('date').mean().reset_index()
    
    fig = px.line(df, x='date', y=[f'Topic {i}' for i in range(lda_model.n_components)],
                  title='Topic Trends Over Time')
    st.plotly_chart(fig)

def create_topic_similarity_network(lda_model, feature_names):
    # Calculate topic similarity
    topic_term_dists = lda_model.components_ / lda_model.components_.sum(axis=1)[:, np.newaxis]
    topic_similarity = np.dot(topic_term_dists, topic_term_dists.T)
    
    # Create network graph
    G = nx.Graph()
    for i in range(lda_model.n_components):
        G.add_node(i)
        for j in range(i+1, lda_model.n_components):
            if topic_similarity[i, j] > 0.2:  # Adjust threshold as needed
                G.add_edge(i, j, weight=topic_similarity[i, j])
    
    # Get node positions
    pos = nx.spring_layout(G)
    
    # Create edges trace
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.5, color='#888'), hoverinfo='none', mode='lines')
    
    # Create nodes trace
    node_x, node_y = [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers', hoverinfo='text',
                            marker=dict(showscale=True, colorscale='YlGnBu', size=10))
    
    # Add node info
    node_adjacencies = []
    node_text = []
    for node, adjacencies in enumerate(G.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append(f'Topic {node}')
    node_trace.marker.color = node_adjacencies
    node_trace.text = node_text
    
    # Create the figure
    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(title='Topic Similarity Network', showlegend=False,
                                     hovermode='closest', margin=dict(b=20,l=5,r=5,t=40)))
    st.plotly_chart(fig)

def display_top_articles(lda_model, articles):
    doc_topic_dist = lda_model.transform([a['content'] for a in articles])
    for topic in range(lda_model.n_components):
        st.subheader(f"Top Articles for Topic {topic}")
        top_doc_indices = doc_topic_dist[:, topic].argsort()[-5:][::-1]
        for idx in top_doc_indices:
            st.write(f"- {articles[idx]['title']} (Topic proportion: {doc_topic_dist[idx, topic]:.2f})")

def create_topic_proportion_chart(lda_model, articles):
    doc_topic_dist = lda_model.transform([a['content'] for a in articles])
    df = pd.DataFrame(doc_topic_dist, columns=[f'Topic {i}' for i in range(lda_model.n_components)])
    df['Article'] = [a['title'] for a in articles]
    df = df.set_index('Article')
    
    fig = px.bar(df, title='Topic Proportions per Article', barmode='stack')
    st.plotly_chart(fig)