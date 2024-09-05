import streamlit as st
import plotly.graph_objs as go
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

def visualize_topics(lda_model, corpus, dictionary):
    st.header("Topic Visualization")

    # Topic-Term Heatmap
    topic_term_heatmap(lda_model)

    # Interactive Topic Explorer
    interactive_topic_explorer(lda_model)

    # Topic Distribution Across Documents
    topic_distribution(lda_model, corpus)

    # Topic Similarity Network
    topic_similarity_network(lda_model)

def topic_term_heatmap(lda_model):
    st.subheader("Topic-Term Heatmap")
    
    # Get top 10 words for each topic
    topic_words = [[word for word, _ in lda_model.show_topic(topic_id, topn=10)] for topic_id in range(lda_model.num_topics)]
    
    # Create heatmap data
    heatmap_data = []
    for topic_id, words in enumerate(topic_words):
        for word in words:
            weight = dict(lda_model.show_topic(topic_id, topn=10))[word]
            heatmap_data.append([f"Topic {topic_id+1}", word, weight])

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

def interactive_topic_explorer(lda_model):
    st.subheader("Interactive Topic Explorer")

    num_topics = lda_model.num_topics
    topic_words = []
    for i in range(num_topics):
        topic_words.append([word for word, _ in lda_model.show_topic(i, topn=10)])

    # Create a DataFrame for easy manipulation
    df = pd.DataFrame(topic_words).T
    df.columns = [f'Topic {i+1}' for i in range(num_topics)]

    # Melt the DataFrame for use with Plotly
    df_melted = df.melt(var_name='Topic', value_name='Word')
    df_melted['Size'] = [dict(lda_model.show_topic(int(topic.split()[-1])-1)).get(word, 0) for topic, word in zip(df_melted['Topic'], df_melted['Word'])]

    # Create an interactive bubble chart
    fig = px.scatter(df_melted, x='Topic', y='Word', size='Size', color='Topic',
                     hover_name='Word', size_max=60, height=600)

    fig.update_layout(
        title='Top Words per Topic',
        xaxis_title='Topics',
        yaxis_title='Words',
        showlegend=False
    )

    st.plotly_chart(fig, use_container_width=True)

    # Add word clouds
    st.subheader("Topic Word Clouds")
    cols = st.columns(2)
    for topic_id in range(num_topics):
        word_freq = dict(lda_model.show_topic(topic_id, topn=30))
        wc = WordCloud(background_color="white", max_words=30, width=400, height=400)
        wc.generate_from_frequencies(word_freq)
        
        with cols[topic_id % 2]:
            st.write(f"Topic {topic_id+1}")
            fig, ax = plt.subplots(figsize=(5,5))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)

def topic_distribution(lda_model, corpus):
    st.subheader("Topic Distribution Across Documents")

    topic_weights = []
    for doc in corpus:
        topic_weights.append([weight for _, weight in lda_model[doc]])

    df = pd.DataFrame(topic_weights)
    df.columns = [f'Topic {i+1}' for i in range(lda_model.num_topics)]

    fig = px.area(df, height=500)
    fig.update_layout(
        title='Topic Distribution Across Documents',
        xaxis_title='Documents',
        yaxis_title='Topic Weight',
        legend_title='Topics',
        hovermode='x unified'
    )

    st.plotly_chart(fig, use_container_width=True)

def topic_similarity_network(lda_model):
    st.subheader("Topic Similarity Network")

    num_topics = lda_model.num_topics

    if num_topics < 3:
        st.write("Not enough topics to create a meaningful network visualization.")
        return

    # Calculate topic similarities
    topic_similarities = np.zeros((num_topics, num_topics))
    for i in range(num_topics):
        for j in range(i, num_topics):
            terms_i = dict(lda_model.get_topic_terms(i, 30))
            terms_j = dict(lda_model.get_topic_terms(j, 30))
            common_terms = set(terms_i.keys()) & set(terms_j.keys())
            similarity = sum(min(terms_i[t], terms_j[t]) for t in common_terms)
            topic_similarities[i, j] = similarity
            topic_similarities[j, i] = similarity

    # Normalize similarities
    topic_similarities /= np.max(topic_similarities)

    # Use PCA for dimensionality reduction
    pca = PCA(n_components=2, random_state=42)
    topic_coordinates = pca.fit_transform(topic_similarities)

    # Create edges
    edge_traces = []
    for i in range(num_topics):
        for j in range(i+1, num_topics):
            x0, y0 = topic_coordinates[i]
            x1, y1 = topic_coordinates[j]
            similarity = topic_similarities[i, j]
            if similarity > 0.1:  # Only draw edges for non-trivial similarities
                edge_trace = go.Scatter(
                    x=[x0, x1, None], y=[y0, y1, None],
                    line=dict(width=similarity * 3, color='rgba(150,150,150,0.8)'),
                    hoverinfo='none',
                    mode='lines')
                edge_traces.append(edge_trace)

    # Create nodes
    node_trace = go.Scatter(
        x=topic_coordinates[:, 0], y=topic_coordinates[:, 1],
        mode='markers+text',
        hoverinfo='text',
        text=[f'Topic {i+1}' for i in range(num_topics)],
        textposition="top center",
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            size=20,
            color=[sum(topic_similarities[i]) for i in range(num_topics)],
            colorbar=dict(
                thickness=15,
                title='Topic Connectivity',
                xanchor='left',
                titleside='right'
            )
        )
    )

    # Create the figure
    fig = go.Figure(data=edge_traces + [node_trace],
                    layout=go.Layout(
                        title='Topic Similarity Network',
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )

    st.plotly_chart(fig, use_container_width=True)