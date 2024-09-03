import plotly.graph_objects as go
import plotly.colors as colors

def create_topic_visualization(lda_model, dictionary, num_keywords=30):
    if not lda_model or not dictionary:
        raise ValueError("LDA model and dictionary must be provided")

    topic_keywords = []
    for idx in range(lda_model.num_topics):
        topic_keywords.append([w for w, _ in lda_model.show_topic(idx, topn=num_keywords)])
    
    fig = go.Figure()
    
    color_scale = colors.sequential.Viridis

    for i, keywords in enumerate(topic_keywords):
        fig.add_trace(go.Bar(
            x=keywords,
            y=[i+1]*len(keywords),
            name=f'Topic {i+1}',
            orientation='h',
            marker_color=color_scale[i % len(color_scale)]
        ))
    
    fig.update_layout(
        title=f'Top {num_keywords} Keywords per Topic',
        xaxis_title='Keywords',
        yaxis_title='Topics',
        barmode='stack',
        height=100 * lda_model.num_topics,  # Adjust height based on number of topics
        margin=dict(l=100, r=20, t=70, b=70)
    )
    
    return fig