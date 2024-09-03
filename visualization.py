import plotly.graph_objects as go

def create_topic_visualization(lda_model, dictionary):
    topic_keywords = []
    for idx, topic in lda_model.print_topics(-1):
        topic_keywords.append([w for w, _ in lda_model.show_topic(idx, topn=30)])
    
    fig = go.Figure()
    
    for i, keywords in enumerate(topic_keywords):
        fig.add_trace(go.Bar(
            x=keywords,
            y=[i+1]*len(keywords),
            name=f'Topic {i+1}',
            orientation='h'
        ))
    
    fig.update_layout(
        title='Top 30 Keywords per Topic',
        xaxis_title='Keywords',
        yaxis_title='Topics',
        barmode='stack'
    )
    
    return fig