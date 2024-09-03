import pyLDAvis
import pyLDAvis.gensim_models
import streamlit as st
import streamlit.components.v1 as components

def create_topic_visualization(lda_model, corpus, dictionary):
    # Prepare the visualization
    vis_data = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
    
    # Save the visualization as an HTML file
    pyLDAvis.save_html(vis_data, 'lda_visualization.html')
    
    # Read the saved HTML file
    with open('lda_visualization.html', 'r', encoding='utf-8') as f:
        html_string = f.read()
    
    # Display the visualization in Streamlit
    components.html(html_string, width=1300, height=800)

def display_top_terms(lda_model, num_words=30):
    for idx, topic in lda_model.print_topics(-1, num_words):
        st.write(f"Topic {idx + 1}:")
        terms = topic.split('+')
        for term in terms:
            weight, word = term.split('*')
            st.write(f"  {word.strip()[1:-1]}: {float(weight):.4f}")
        st.write("\n")

def visualize_topics(lda_model, corpus, dictionary):
    st.write("Intertopic Distance Map:")
    create_topic_visualization(lda_model, corpus, dictionary)
    
    st.write("\nTop 30 Most Salient Terms for Each Topic:")
    display_top_terms(lda_model)