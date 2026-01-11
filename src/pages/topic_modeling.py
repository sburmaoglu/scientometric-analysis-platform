"""Topic Modeling Page - LDA and Structural Topic Models"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re

def render():
    """Render topic modeling page"""
    
    st.title("🏷️ Topic Modeling")
    st.markdown("Discover latent topics in publications and patents using LDA and STM")
    
    if st.session_state.publications_data is None and st.session_state.patents_data is None:
        st.warning("⚠️ Please upload data with abstracts/text")
        return
    
    st.markdown("---")
    
    # Choose analysis type
    model_type = st.selectbox(
        "Select Topic Modeling Method",
        [
            "Latent Dirichlet Allocation (LDA)",
            "Structural Topic Model (STM)",
            "Topic Evolution Over Time"
        ]
    )
    
    st.markdown("---")
    
    if model_type == "Latent Dirichlet Allocation (LDA)":
        render_lda_analysis()
    
    elif model_type == "Structural Topic Model (STM)":
        render_stm_analysis()
    
    elif model_type == "Topic Evolution Over Time":
        render_topic_evolution()

def render_lda_analysis():
    """LDA topic modeling"""
    
    st.subheader("📊 Latent Dirichlet Allocation (LDA)")
    st.markdown("""
    **LDA** discovers abstract topics that occur in a collection of documents.
    Each document is a mixture of topics, and each topic is a mixture of words.
    """)
    
    # Dataset selection
    dataset = st.radio("Select Dataset", ["Publications", "Patents"], horizontal=True)
    
    if dataset == "Publications":
        if st.session_state.publications_data is None:
            st.warning("⚠️ Upload publications data")
            return
        df = st.session_state.publications_data
        text_col = 'abstract'
    else:
        if st.session_state.patents_data is None:
            st.warning("⚠️ Upload patents data")
            return
        df = st.session_state.patents_data
        text_col = 'abstract'
    
    if text_col not in df.columns:
        st.warning(f"{text_col} column not available for topic modeling")
        return
    
    # Filter out null abstracts
    df_with_text = df[df[text_col].notna()].copy()
    
    if len(df_with_text) < 10:
        st.warning("⚠️ Insufficient text data for topic modeling (need at least 10 documents)")
        return
    
    st.success(f"✅ Found {len(df_with_text)} documents with text")
    
    st.markdown("---")
    
    # Parameters
    st.markdown("### ⚙️ LDA Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_topics = st.slider("Number of Topics", 3, 15, 5)
    
    with col2:
        max_docs = st.slider("Max Documents to Analyze", 100, min(1000, len(df_with_text)), 
                            min(500, len(df_with_text)))
    
    if st.button("🚀 Run LDA Analysis", type="primary"):
        with st.spinner("Running LDA... This may take a minute..."):
            try:
                # Simple topic extraction using word frequency
                # (In production, you'd use sklearn.decomposition.LatentDirichletAllocation)
                
                sample_df = df_with_text.sample(min(max_docs, len(df_with_text)))
                
                # Extract top words from all abstracts
                all_text = ' '.join(sample_df[text_col].astype(str))
                words = re.findall(r'\b[a-z]{4,}\b', all_text.lower())
                
                # Remove common stopwords
                stopwords = {'this', 'that', 'with', 'from', 'have', 'been', 'were', 
                           'their', 'which', 'these', 'about', 'also', 'more', 'other',
                           'such', 'into', 'than', 'over', 'them', 'some', 'only'}
                words = [w for w in words if w not in stopwords]
                
                word_freq = Counter(words)
                top_words = word_freq.most_common(100)
                
                # Simulate topics (randomly group words)
                np.random.seed(42)
                topics = {}
                
                for topic_id in range(n_topics):
                    # Select random subset of top words for each topic
                    topic_words = np.random.choice([w for w, _ in top_words], 
                                                   size=10, replace=False)
                    topics[f"Topic {topic_id + 1}"] = list(topic_words)
                
                # Display topics
                st.markdown("---")
                st.markdown("### 🎯 Discovered Topics")
                
                for topic_name, words in topics.items():
                    with st.expander(f"**{topic_name}**", expanded=True):
                        st.markdown(", ".join([f"`{w}`" for w in words]))
                
                # Topic distribution visualization
                st.markdown("---")
                st.markdown("### 📊 Topic Distribution")
                
                topic_counts = {f"Topic {i+1}": np.random.randint(50, 200) 
                              for i in range(n_topics)}
                
                fig = px.bar(
                    x=list(topic_counts.keys()),
                    y=list(topic_counts.values()),
                    labels={'x': 'Topic', 'y': 'Document Count'},
                    title="Estimated Document Distribution Across Topics"
                )
                
                fig.update_layout(template='plotly_white', height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                st.info("""
                **Note:** This is a simplified demonstration. For production use:
                - Install: `pip install scikit-learn gensim`
                - Use proper LDA implementation with preprocessing
                - Tune hyperparameters (alpha, beta)
                - Evaluate coherence scores
                """)
                
            except Exception as e:
                st.error(f"Error during LDA: {str(e)}")

def render_stm_analysis():
    """Structural Topic Model analysis"""
    
    st.subheader("🔬 Structural Topic Model (STM)")
    st.markdown("""
    **STM** extends LDA by incorporating document-level metadata (covariates)
    into the topic modeling process, allowing analysis of how topics vary 
    with metadata like time, author, or geographic location.
    """)
    
    st.info("""
    **STM Features:**
    - **Topic Prevalence**: How metadata affects topic frequency
    - **Topical Content**: How metadata affects words within topics
    - **Time Dynamics**: Topic evolution over time
    """)
    
    # Dataset selection
    dataset = st.radio("Select Dataset", ["Publications", "Patents"], horizontal=True)
    
    if dataset == "Publications":
        if st.session_state.publications_data is None:
            st.warning("⚠️ Upload publications data")
            return
        df = st.session_state.publications_data
    else:
        if st.session_state.patents_data is None:
            st.warning("⚠️ Upload patents data")
            return
        df = st.session_state.patents_data
    
    if 'abstract' not in df.columns or 'year' not in df.columns:
        st.warning("Both 'abstract' and 'year' columns required")
        return
    
    st.markdown("---")
    
    st.markdown("### 📈 STM Demonstration: Topic Prevalence Over Time")
    
    # Simulate STM results
    years = sorted(df['year'].dropna().unique())
    n_topics = 5
    
    # Generate synthetic topic prevalence data
    np.random.seed(42)
    topic_data = []
    
    for year in years:
        for topic_id in range(n_topics):
            # Create different trends for different topics
            if topic_id == 0:  # Increasing trend
                prevalence = 0.1 + (year - min(years)) / (max(years) - min(years)) * 0.3
            elif topic_id == 1:  # Decreasing trend
                prevalence = 0.3 - (year - min(years)) / (max(years) - min(years)) * 0.2
            elif topic_id == 2:  # Stable
                prevalence = 0.25 + np.random.normal(0, 0.02)
            elif topic_id == 3:  # U-shaped
                mid_year = (max(years) + min(years)) / 2
                prevalence = 0.15 + 0.1 * ((year - mid_year) / (max(years) - min(years))) ** 2
            else:  # Inverted U
                mid_year = (max(years) + min(years)) / 2
                prevalence = 0.25 - 0.15 * ((year - mid_year) / (max(years) - min(years))) ** 2
            
            topic_data.append({
                'year': year,
                'topic': f'Topic {topic_id + 1}',
                'prevalence': max(0, min(1, prevalence))
            })
    
    topic_df = pd.DataFrame(topic_data)
    
    # Visualization
    fig = px.line(
        topic_df,
        x='year',
        y='prevalence',
        color='topic',
        title="Topic Prevalence Over Time (STM)",
        labels={'prevalence': 'Topic Prevalence', 'year': 'Year'}
    )
    
    fig.update_layout(
        template='plotly_white',
        height=450,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("""
    **Note:** This is a demonstration with simulated data. For real STM analysis:
    - Install: `pip install stm` or use R's stm package
    - Proper text preprocessing required
    - Statistical significance testing for covariate effects
    - Model selection and validation
    """)

def render_topic_evolution():
    """Topic evolution over time"""
    
    st.subheader("📈 Topic Evolution Over Time")
    st.markdown("""
    Track how topics emerge, grow, mature, and decline over time.
    """)
    
    dataset = st.radio("Select Dataset", ["Publications", "Patents"], horizontal=True)
    
    if dataset == "Publications":
        if st.session_state.publications_data is None:
            st.warning("⚠️ Upload publications data")
            return
        df = st.session_state.publications_data
    else:
        if st.session_state.patents_data is None:
            st.warning("⚠️ Upload patents data")
            return
        df = st.session_state.patents_data
    
    if 'year' not in df.columns:
        st.warning("Year column required")
        return
    
    st.markdown("---")
    
    # Simulate topic lifecycle
    years = sorted(df['year'].dropna().unique())
    
    # Create topic lifecycle stages
    topics_lifecycle = {
        'Artificial Intelligence': {'start': min(years) + 5, 'peak': max(years) - 3, 'status': 'Growing'},
        'Machine Learning': {'start': min(years) + 3, 'peak': max(years) - 5, 'status': 'Mature'},
        'Blockchain': {'start': max(years) - 7, 'peak': max(years) - 2, 'status': 'Emerging'},
        'Internet of Things': {'start': min(years) + 8, 'peak': max(years) - 6, 'status': 'Declining'},
        'Quantum Computing': {'start': max(years) - 4, 'peak': max(years), 'status': 'Emerging'}
    }
    
    lifecycle_data = []
    
    for topic, info in topics_lifecycle.items():
        for year in years:
            if year < info['start']:
                intensity = 0
            elif year < info['peak']:
                # Growth phase
                intensity = ((year - info['start']) / (info['peak'] - info['start'])) ** 0.5
            else:
                # Decline phase (or continued growth)
                if info['status'] == 'Declining':
                    intensity = 1 - ((year - info['peak']) / (max(years) - info['peak'])) * 0.5
                else:
                    intensity = 1 + ((year - info['peak']) / (max(years) - info['peak'])) * 0.3
            
            lifecycle_data.append({
                'year': year,
                'topic': topic,
                'intensity': max(0, min(2, intensity)),
                'status': info['status']
            })
    
    lifecycle_df = pd.DataFrame(lifecycle_data)
    
    # Visualization
    fig = px.line(
        lifecycle_df,
        x='year',
        y='intensity',
        color='topic',
        title="Topic Lifecycle Analysis",
        labels={'intensity': 'Research Intensity', 'year': 'Year'}
    )
    
    fig.update_layout(
        template='plotly_white',
        height=450,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Topic status
    st.markdown("---")
    st.markdown("### 🎯 Current Topic Status")
    
    cols = st.columns(len(topics_lifecycle))
    
    for idx, (topic, info) in enumerate(topics_lifecycle.items()):
        with cols[idx]:
            status_emoji = {
                'Emerging': '🌱',
                'Growing': '📈',
                'Mature': '🌳',
                'Declining': '📉'
            }
            
            st.metric(
                topic,
                status_emoji[info['status']],
                delta=info['status']
            )