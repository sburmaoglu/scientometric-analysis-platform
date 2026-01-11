"""Advanced Analytics Page - Link Prediction, Entropy, Divergence, TRL"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
from collections import Counter
import networkx as nx

def render():
    """Render advanced analytics page"""
    
    st.title("🔬 Advanced Analytics")
    st.markdown("Advanced scientometric analysis methods")
    
    if st.session_state.publications_data is None and st.session_state.patents_data is None:
        st.warning("⚠️ Please upload data first")
        return
    
    st.markdown("---")
    
    # Analysis selection
    analysis_type = st.selectbox(
        "Select Analysis Type",
        [
            "Shannon Entropy Analysis",
            "KL Divergence Analysis",
            "Technology Readiness Level (TRL)",
            "Publication-Patent Time Lag",
            "Link Prediction"
        ]
    )
    
    st.markdown("---")
    
    if analysis_type == "Shannon Entropy Analysis":
        render_shannon_entropy()
    
    elif analysis_type == "KL Divergence Analysis":
        render_divergence_analysis()
    
    elif analysis_type == "Technology Readiness Level (TRL)":
        render_trl_analysis()
    
    elif analysis_type == "Publication-Patent Time Lag":
        render_time_lag_analysis()
    
    elif analysis_type == "Link Prediction":
        render_link_prediction()

def render_shannon_entropy():
    """Shannon entropy analysis for diversity measurement"""
    
    st.subheader("📊 Shannon Entropy Analysis")
    st.markdown("""
    **Shannon Entropy** measures the diversity/uncertainty in a distribution.
    Higher entropy = more diverse/uncertain, Lower entropy = more concentrated.
    
    **Formula:** H(X) = -Σ p(x) log₂ p(x)
    """)
    
    # Choose dataset
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
    
    st.markdown("---")
    
    # Calculate entropy for different dimensions
    st.markdown("### 📈 Entropy Metrics")
    
    results = {}
    
    # Temporal entropy
    if 'year' in df.columns:
        year_dist = df['year'].value_counts(normalize=True)
        temporal_entropy = entropy(year_dist, base=2)
        results['Temporal Entropy'] = temporal_entropy
    
    # Author/Inventor diversity
    if dataset == "Publications" and 'author' in df.columns:
        all_authors = []
        for authors in df['author'].dropna():
            all_authors.extend(str(authors).split(';'))
        author_dist = pd.Series(all_authors).value_counts(normalize=True)
        author_entropy = entropy(author_dist, base=2)
        results['Author Diversity Entropy'] = author_entropy
    
    elif dataset == "Patents" and 'inventor' in df.columns:
        all_inventors = []
        for inventors in df['inventor'].dropna():
            all_inventors.extend(str(inventors).split(';'))
        inventor_dist = pd.Series(all_inventors).value_counts(normalize=True)
        results['Inventor Diversity Entropy'] = inventor_dist
    
    # Geographic entropy
    geo_col = 'country' if dataset == "Publications" else 'jurisdiction'
    if geo_col in df.columns:
        geo_dist = df[geo_col].value_counts(normalize=True)
        geo_entropy = entropy(geo_dist, base=2)
        results['Geographic Entropy'] = geo_entropy
    
    # Technology entropy (for patents)
    if dataset == "Patents" and 'ipc_class' in df.columns:
        all_classes = []
        for classes in df['ipc_class'].dropna():
            classes_list = str(classes).split(';')
            all_classes.extend([c.strip()[:4] for c in classes_list])
        tech_dist = pd.Series(all_classes).value_counts(normalize=True)
        tech_entropy = entropy(tech_dist, base=2)
        results['Technology Entropy'] = tech_entropy
    
    # Display results
    cols = st.columns(len(results))
    for idx, (metric, value) in enumerate(results.items()):
        with cols[idx]:
            st.metric(metric, f"{value:.3f}")
    
    st.markdown("---")
    
    # Visualization
    st.markdown("### 📊 Entropy Over Time")
    
    if 'year' in df.columns:
        years = sorted(df['year'].dropna().unique())
        entropy_over_time = []
        
        for year in years:
            year_df = df[df['year'] == year]
            
            # Calculate entropy for this year's geographic distribution
            if geo_col in year_df.columns:
                dist = year_df[geo_col].value_counts(normalize=True)
                year_entropy = entropy(dist, base=2)
                entropy_over_time.append(year_entropy)
            else:
                entropy_over_time.append(np.nan)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=years,
            y=entropy_over_time,
            mode='lines+markers',
            name='Geographic Entropy',
            line=dict(width=3, color='#667eea')
        ))
        
        fig.update_layout(
            title="Geographic Entropy Over Time",
            xaxis_title="Year",
            yaxis_title="Shannon Entropy (bits)",
            template='plotly_white',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("""
        **Interpretation:**
        - **Increasing entropy** = Growing diversity/internationalization
        - **Decreasing entropy** = Increasing concentration
        - **Stable entropy** = Consistent distribution pattern
        """)

def render_divergence_analysis():
    """KL Divergence and JS Distance analysis"""
    
    st.subheader("📊 Divergence Analysis")
    st.markdown("""
    **Kullback-Leibler (KL) Divergence** measures how one probability distribution 
    differs from another reference distribution.
    
    **Jensen-Shannon Distance** is a symmetric version of KL divergence.
    """)
    
    # Need both datasets
    if st.session_state.publications_data is None or st.session_state.patents_data is None:
        st.warning("⚠️ Both publications and patents data required")
        return
    
    pubs_df = st.session_state.publications_data
    pats_df = st.session_state.patents_data
    
    st.markdown("---")
    
    # Temporal divergence
    st.markdown("### 📈 Temporal Distribution Divergence")
    
    if 'year' in pubs_df.columns and 'year' in pats_df.columns:
        # Get common year range
        all_years = sorted(set(pubs_df['year'].dropna()) | set(pats_df['year'].dropna()))
        
        pubs_temporal = pubs_df['year'].value_counts()
        pats_temporal = pats_df['year'].value_counts()
        
        # Normalize to probability distributions
        pubs_dist = pd.Series([pubs_temporal.get(y, 0) for y in all_years])
        pats_dist = pd.Series([pats_temporal.get(y, 0) for y in all_years])
        
        pubs_dist = pubs_dist / pubs_dist.sum()
        pats_dist = pats_dist / pats_dist.sum()
        
        # Calculate JS distance
        js_distance = jensenshannon(pubs_dist, pats_dist)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Jensen-Shannon Distance", f"{js_distance:.4f}")
        
        with col2:
            similarity = 1 - js_distance
            st.metric("Distribution Similarity", f"{similarity:.2%}")
        
        # Visualization
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=all_years,
            y=pubs_dist,
            mode='lines+markers',
            name='Publications',
            line=dict(width=3, color='#3498db')
        ))
        
        fig.add_trace(go.Scatter(
            x=all_years,
            y=pats_dist,
            mode='lines+markers',
            name='Patents',
            line=dict(width=3, color='#e74c3c')
        ))
        
        fig.update_layout(
            title="Temporal Distribution Comparison",
            xaxis_title="Year",
            yaxis_title="Probability",
            template='plotly_white',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.info("""
        **Interpretation:**
        - **JS Distance = 0**: Identical distributions
        - **JS Distance = 1**: Completely different distributions
        - **Typical range**: 0.1-0.4 for related fields
        """)

def render_trl_analysis():
    """Technology Readiness Level analysis"""
    
    st.subheader("🚀 Technology Readiness Level (TRL) Analysis")
    st.markdown("""
    **Technology Readiness Level (TRL)** is a method for estimating technology maturity.
    
    **Scale:**
    - TRL 1-3: Basic research (publications dominant)
    - TRL 4-6: Technology development (mixed)
    - TRL 7-9: System deployment (patents dominant)
    """)
    
    if st.session_state.publications_data is None or st.session_state.patents_data is None:
        st.warning("⚠️ Both publications and patents data required")
        return
    
    pubs_df = st.session_state.publications_data
    pats_df = st.session_state.patents_data
    
    if 'year' not in pubs_df.columns or 'year' not in pats_df.columns:
        st.warning("Year data required")
        return
    
    st.markdown("---")
    
    # Calculate TRL proxy based on pub/patent ratio
    st.markdown("### 📊 TRL Evolution Over Time")
    
    years = sorted(set(pubs_df['year'].dropna()) & set(pats_df['year'].dropna()))
    
    trl_data = []
    
    for year in years:
        pub_count = len(pubs_df[pubs_df['year'] == year])
        pat_count = len(pats_df[pats_df['year'] == year])
        
        total = pub_count + pat_count
        if total > 0:
            patent_ratio = pat_count / total
            
            # TRL estimation (simplified)
            if patent_ratio < 0.2:
                trl = 1 + (patent_ratio / 0.2) * 2  # TRL 1-3
            elif patent_ratio < 0.5:
                trl = 3 + ((patent_ratio - 0.2) / 0.3) * 3  # TRL 3-6
            else:
                trl = 6 + ((patent_ratio - 0.5) / 0.5) * 3  # TRL 6-9
            
            trl_data.append({
                'year': year,
                'trl': trl,
                'pub_count': pub_count,
                'pat_count': pat_count,
                'patent_ratio': patent_ratio
            })
    
    trl_df = pd.DataFrame(trl_data)
    
    # Visualization
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=trl_df['year'],
        y=trl_df['trl'],
        mode='lines+markers',
        name='TRL',
        line=dict(width=3, color='#667eea'),
        marker=dict(size=10)
    ))
    
    # Add TRL zones
    fig.add_hrect(y0=1, y1=3, fillcolor="lightblue", opacity=0.2, 
                  annotation_text="Basic Research", annotation_position="right")
    fig.add_hrect(y0=3, y1=6, fillcolor="lightyellow", opacity=0.2,
                  annotation_text="Development", annotation_position="right")
    fig.add_hrect(y0=6, y1=9, fillcolor="lightgreen", opacity=0.2,
                  annotation_text="Deployment", annotation_position="right")
    
    fig.update_layout(
        title="Technology Readiness Level Evolution",
        xaxis_title="Year",
        yaxis_title="TRL",
        yaxis_range=[0, 10],
        template='plotly_white',
        height=450
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Current TRL
    if len(trl_df) > 0:
        current_trl = trl_df.iloc[-1]['trl']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Current TRL", f"{current_trl:.1f}")
        
        with col2:
            if len(trl_df) > 1:
                trl_change = trl_df.iloc[-1]['trl'] - trl_df.iloc[0]['trl']
                st.metric("TRL Change", f"{trl_change:+.1f}")
        
        with col3:
            latest_ratio = trl_df.iloc[-1]['patent_ratio']
            st.metric("Patent Ratio", f"{latest_ratio:.1%}")

def render_time_lag_analysis():
    """Analyze time lag between publications and patents"""
    
    st.subheader("⏱️ Publication-Patent Time Lag Analysis")
    st.markdown("""
    **Time Lag Analysis** measures the time between scientific publication 
    and patent filing, indicating technology transfer speed.
    """)
    
    if st.session_state.publications_data is None or st.session_state.patents_data is None:
        st.warning("⚠️ Both publications and patents data required")
        return
    
    pubs_df = st.session_state.publications_data
    pats_df = st.session_state.patents_data
    
    if 'year' not in pubs_df.columns or 'year' not in pats_df.columns:
        st.warning("Year data required")
        return
    
    st.markdown("---")
    
    # Calculate average time lag per year
    st.markdown("### 📊 Average Time Lag Over Time")
    
    pub_years = pubs_df.groupby('year').size().reset_index(name='pub_count')
    pat_years = pats_df.groupby('year').size().reset_index(name='pat_count')
    
    merged = pd.merge(pub_years, pat_years, on='year', how='outer').fillna(0)
    
    # Calculate cumulative lag
    merged['pub_cumsum'] = merged['pub_count'].cumsum()
    merged['pat_cumsum'] = merged['pat_count'].cumsum()
    
    # Estimate lag (simplified method)
    lags = []
    for idx, row in merged.iterrows():
        if row['pat_count'] > 0 and idx > 0:
            # Find when similar publication volume occurred
            pub_cumsum = row['pub_cumsum']
            # Look back to find matching publication volume
            lag_years = 0
            for past_idx in range(idx-1, -1, -1):
                if merged.iloc[past_idx]['pub_cumsum'] <= pub_cumsum * 0.9:
                    lag_years = idx - past_idx
                    break
            lags.append(lag_years)
        else:
            lags.append(np.nan)
    
    merged['estimated_lag'] = lags
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=merged['year'],
        y=merged['estimated_lag'],
        mode='lines+markers',
        name='Time Lag',
        line=dict(width=3, color='#9b59b6'),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title="Estimated Time Lag Between Publications and Patents",
        xaxis_title="Year",
        yaxis_title="Lag (years)",
        template='plotly_white',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    valid_lags = merged['estimated_lag'].dropna()
    if len(valid_lags) > 0:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Average Lag", f"{valid_lags.mean():.1f} years")
        
        with col2:
            st.metric("Median Lag", f"{valid_lags.median():.1f} years")
        
        with col3:
            st.metric("Min Lag", f"{valid_lags.min():.1f} years")

def render_link_prediction():
    """Link prediction analysis for collaboration/citation networks and keyword co-occurrences"""
    
    st.subheader("🔗 Link Prediction Analysis")
    st.markdown("""
    **Link Prediction** identifies potential future connections based on network structure.
    
    **Available Networks:**
    - **Collaboration Networks**: Author/Inventor co-authorships
    - **Keyword Networks**: Keyword co-occurrences
    - **Hybrid Networks**: Combined collaboration + keyword patterns
    """)
    
    # Network type selection
    network_type = st.radio(
        "Select Network Type",
        ["👥 Collaboration Network", "🏷️ Keyword Co-occurrence Network", "🔀 Hybrid Network"],
        horizontal=True
    )
    
    dataset = st.radio("Select Dataset", ["Publications", "Patents"], horizontal=True)
    
    if dataset == "Publications":
        if st.session_state.publications_data is None:
            st.warning("⚠️ Upload publications data")
            return
        df = st.session_state.publications_data
        entity_col = 'author'
        keyword_col = 'keywords'
    else:
        if st.session_state.patents_data is None:
            st.warning("⚠️ Upload patents data")
            return
        df = st.session_state.patents_data
        entity_col = 'inventor'
        keyword_col = 'ipc_class'  # or could use abstract keywords
    
    st.markdown("---")
    
    if network_type == "👥 Collaboration Network":
        render_collaboration_network(df, entity_col, dataset)
    
    elif network_type == "🏷️ Keyword Co-occurrence Network":
        render_keyword_network(df, keyword_col, dataset)
    
    elif network_type == "🔀 Hybrid Network":
        render_hybrid_network(df, entity_col, keyword_col, dataset)

def render_collaboration_network(df, entity_col, dataset):
    """Collaboration network link prediction"""
    
    st.markdown("### 👥 Collaboration Network Analysis")
    
    if entity_col not in df.columns:
        st.warning(f"{entity_col} data not available")
        return
    
    import networkx as nx
    
    # Build collaboration network
    with st.spinner("Building collaboration network..."):
        G = nx.Graph()
        
        for entities in df[entity_col].dropna():
            entity_list = [e.strip() for e in str(entities).split(';') if e.strip()]
            
            # Add edges between all pairs
            for i in range(len(entity_list)):
                for j in range(i + 1, len(entity_list)):
                    if G.has_edge(entity_list[i], entity_list[j]):
                        G[entity_list[i]][entity_list[j]]['weight'] += 1
                    else:
                        G.add_edge(entity_list[i], entity_list[j], weight=1)
    
    st.success(f"✅ Network built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    
    # Network metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Nodes", G.number_of_nodes())
    
    with col2:
        st.metric("Total Edges", G.number_of_edges())
    
    with col3:
        density = nx.density(G)
        st.metric("Density", f"{density:.4f}")
    
    with col4:
        if len(G) > 0:
            avg_degree = sum(dict(G.degree()).values()) / len(G)
            st.metric("Avg Degree", f"{avg_degree:.2f}")
    
    st.markdown("---")
    
    # Link prediction
    st.markdown("### 🔮 Predicted Future Collaborations")
    
    method = st.selectbox(
        "Prediction Method",
        ["Common Neighbors", "Jaccard Coefficient", "Adamic-Adar Index", "Preferential Attachment"]
    )
    
    top_n = st.slider("Number of Predictions", 10, 50, 20)
    
    with st.spinner(f"Calculating {method}..."):
        if method == "Common Neighbors":
            predictions = list(nx.common_neighbor_centrality(G))
        elif method == "Jaccard Coefficient":
            predictions = list(nx.jaccard_coefficient(G))
        elif method == "Adamic-Adar Index":
            predictions = list(nx.adamic_adar_index(G))
        else:  # Preferential Attachment
            predictions = list(nx.preferential_attachment(G))
        
        # Sort by score
        predictions = sorted(predictions, key=lambda x: x[2], reverse=True)[:top_n]
    
    # Display predictions
    pred_df = pd.DataFrame(predictions, columns=['Entity 1', 'Entity 2', 'Score'])
    pred_df['Score'] = pred_df['Score'].round(4)
    
    st.dataframe(pred_df, use_container_width=True, hide_index=True)
    
    # Visualization
    st.markdown("---")
    st.markdown("### 📊 Score Distribution")
    
    fig = px.histogram(
        pred_df,
        x='Score',
        nbins=20,
        title=f"Distribution of {method} Scores",
        labels={'Score': 'Prediction Score', 'count': 'Frequency'}
    )
    
    fig.update_layout(template='plotly_white', height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Download predictions
    csv = pred_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download Predictions",
        csv,
        f"collaboration_predictions_{dataset.lower()}.csv",
        "text/csv"
    )

def render_keyword_network(df, keyword_col, dataset):
    """Keyword co-occurrence network link prediction"""
    
    st.markdown("### 🏷️ Keyword Co-occurrence Network")
    st.markdown("""
    Analyzes which keywords frequently appear together and predicts future keyword associations.
    Useful for identifying emerging research themes and topic convergence.
    """)
    
    # Check for keywords
    if keyword_col not in df.columns:
        # Try alternative keyword columns
        alt_cols = ['keywords', 'author_keywords', 'Keywords', 'abstract']
        keyword_col = None
        for col in alt_cols:
            if col in df.columns:
                keyword_col = col
                break
        
        if keyword_col is None:
            st.warning("No keyword data available. Extracting keywords from abstracts...")
            
            if 'abstract' in df.columns:
                keyword_col = 'abstract'
                df = extract_keywords_from_text(df, 'abstract')
                keyword_col = 'extracted_keywords'
            else:
                st.error("No text data available for keyword extraction")
                return
    
    st.info(f"Using column: **{keyword_col}**")
    
    # Build keyword co-occurrence network
    with st.spinner("Building keyword network..."):
        import networkx as nx
        
        G = nx.Graph()
        
        # Filter parameters
        col1, col2 = st.columns(2)
        
        with col1:
            min_cooccurrence = st.slider("Minimum Co-occurrence", 1, 10, 2,
                                        help="Minimum times keywords must appear together")
        
        with col2:
            top_keywords = st.slider("Top Keywords to Include", 50, 500, 100,
                                    help="Focus on most frequent keywords")
        
        # Extract all keywords
        all_keywords = []
        keyword_docs = []
        
        for keywords_str in df[keyword_col].dropna():
            if pd.notna(keywords_str):
                # Split by semicolon or comma
                keywords = re.split(r'[;,]', str(keywords_str))
                keywords = [k.strip().lower() for k in keywords if k.strip()]
                
                if len(keywords) > 0:
                    all_keywords.extend(keywords)
                    keyword_docs.append(keywords)
        
        # Get top keywords
        keyword_freq = Counter(all_keywords)
        top_keyword_list = [k for k, _ in keyword_freq.most_common(top_keywords)]
        
        # Build edges between keywords that co-occur
        for keywords in keyword_docs:
            # Filter to top keywords
            keywords = [k for k in keywords if k in top_keyword_list]
            
            # Add edges
            for i in range(len(keywords)):
                for j in range(i + 1, len(keywords)):
                    if G.has_edge(keywords[i], keywords[j]):
                        G[keywords[i]][keywords[j]]['weight'] += 1
                    else:
                        G.add_edge(keywords[i], keywords[j], weight=1)
        
        # Filter edges by minimum co-occurrence
        edges_to_remove = [(u, v) for u, v, d in G.edges(data=True) 
                          if d['weight'] < min_cooccurrence]
        G.remove_edges_from(edges_to_remove)
        
        # Remove isolated nodes
        G.remove_nodes_from(list(nx.isolates(G)))
    
    st.success(f"✅ Network built: {G.number_of_nodes()} keywords, {G.number_of_edges()} co-occurrences")
    
    # Network metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Unique Keywords", G.number_of_nodes())
    
    with col2:
        st.metric("Co-occurrences", G.number_of_edges())
    
    with col3:
        density = nx.density(G)
        st.metric("Network Density", f"{density:.4f}")
    
    with col4:
        if len(G) > 0:
            components = nx.number_connected_components(G)
            st.metric("Topic Clusters", components)
    
    st.markdown("---")
    
    # Most central keywords
    st.markdown("### ⭐ Most Central Keywords")
    
    if len(G) > 0:
        degree_centrality = nx.degree_centrality(G)
        betweenness = nx.betweenness_centrality(G)
        
        central_keywords = pd.DataFrame([
            {
                'Keyword': keyword,
                'Degree Centrality': degree_centrality[keyword],
                'Betweenness': betweenness[keyword],
                'Connections': G.degree(keyword)
            }
            for keyword in list(degree_centrality.keys())[:20]
        ]).sort_values('Degree Centrality', ascending=False)
        
        st.dataframe(central_keywords, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Link prediction for keywords
    st.markdown("### 🔮 Predicted Future Keyword Associations")
    
    method = st.selectbox(
        "Prediction Method",
        ["Common Neighbors", "Jaccard Coefficient", "Adamic-Adar Index", "Resource Allocation"]
    )
    
    top_n = st.slider("Number of Predictions", 10, 50, 20)
    
    with st.spinner(f"Calculating {method}..."):
        if method == "Common Neighbors":
            predictions = list(nx.common_neighbor_centrality(G))
        elif method == "Jaccard Coefficient":
            predictions = list(nx.jaccard_coefficient(G))
        elif method == "Adamic-Adar Index":
            predictions = list(nx.adamic_adar_index(G))
        else:  # Resource Allocation
            predictions = list(nx.resource_allocation_index(G))
        
        # Sort by score
        predictions = sorted(predictions, key=lambda x: x[2], reverse=True)[:top_n]
    
    # Display predictions
    pred_df = pd.DataFrame(predictions, columns=['Keyword 1', 'Keyword 2', 'Association Score'])
    pred_df['Association Score'] = pred_df['Association Score'].round(4)
    
    st.dataframe(pred_df, use_container_width=True, hide_index=True)
    
    st.info("""
    **Interpretation:**
    - **High scores** = Keywords likely to co-occur in future research
    - Identifies emerging research themes and topic convergence
    - Useful for forecasting interdisciplinary connections
    """)
    
    # Visualize top predictions
    st.markdown("---")
    st.markdown("### 📊 Top Predicted Associations")
    
    top_10 = pred_df.head(10).copy()
    top_10['Pair'] = top_10['Keyword 1'] + ' ↔ ' + top_10['Keyword 2']
    
    fig = px.bar(
        top_10,
        x='Association Score',
        y='Pair',
        orientation='h',
        title="Top 10 Predicted Keyword Associations",
        color='Association Score',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        template='plotly_white',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Download predictions
    csv = pred_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        "📥 Download Keyword Predictions",
        csv,
        f"keyword_predictions_{dataset.lower()}.csv",
        "text/csv"
    )

def render_hybrid_network(df, entity_col, keyword_col, dataset):
    """Hybrid network combining collaboration and keyword patterns"""
    
    st.markdown("### 🔀 Hybrid Network Analysis")
    st.markdown("""
    Combines **collaboration patterns** with **keyword co-occurrences** to predict:
    - Researchers likely to collaborate based on shared research interests
    - Keyword associations strengthened by author collaborations
    """)
    
    if entity_col not in df.columns:
        st.warning(f"{entity_col} data not available")
        return
    
    import networkx as nx
    
    # Build bipartite network (authors-keywords)
    with st.spinner("Building hybrid network..."):
        B = nx.Graph()
        
        # Add nodes with bipartite attribute
        authors = set()
        keywords = set()
        
        for idx, row in df.iterrows():
            if pd.notna(row.get(entity_col)) and pd.notna(row.get(keyword_col)):
                # Get authors
                author_list = [a.strip() for a in str(row[entity_col]).split(';') if a.strip()]
                
                # Get keywords
                keyword_list = re.split(r'[;,]', str(row[keyword_col]))
                keyword_list = [k.strip().lower() for k in keyword_list if k.strip()]
                
                authors.update(author_list)
                keywords.update(keyword_list[:5])  # Limit keywords per document
                
                # Add edges between authors and keywords
                for author in author_list:
                    for keyword in keyword_list[:5]:
                        if B.has_edge(author, keyword):
                            B[author][keyword]['weight'] += 1
                        else:
                            B.add_edge(author, keyword, weight=1)
        
        # Set bipartite attribute
        B.add_nodes_from(authors, bipartite=0)
        B.add_nodes_from(keywords, bipartite=1)
    
    st.success(f"✅ Hybrid network: {len(authors)} authors, {len(keywords)} keywords, {B.number_of_edges()} connections")
    
    # Network metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Authors", len(authors))
    
    with col2:
        st.metric("Keywords", len(keywords))
    
    with col3:
        st.metric("Connections", B.number_of_edges())
    
    st.markdown("---")
    
    # Project to author-author network based on shared keywords
    st.markdown("### 👥 Author Collaboration Prediction (Based on Shared Keywords)")
    
    with st.spinner("Projecting to author network..."):
        from networkx.algorithms import bipartite
        
        # Project onto authors
        author_nodes = {n for n, d in B.nodes(data=True) if d.get('bipartite') == 0}
        G_authors = bipartite.weighted_projected_graph(B, author_nodes)
    
    st.info(f"Projected network: {G_authors.number_of_nodes()} authors, {G_authors.number_of_edges()} potential collaborations")
    
    # Calculate predictions
    method = st.selectbox(
        "Prediction Method",
        ["Common Neighbors", "Adamic-Adar Index", "Resource Allocation"]
    )
    
    with st.spinner(f"Calculating {method}..."):
        if method == "Common Neighbors":
            predictions = list(nx.common_neighbor_centrality(G_authors))
        elif method == "Adamic-Adar Index":
            predictions = list(nx.adamic_adar_index(G_authors))
        else:  # Resource Allocation
            predictions = list(nx.resource_allocation_index(G_authors))
        
        predictions = sorted(predictions, key=lambda x: x[2], reverse=True)[:20]
    
    # Display predictions
    pred_df = pd.DataFrame(predictions, columns=['Author 1', 'Author 2', 'Score'])
    pred_df['Score'] = pred_df['Score'].round(4)
    
    st.dataframe(pred_df, use_container_width=True, hide_index=True)
    
    st.success("""
    ✅ **Hybrid Prediction Advantage:**
    - Identifies collaborations based on **shared research interests**
    - More accurate than structure-only predictions
    - Reveals interdisciplinary research opportunities
    """)

def extract_keywords_from_text(df, text_col, n_keywords=5):
    """Extract keywords from text using TF-IDF"""
    
    from sklearn.feature_extraction.text import TfidfVectorizer
    import re
    
    # Get texts
    texts = df[text_col].dropna().astype(str).tolist()
    
    # TF-IDF
    vectorizer = TfidfVectorizer(
        max_features=100,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=2
    )
    
    try:
        tfidf_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        
        # Extract top keywords per document
        keywords_list = []
        for doc_idx in range(len(texts)):
            doc_tfidf = tfidf_matrix[doc_idx].toarray()[0]
            top_indices = doc_tfidf.argsort()[-n_keywords:][::-1]
            keywords = [feature_names[i] for i in top_indices if doc_tfidf[i] > 0]
            keywords_list.append('; '.join(keywords))
        
        df['extracted_keywords'] = keywords_list
        
    except:
        df['extracted_keywords'] = ''
    
    return df
