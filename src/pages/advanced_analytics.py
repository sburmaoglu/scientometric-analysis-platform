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
    """Link prediction analysis for collaboration/citation networks"""
    
    st.subheader("🔗 Link Prediction Analysis")
    st.markdown("""
    **Link Prediction** identifies potential future collaborations or citations 
    based on network structure and patterns.
    
    **Methods:**
    - Common Neighbors
    - Preferential Attachment
    - Adamic-Adar Index
    """)
    
    dataset = st.radio("Select Dataset", ["Publications", "Patents"], horizontal=True)
    
    if dataset == "Publications":
        if st.session_state.publications_data is None:
            st.warning("⚠️ Upload publications data")
            return
        df = st.session_state.publications_data
        entity_col = 'author'
    else:
        if st.session_state.patents_data is None:
            st.warning("⚠️ Upload patents data")
            return
        df = st.session_state.patents_data
        entity_col = 'inventor'
    
    if entity_col not in df.columns:
        st.warning(f"{entity_col} data not available")
        return
    
    st.markdown("---")
    
    # Build collaboration network
    st.markdown("### 🕸️ Collaboration Network Construction")
    
    with st.spinner("Building collaboration network..."):
        G = nx.Graph()
        
        # Add edges for co-authors/co-inventors
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
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Nodes", G.number_of_nodes())
    
    with col2:
        st.metric("Total Edges", G.number_of_edges())
    
    with col3:
        density = nx.density(G)
        st.metric("Network Density", f"{density:.4f}")
    
    st.markdown("---")
    
    # Link prediction
    st.markdown("### 🔮 Predicted Future Collaborations")
    
    method = st.selectbox(
        "Prediction Method",
        ["Common Neighbors", "Adamic-Adar Index", "Preferential Attachment"]
    )
    
    with st.spinner(f"Calculating {method}..."):
        if method == "Common Neighbors":
            predictions = list(nx.common_neighbor_centrality(G))
        elif method == "Adamic-Adar Index":
            predictions = list(nx.adamic_adar_index(G))
        else:  # Preferential Attachment
            predictions = list(nx.preferential_attachment(G))
        
        # Sort by score
        predictions = sorted(predictions, key=lambda x: x[2], reverse=True)[:20]
    
    # Display predictions
    pred_df = pd.DataFrame(predictions, columns=['Entity 1', 'Entity 2', 'Score'])
    
    st.dataframe(pred_df, use_container_width=True, hide_index=True)
    
    st.info("""
    **Interpretation:**
    - **Higher scores** = More likely future collaboration
    - These predictions are based on current network structure
    - Consider domain expertise when evaluating predictions
    """)