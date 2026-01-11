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
import re

def render():
    """Render advanced analytics page"""
    
    st.title("🔬 Advanced Analytics")
    st.markdown("Advanced scientometric analysis methods with flexible unit selection")
    
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

def get_available_units(df, dataset_type):
    """Get available analysis units for the dataset"""
    
    units = {}
    
    # Always available
    if 'year' in df.columns:
        units['Temporal'] = 'year'
    
    # Entity-based (authors/inventors)
    if dataset_type == 'publications':
        if 'author' in df.columns:
            units['Authors'] = 'author'
    else:
        if 'inventor' in df.columns:
            units['Inventors'] = 'inventor'
        if 'assignee' in df.columns:
            units['Organizations'] = 'assignee'
    
    # Keyword-based
    keyword_candidates = ['keywords', 'author_keywords', 'Keywords', 'ipc_class', 'cpc_class']
    for col in keyword_candidates:
        if col in df.columns:
            units['Keywords'] = col
            break
    
    # Geographic
    geo_col = 'country' if dataset_type == 'publications' else 'jurisdiction'
    if geo_col in df.columns:
        units['Geographic'] = geo_col
    
    # Journal/Technology
    if dataset_type == 'publications':
        if 'journal' in df.columns:
            units['Journals'] = 'journal'
    else:
        if 'ipc_class' in df.columns:
            units['Technology Classes'] = 'ipc_class'
    
    return units

def parse_entity_list(entity_str, separator=';'):
    """Parse semicolon or comma separated entities"""
    if pd.isna(entity_str):
        return []
    
    # Try semicolon first, then comma
    entities = re.split(r'[;,]', str(entity_str))
    entities = [e.strip().lower() for e in entities if e.strip()]
    
    # For IPC/CPC classes, take first 4 characters
    if len(entities) > 0 and len(entities[0]) > 4 and entities[0][0].isalpha():
        entities = [e[:4] for e in entities]
    
    return entities

def render_shannon_entropy():
    """Shannon entropy analysis with unit selection"""
    
    st.subheader("📊 Shannon Entropy Analysis")
    st.markdown("""
    **Shannon Entropy** measures diversity/uncertainty in a distribution.
    
    **Formula:** H(X) = -Σ p(x) log₂ p(x)
    
    - **Higher entropy** = More diverse/uncertain
    - **Lower entropy** = More concentrated
    """)
    
    # Dataset selection
    dataset = st.radio("Select Dataset", ["Publications", "Patents"], horizontal=True)
    
    if dataset == "Publications":
        if st.session_state.publications_data is None:
            st.warning("⚠️ Upload publications data")
            return
        df = st.session_state.publications_data
        dataset_type = 'publications'
    else:
        if st.session_state.patents_data is None:
            st.warning("⚠️ Upload patents data")
            return
        df = st.session_state.patents_data
        dataset_type = 'patents'
    
    # Get available units
    available_units = get_available_units(df, dataset_type)
    
    if not available_units:
        st.error("No analysis units available in this dataset")
        return
    
    st.markdown("---")
    
    # Unit selection
    st.markdown("### 🎯 Select Analysis Units")
    
    selected_units = st.multiselect(
        "Choose units to analyze",
        list(available_units.keys()),
        default=list(available_units.keys())[:2] if len(available_units) >= 2 else list(available_units.keys())
    )
    
    if not selected_units:
        st.warning("Please select at least one analysis unit")
        return
    
    st.markdown("---")
    
    # Calculate entropy for each selected unit
    st.markdown("### 📈 Entropy Metrics")
    
    results = {}
    
    for unit_name in selected_units:
        col_name = available_units[unit_name]
        
        if col_name in df.columns:
            # Handle list-based columns (authors, keywords, etc.)
            if unit_name in ['Authors', 'Inventors', 'Organizations', 'Keywords', 'Technology Classes']:
                all_items = []
                for item_str in df[col_name].dropna():
                    all_items.extend(parse_entity_list(item_str))
                
                if all_items:
                    item_dist = pd.Series(all_items).value_counts(normalize=True)
                    unit_entropy = entropy(item_dist, base=2)
                    results[unit_name] = unit_entropy
            else:
                # Direct columns (year, journal, geographic)
                item_dist = df[col_name].value_counts(normalize=True)
                if len(item_dist) > 0:
                    unit_entropy = entropy(item_dist, base=2)
                    results[unit_name] = unit_entropy
    
    # Display results
    if results:
        cols = st.columns(min(len(results), 4))
        for idx, (metric, value) in enumerate(results.items()):
            with cols[idx % 4]:
                st.metric(f"{metric} Entropy", f"{value:.3f} bits")
    
    st.markdown("---")
    
    # Temporal entropy evolution (if year available)
    if 'year' in df.columns and len(selected_units) > 0:
        st.markdown("### 📊 Entropy Evolution Over Time")
        
        # Let user select which unit to track over time
        unit_to_track = st.selectbox(
            "Select unit to track over time",
            [u for u in selected_units if u != 'Temporal']
        )
        
        if unit_to_track:
            col_name = available_units[unit_to_track]
            
            years = sorted(df['year'].dropna().unique())
            entropy_over_time = []
            
            for year in years:
                year_df = df[df['year'] == year]
                
                # Calculate entropy for this year
                if unit_to_track in ['Authors', 'Inventors', 'Organizations', 'Keywords', 'Technology Classes']:
                    all_items = []
                    for item_str in year_df[col_name].dropna():
                        all_items.extend(parse_entity_list(item_str))
                    
                    if all_items:
                        item_dist = pd.Series(all_items).value_counts(normalize=True)
                        year_entropy = entropy(item_dist, base=2)
                        entropy_over_time.append(year_entropy)
                    else:
                        entropy_over_time.append(np.nan)
                else:
                    item_dist = year_df[col_name].value_counts(normalize=True)
                    if len(item_dist) > 1:
                        year_entropy = entropy(item_dist, base=2)
                        entropy_over_time.append(year_entropy)
                    else:
                        entropy_over_time.append(np.nan)
            
            # Plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=years,
                y=entropy_over_time,
                mode='lines+markers',
                name=f'{unit_to_track} Entropy',
                line=dict(width=3, color='#667eea'),
                marker=dict(size=8)
            ))
            
            fig.update_layout(
                title=f"{unit_to_track} Entropy Over Time",
                xaxis_title="Year",
                yaxis_title="Shannon Entropy (bits)",
                template='plotly_white',
                height=400,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("""
            **Interpretation:**
            - **Increasing entropy** → Growing diversity/internationalization
            - **Decreasing entropy** → Increasing concentration
            - **Stable entropy** → Consistent distribution pattern
            """)
    
    # Comparative bar chart
    if len(results) > 1:
        st.markdown("---")
        st.markdown("### 📊 Comparative Entropy")
        
        results_df = pd.DataFrame(list(results.items()), columns=['Unit', 'Entropy'])
        
        fig = px.bar(
            results_df,
            x='Unit',
            y='Entropy',
            title="Entropy Comparison Across Units",
            color='Entropy',
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(template='plotly_white', height=400)
        st.plotly_chart(fig, use_container_width=True)

def render_divergence_analysis():
    """KL Divergence with unit selection"""
    
    st.subheader("📊 Divergence Analysis")
    st.markdown("""
    **Jensen-Shannon Distance** measures how different two probability distributions are.
    
    **Range:** [0, 1]
    - **0** = Identical distributions
    - **1** = Completely different distributions
    """)
    
    # Need both datasets
    if st.session_state.publications_data is None or st.session_state.patents_data is None:
        st.warning("⚠️ Both publications and patents data required")
        return
    
    pubs_df = st.session_state.publications_data
    pats_df = st.session_state.patents_data
    
    st.markdown("---")
    
    # Get available units for comparison
    st.markdown("### 🎯 Select Comparison Unit")
    
    # Find common units
    pubs_units = get_available_units(pubs_df, 'publications')
    pats_units = get_available_units(pats_df, 'patents')
    
    # Common units that can be compared
    comparable_units = {}
    
    # Temporal is always comparable
    if 'Temporal' in pubs_units and 'Temporal' in pats_units:
        comparable_units['Temporal Distribution'] = ('year', 'year')
    
    # Geographic comparison
    if 'Geographic' in pubs_units and 'Geographic' in pats_units:
        comparable_units['Geographic Distribution'] = ('country', 'jurisdiction')
    
    # Keywords/Technology
    if 'Keywords' in pubs_units and 'Technology Classes' in pats_units:
        comparable_units['Keywords vs Technology'] = (pubs_units['Keywords'], pats_units['Technology Classes'])
    
    if not comparable_units:
        st.error("No comparable units found between publications and patents")
        return
    
    comparison_type = st.selectbox("Select comparison", list(comparable_units.keys()))
    
    pubs_col, pats_col = comparable_units[comparison_type]
    
    st.markdown("---")
    
    if st.button("🔍 Calculate Divergence", type="primary"):
        with st.spinner("Calculating divergence..."):
            # Get distributions
            if comparison_type in ['Keywords vs Technology']:
                # Parse lists
                pubs_items = []
                for item_str in pubs_df[pubs_col].dropna():
                    pubs_items.extend(parse_entity_list(item_str))
                
                pats_items = []
                for item_str in pats_df[pats_col].dropna():
                    pats_items.extend(parse_entity_list(item_str))
                
                # Get common vocabulary
                all_items = set(pubs_items + pats_items)
                
                pubs_counts = Counter(pubs_items)
                pats_counts = Counter(pats_items)
                
                # Create aligned distributions
                pubs_dist = np.array([pubs_counts.get(item, 0) for item in all_items])
                pats_dist = np.array([pats_counts.get(item, 0) for item in all_items])
                
            else:
                # Direct distributions
                if comparison_type == 'Temporal Distribution':
                    all_values = sorted(set(pubs_df[pubs_col].dropna()) | set(pats_df[pats_col].dropna()))
                else:
                    all_values = sorted(set(pubs_df[pubs_col].dropna()) | set(pats_df[pats_col].dropna()))
                
                pubs_counts = pubs_df[pubs_col].value_counts()
                pats_counts = pats_df[pats_col].value_counts()
                
                pubs_dist = np.array([pubs_counts.get(v, 0) for v in all_values])
                pats_dist = np.array([pats_counts.get(v, 0) for v in all_values])
            
            # Normalize
            pubs_dist = pubs_dist / pubs_dist.sum() if pubs_dist.sum() > 0 else pubs_dist
            pats_dist = pats_dist / pats_dist.sum() if pats_dist.sum() > 0 else pats_dist
            
            # Calculate JS distance
            js_distance = jensenshannon(pubs_dist, pats_dist)
            
            # Display result
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Jensen-Shannon Distance", f"{js_distance:.4f}")
            
            with col2:
                similarity = 1 - js_distance
                st.metric("Distribution Similarity", f"{similarity:.2%}")
            
            # Visualization
            if comparison_type == 'Temporal Distribution':
                st.markdown("### 📈 Distribution Comparison")
                
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=all_values,
                    y=pubs_dist,
                    mode='lines+markers',
                    name='Publications',
                    line=dict(width=3, color='#3498db')
                ))
                
                fig.add_trace(go.Scatter(
                    x=all_values,
                    y=pats_dist,
                    mode='lines+markers',
                    name='Patents',
                    line=dict(width=3, color='#e74c3c')
                ))
                
                fig.update_layout(
                    title=f"{comparison_type} Comparison",
                    xaxis_title=pubs_col.title(),
                    yaxis_title="Probability",
                    template='plotly_white',
                    height=450,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            st.info(f"""
            **Interpretation for {comparison_type}:**
            - **JS Distance = {js_distance:.3f}**
            - This indicates {"very similar" if js_distance < 0.2 else "moderately similar" if js_distance < 0.5 else "quite different"} distributions
            - Typical range for related fields: 0.1-0.4
            """)

def render_trl_analysis():
    """TRL analysis - keep as is from previous version"""
    
    st.subheader("🚀 Technology Readiness Level (TRL) Analysis")
    st.markdown("""
    **TRL Scale:**
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
    
    # Calculate TRL
    years = sorted(set(pubs_df['year'].dropna()) & set(pats_df['year'].dropna()))
    
    trl_data = []
    
    for year in years:
        pub_count = len(pubs_df[pubs_df['year'] == year])
        pat_count = len(pats_df[pats_df['year'] == year])
        
        total = pub_count + pat_count
        if total > 0:
            patent_ratio = pat_count / total
            
            # TRL estimation
            if patent_ratio < 0.2:
                trl = 1 + (patent_ratio / 0.2) * 2
            elif patent_ratio < 0.5:
                trl = 3 + ((patent_ratio - 0.2) / 0.3) * 3
            else:
                trl = 6 + ((patent_ratio - 0.5) / 0.5) * 3
            
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
    """Time lag analysis - keep as is"""
    
    st.subheader("⏱️ Publication-Patent Time Lag Analysis")
    st.markdown("Measures the time between scientific publication and patent filing")
    
    if st.session_state.publications_data is None or st.session_state.patents_data is None:
        st.warning("⚠️ Both datasets required")
        return
    
    pubs_df = st.session_state.publications_data
    pats_df = st.session_state.patents_data
    
    if 'year' not in pubs_df.columns or 'year' not in pats_df.columns:
        st.warning("Year data required")
        return
    
    st.markdown("---")
    
    pub_years = pubs_df.groupby('year').size().reset_index(name='pub_count')
    pat_years = pats_df.groupby('year').size().reset_index(name='pat_count')
    
    merged = pd.merge(pub_years, pat_years, on='year', how='outer').fillna(0)
    merged['pub_cumsum'] = merged['pub_count'].cumsum()
    merged['pat_cumsum'] = merged['pat_count'].cumsum()
    
    # Estimate lag
    lags = []
    for idx, row in merged.iterrows():
        if row['pat_count'] > 0 and idx > 0:
            pub_cumsum = row['pub_cumsum']
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
    """Link prediction with unit selection"""
    
    st.subheader("🔗 Link Prediction Analysis")
    st.markdown("""
    **Link Prediction** identifies potential future connections in networks.
    
    **Available Networks:**
    - Entity collaboration networks (authors, inventors, organizations)
    - Keyword/topic co-occurrence networks
    - Hybrid networks
    """)
    
    # Dataset selection
    dataset = st.radio("Select Dataset", ["Publications", "Patents"], horizontal=True)
    
    if dataset == "Publications":
        if st.session_state.publications_data is None:
            st.warning("⚠️ Upload publications data")
            return
        df = st.session_state.publications_data
        dataset_type = 'publications'
    else:
        if st.session_state.patents_data is None:
            st.warning("⚠️ Upload patents data")
            return
        df = st.session_state.patents_data
        dataset_type = 'patents'
    
    # Get available units
    available_units = get_available_units(df, dataset_type)
    
    # Filter to network-applicable units
    network_units = {k: v for k, v in available_units.items() 
                    if k in ['Authors', 'Inventors', 'Organizations', 'Keywords', 'Technology Classes']}
    
    if not network_units:
        st.error("No network-compatible units available")
        return
    
    st.markdown("---")
    
    # Unit selection
    st.markdown("### 🎯 Select Network Type")
    
    selected_unit = st.selectbox(
        "Choose entity/unit for network analysis",
        list(network_units.keys())
    )
    
    col_name = network_units[selected_unit]
    
    st.markdown("---")
    
    # Network construction parameters
    col1, col2 = st.columns(2)
    
    with col1:
        min_cooccurrence = st.slider(
            "Minimum Co-occurrence",
            1, 10, 2,
            help="Minimum times entities must appear together"
        )
    
    with col2:
        top_n_entities = st.slider(
            "Top Entities to Include",
            50, 500, 100,
            help="Focus on most frequent entities"
        )
    
    if st.button("🚀 Build Network & Predict Links", type="primary"):
        with st.spinner(f"Building {selected_unit} network..."):
            # Build network
            G = nx.Graph()
            
            # Parse all entities
            all_entities = []
            entity_docs = []
            
            for entity_str in df[col_name].dropna():
                entities = parse_entity_list(entity_str)
                
                if len(entities) > 0:
                    all_entities.extend(entities)
                    entity_docs.append(entities)
            
            # Get top entities
            entity_freq = Counter(all_entities)
            top_entity_list = [e for e, _ in entity_freq.most_common(top_n_entities)]
            
            # Build co-occurrence edges
            for entities in entity_docs:
                # Filter to top entities
                entities = [e for e in entities if e in top_entity_list]
                
                # Add edges
                for i in range(len(entities)):
                    for j in range(i + 1, len(entities)):
                        if G.has_edge(entities[i], entities[j]):
                            G[entities[i]][entities[j]]['weight'] += 1
                        else:
                            G.add_edge(entities[i], entities[j], weight=1)
            
            # Filter by minimum co-occurrence
            edges_to_remove = [(u, v) for u, v, d in G.edges(data=True)
                              if d['weight'] < min_cooccurrence]
            G.remove_edges_from(edges_to_remove)
            
            # Remove isolated nodes
            G.remove_nodes_from(list(nx.isolates(G)))
        
        st.success(f"✅ Network built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # Network metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Nodes", G.number_of_nodes())
        
        with col2:
            st.metric("Edges", G.number_of_edges())
        
        with col3:
            density = nx.density(G)
            st.metric("Density", f"{density:.4f}")
        
        with col4:
            if len(G) > 0:
                components = nx.number_connected_components(G)
                st.metric("Components", components)
        
        st.markdown("---")
        
        # Most central entities
        st.markdown("### ⭐ Most Central Entities")
        
        if len(G) > 0:
            degree_centrality = nx.degree_centrality(G)
            betweenness = nx.betweenness_centrality(G)
            
            central_df = pd.DataFrame([
                {
                    'Entity': entity,
                    'Degree Centrality': degree_centrality[entity],
                    'Betweenness': betweenness[entity],
                    'Connections': G.degree(entity)
                }
                for entity in sorted(degree_centrality, key=degree_centrality.get, reverse=True)[:20]
            ])
            
            st.dataframe(central_df, use_container_width=True, hide_index=True)
        
        st.markdown("---")
        
        # Link prediction
        st.markdown("### 🔮 Predicted Future Links")
        
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
        pred_df = pd.DataFrame(predictions, columns=['Entity 1', 'Entity 2', 'Link Score'])
        pred_df['Link Score'] = pred_df['Link Score'].round(4)
        
        st.dataframe(pred_df, use_container_width=True, hide_index=True)
        
        # Visualization
        st.markdown("### 📊 Top Predicted Links")
        
        top_10 = pred_df.head(10).copy()
        top_10['Pair'] = top_10['Entity 1'] + ' ↔ ' + top_10['Entity 2']
        
        fig = px.bar(
            top_10,
            x='Link Score',
            y='Pair',
            orientation='h',
            title=f"Top 10 Predicted {selected_unit} Links",
            color='Link Score',
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            yaxis={'categoryorder': 'total ascending'},
            template='plotly_white',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.info(f"""
        **Interpretation for {selected_unit}:**
        - **Higher scores** = More likely future connections
        - These predictions are based on current network structure
        - Consider domain expertise when evaluating predictions
        """)
        
        # Download
        csv = pred_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Predictions",
            csv,
            f"link_predictions_{selected_unit.lower()}_{dataset.lower()}.csv",
            "text/csv"
        )
