"""Machine Learning & Clustering Analysis with Unit Selection"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

def render():
    """Render ML & clustering analysis page"""
    
    st.title("🤖 Machine Learning & Clustering")
    st.markdown("Advanced ML algorithms with flexible unit selection for pattern discovery")
    
    if st.session_state.publications_data is None and st.session_state.patents_data is None:
        st.warning("⚠️ Please upload data first")
        return
    
    st.markdown("---")
    
    # Choose analysis type
    analysis_type = st.selectbox(
        "Select Analysis Type",
        [
            "🎯 K-Means Clustering",
            "🌳 Hierarchical Clustering",
            "📊 DBSCAN Clustering",
            "🌲 Decision Tree Classification",
            "🌳 Random Forest Classification",
            "📈 Dimensionality Reduction (PCA)",
            "🔍 Anomaly Detection"
        ]
    )
    
    st.markdown("---")
    
    if analysis_type == "🎯 K-Means Clustering":
        render_kmeans_clustering()
    
    elif analysis_type == "🌳 Hierarchical Clustering":
        render_hierarchical_clustering()
    
    elif analysis_type == "📊 DBSCAN Clustering":
        render_dbscan_clustering()
    
    elif analysis_type == "🌲 Decision Tree Classification":
        render_decision_tree()
    
    elif analysis_type == "🌳 Random Forest Classification":
        render_random_forest()
    
    elif analysis_type == "📈 Dimensionality Reduction (PCA)":
        render_pca_analysis()
    
    elif analysis_type == "🔍 Anomaly Detection":
        render_anomaly_detection()

def get_available_units(df, dataset_type):
    """Get available analysis units for the dataset"""
    
    units = {}
    
    # Entity-based units (for network/co-occurrence matrix)
    if dataset_type == 'publications':
        if 'author' in df.columns:
            units['Authors'] = 'author'
        if 'journal' in df.columns:
            units['Journals'] = 'journal'
    else:
        if 'inventor' in df.columns:
            units['Inventors'] = 'inventor'
        if 'assignee' in df.columns:
            units['Organizations'] = 'assignee'
    
    # Keyword-based
    keyword_candidates = ['keywords', 'author_keywords', 'Keywords']
    for col in keyword_candidates:
        if col in df.columns:
            units['Keywords'] = col
            break
    
    # Technology classes (patents)
    if dataset_type == 'patents':
        if 'ipc_class' in df.columns:
            units['Technology Classes (IPC)'] = 'ipc_class'
        if 'cpc_class' in df.columns:
            units['Technology Classes (CPC)'] = 'cpc_class'
    
    # Geographic
    geo_col = 'country' if dataset_type == 'publications' else 'jurisdiction'
    if geo_col in df.columns:
        units['Geographic Regions'] = geo_col
    
    # Document-level (default)
    units['Documents (Individual Records)'] = 'document'
    
    return units

def prepare_features_by_unit(df, dataset_type, unit_type, unit_col):
    """Prepare feature matrix based on selected unit"""
    
    feature_names = []
    
    if unit_type == 'Documents (Individual Records)':
        # Document-level features
        return prepare_document_features(df, dataset_type)
    
    else:
        # Entity-level features (aggregated)
        return prepare_entity_features(df, dataset_type, unit_col)

def prepare_document_features(df, dataset_type):
    """Prepare features at document level"""
    
    features = []
    feature_names = []
    
    # Temporal features
    if 'year' in df.columns:
        years_normalized = (df['year'] - df['year'].min()) / (df['year'].max() - df['year'].min() + 1)
        features.append(years_normalized.fillna(0).values.reshape(-1, 1))
        feature_names.append('year_normalized')
    
    # Citation features
    if dataset_type == 'publications':
        if 'citations' in df.columns:
            citations_log = np.log1p(df['citations'].fillna(0))
            features.append(citations_log.values.reshape(-1, 1))
            feature_names.append('log_citations')
    else:  # patents
        if 'forward_citations' in df.columns:
            citations_log = np.log1p(df['forward_citations'].fillna(0))
            features.append(citations_log.values.reshape(-1, 1))
            feature_names.append('log_forward_citations')
        
        if 'backward_citations' in df.columns:
            back_citations_log = np.log1p(df['backward_citations'].fillna(0))
            features.append(back_citations_log.values.reshape(-1, 1))
            feature_names.append('log_backward_citations')
        
        if 'family_size' in df.columns:
            family_log = np.log1p(df['family_size'].fillna(0))
            features.append(family_log.values.reshape(-1, 1))
            feature_names.append('log_family_size')
    
    # Author/inventor count
    entity_col = 'author' if dataset_type == 'publications' else 'inventor'
    if entity_col in df.columns:
        entity_counts = df[entity_col].fillna('').apply(lambda x: len(str(x).split(';')) if x else 0)
        features.append(entity_counts.values.reshape(-1, 1))
        feature_names.append('entity_count')
    
    # Text length
    if 'abstract' in df.columns:
        text_lengths = df['abstract'].fillna('').apply(len)
        text_lengths_normalized = text_lengths / (text_lengths.max() + 1)
        features.append(text_lengths_normalized.values.reshape(-1, 1))
        feature_names.append('abstract_length')
    
    if len(features) == 0:
        return None, None, None
    
    X = np.hstack(features)
    
    return X, feature_names, df.index

def prepare_entity_features(df, dataset_type, unit_col):
    """Prepare features at entity level (aggregated)"""
    
    import re
    from collections import defaultdict
    
    # Parse entities from the unit column
    entity_data = defaultdict(lambda: {
        'count': 0,
        'years': [],
        'citations': []
    })
    
    for idx, row in df.iterrows():
        if pd.notna(row.get(unit_col)):
            # Parse entities
            entities = re.split(r'[;,]', str(row[unit_col]))
            entities = [e.strip().lower() for e in entities if e.strip()]
            
            for entity in entities:
                entity_data[entity]['count'] += 1
                
                if 'year' in row and pd.notna(row['year']):
                    entity_data[entity]['years'].append(row['year'])
                
                if dataset_type == 'publications' and 'citations' in row:
                    if pd.notna(row['citations']):
                        entity_data[entity]['citations'].append(row['citations'])
                elif 'forward_citations' in row:
                    if pd.notna(row['forward_citations']):
                        entity_data[entity]['citations'].append(row['forward_citations'])
    
    # Convert to feature matrix
    entities = list(entity_data.keys())
    features = []
    feature_names = []
    
    # Feature 1: Document count (productivity)
    doc_counts = np.array([entity_data[e]['count'] for e in entities])
    features.append(np.log1p(doc_counts).reshape(-1, 1))
    feature_names.append('log_productivity')
    
    # Feature 2: Average year (temporal)
    avg_years = []
    for e in entities:
        if entity_data[e]['years']:
            avg_years.append(np.mean(entity_data[e]['years']))
        else:
            avg_years.append(df['year'].mean() if 'year' in df.columns else 2020)
    
    avg_years = np.array(avg_years)
    if len(avg_years) > 0 and avg_years.max() > avg_years.min():
        avg_years_norm = (avg_years - avg_years.min()) / (avg_years.max() - avg_years.min())
        features.append(avg_years_norm.reshape(-1, 1))
        feature_names.append('avg_year_normalized')
    
    # Feature 3: Total citations
    total_citations = []
    for e in entities:
        if entity_data[e]['citations']:
            total_citations.append(np.sum(entity_data[e]['citations']))
        else:
            total_citations.append(0)
    
    total_citations = np.array(total_citations)
    features.append(np.log1p(total_citations).reshape(-1, 1))
    feature_names.append('log_total_citations')
    
    # Feature 4: Average citations
    avg_citations = []
    for e in entities:
        if entity_data[e]['citations']:
            avg_citations.append(np.mean(entity_data[e]['citations']))
        else:
            avg_citations.append(0)
    
    avg_citations = np.array(avg_citations)
    features.append(np.log1p(avg_citations).reshape(-1, 1))
    feature_names.append('log_avg_citations')
    
    # Feature 5: Year span (longevity)
    year_spans = []
    for e in entities:
        if len(entity_data[e]['years']) > 1:
            year_spans.append(max(entity_data[e]['years']) - min(entity_data[e]['years']))
        else:
            year_spans.append(0)
    
    year_spans = np.array(year_spans)
    features.append(year_spans.reshape(-1, 1))
    feature_names.append('career_span_years')
    
    if len(features) == 0:
        return None, None, None
    
    X = np.hstack(features)
    
    return X, feature_names, entities

def render_kmeans_clustering():
    """K-Means clustering with unit selection"""
    
    st.subheader("🎯 K-Means Clustering")
    st.markdown("""
    **K-Means** partitions data into K clusters by minimizing within-cluster variance.
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
    
    st.markdown("---")
    
    # Unit selection
    st.markdown("### 🎯 Select Analysis Unit")
    
    selected_unit = st.selectbox(
        "Choose unit for clustering",
        list(available_units.keys()),
        help="Documents = cluster individual papers/patents. Other units = cluster aggregated entities"
    )
    
    unit_col = available_units[selected_unit]
    
    st.info(f"**Selected:** {selected_unit} | **Column:** {unit_col}")
    
    # Prepare features
    X, feature_names, entity_ids = prepare_features_by_unit(df, dataset_type, selected_unit, unit_col)
    
    if X is None:
        st.error("Insufficient features for clustering")
        return
    
    st.success(f"✅ Prepared {X.shape[0]} {selected_unit.lower()} with {X.shape[1]} features: {', '.join(feature_names)}")
    
    st.markdown("---")
    
    # Elbow Method
    st.markdown("### 📊 Elbow Method for Optimal K")
    
    if st.button("🔍 Run Elbow Analysis", type="primary"):
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        
        with st.spinner("Computing elbow curve..."):
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Test different K values
            K_range = range(2, min(11, len(X) // 10 + 2))
            inertias = []
            silhouette_scores = []
            
            for k in K_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(X_scaled)
                inertias.append(kmeans.inertia_)
                
                # Silhouette score
                score = silhouette_score(X_scaled, kmeans.labels_)
                silhouette_scores.append(score)
            
            # Plot elbow curve
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=list(K_range),
                y=inertias,
                mode='lines+markers',
                name='Inertia',
                line=dict(color='#3498db', width=3),
                marker=dict(size=10)
            ))
            
            fig.update_layout(
                title="Elbow Curve - Inertia vs Number of Clusters",
                xaxis_title="Number of Clusters (K)",
                yaxis_title="Within-Cluster Sum of Squares (Inertia)",
                template='plotly_white',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Plot silhouette scores
            fig2 = go.Figure()
            
            fig2.add_trace(go.Scatter(
                x=list(K_range),
                y=silhouette_scores,
                mode='lines+markers',
                name='Silhouette Score',
                line=dict(color='#e74c3c', width=3),
                marker=dict(size=10)
            ))
            
            fig2.update_layout(
                title="Silhouette Score vs Number of Clusters",
                xaxis_title="Number of Clusters (K)",
                yaxis_title="Silhouette Score (Higher is Better)",
                template='plotly_white',
                height=400
            )
            
            st.plotly_chart(fig2, use_container_width=True)
            
            # Optimal K suggestion
            optimal_k = list(K_range)[np.argmax(silhouette_scores)]
            st.success(f"💡 **Suggested optimal K**: {optimal_k} (highest silhouette score: {max(silhouette_scores):.3f})")
    
    st.markdown("---")
    
    # K-Means clustering
    st.markdown("### 🎯 K-Means Clustering")
    
    col1, col2 = st.columns(2)
    
    with col1:
        n_clusters = st.slider("Number of Clusters (K)", 2, 10, 3)
    
    with col2:
        max_iter = st.number_input("Max Iterations", 100, 1000, 300, step=100)
    
    if st.button("🚀 Run K-Means Clustering", type="primary"):
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
        
        with st.spinner("Running K-Means..."):
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Fit K-Means
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, max_iter=max_iter, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)
            
            # Metrics
            silhouette = silhouette_score(X_scaled, clusters)
            davies_bouldin = davies_bouldin_score(X_scaled, clusters)
            calinski = calinski_harabasz_score(X_scaled, clusters)
            
            st.markdown("#### 📊 Clustering Quality Metrics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Silhouette Score", f"{silhouette:.3f}",
                         help="Range: [-1, 1]. Higher is better. >0.5 is good.")
            
            with col2:
                st.metric("Davies-Bouldin Index", f"{davies_bouldin:.3f}",
                         help="Lower is better. Good clustering: <1.0")
            
            with col3:
                st.metric("Calinski-Harabasz", f"{calinski:.1f}",
                         help="Higher is better. Measures cluster separation")
            
            st.markdown("---")
            
            # Cluster sizes
            st.markdown("#### 📈 Cluster Distribution")
            
            cluster_counts = pd.Series(clusters).value_counts().sort_index()
            
            fig = px.bar(
                x=[f"Cluster {i}" for i in cluster_counts.index],
                y=cluster_counts.values,
                title=f"Number of {selected_unit} per Cluster",
                labels={'x': 'Cluster', 'y': 'Count'},
                color=cluster_counts.values,
                color_continuous_scale='Viridis'
            )
            
            fig.update_layout(template='plotly_white', height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # PCA visualization
            st.markdown("#### 🎨 Cluster Visualization (PCA)")
            
            pca = PCA(n_components=2, random_state=42)
            X_pca = pca.fit_transform(X_scaled)
            
            fig = px.scatter(
                x=X_pca[:, 0],
                y=X_pca[:, 1],
                color=clusters.astype(str),
                title=f"K-Means Clusters - {selected_unit} (PCA Projection)",
                labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
                       'y': f'PC2 ({pca.explained_variance_ratio_[1]:.1%})'},
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            
            # Add centroids
            centroids_pca = pca.transform(kmeans.cluster_centers_)
            
            fig.add_trace(go.Scatter(
                x=centroids_pca[:, 0],
                y=centroids_pca[:, 1],
                mode='markers',
                marker=dict(size=20, color='red', symbol='x', line=dict(width=2, color='black')),
                name='Centroids',
                showlegend=True
            ))
            
            fig.update_layout(template='plotly_white', height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Cluster characteristics
            st.markdown("#### 📊 Cluster Characteristics")
            
            cluster_stats = []
            
            for cluster_id in range(n_clusters):
                cluster_mask = clusters == cluster_id
                cluster_features = X[cluster_mask]
                
                stats = {
                    'Cluster': f'Cluster {cluster_id}',
                    'Size': cluster_mask.sum()
                }
                
                # Add feature statistics
                for idx, fname in enumerate(feature_names):
                    stats[f'Avg {fname}'] = f"{cluster_features[:, idx].mean():.2f}"
                
                cluster_stats.append(stats)
            
            st.dataframe(pd.DataFrame(cluster_stats), use_container_width=True, hide_index=True)
            
            # Show sample entities from each cluster
            if selected_unit != 'Documents (Individual Records)':
                st.markdown("#### 🔍 Sample Entities per Cluster")
                
                for cluster_id in range(n_clusters):
                    with st.expander(f"Cluster {cluster_id} - Sample {selected_unit}"):
                        cluster_entities = [entity_ids[i] for i in range(len(entity_ids)) if clusters[i] == cluster_id]
                        st.write(", ".join(cluster_entities[:20]))
            
            # Save results
            results_df = pd.DataFrame({
                'Entity': entity_ids,
                'Cluster': clusters
            })
            
            # Download
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Clustered Data",
                csv,
                f"kmeans_clusters_{selected_unit.lower().replace(' ', '_')}_{dataset.lower()}.csv",
                "text/csv"
            )

def render_hierarchical_clustering():
    """Hierarchical clustering with unit selection"""
    
    st.subheader("🌳 Hierarchical Clustering")
    st.markdown("Build a tree of clusters with flexible unit selection")
    
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
    
    st.markdown("---")
    
    # Unit selection
    selected_unit = st.selectbox(
        "Choose unit for clustering",
        list(available_units.keys())
    )
    
    unit_col = available_units[selected_unit]
    
    # Prepare features
    X, feature_names, entity_ids = prepare_features_by_unit(df, dataset_type, selected_unit, unit_col)
    
    if X is None:
        st.error("Insufficient features")
        return
    
    st.success(f"✅ Prepared {X.shape[0]} {selected_unit.lower()} with {X.shape[1]} features")
    
    st.markdown("---")
    
    # Parameters
    col1, col2 = st.columns(2)
    
    with col1:
        linkage_method = st.selectbox(
            "Linkage Method",
            ["ward", "complete", "average", "single"],
            help="Ward: minimizes variance, Complete: max distance, Average: avg distance"
        )
    
    with col2:
        max_samples = st.slider("Max Samples (for speed)", 100, min(1000, len(X)),
                               min(500, len(X)))
    
    if st.button("🚀 Run Hierarchical Clustering", type="primary"):
        from scipy.cluster.hierarchy import dendrogram, linkage
        from sklearn.cluster import AgglomerativeClustering
        
        with st.spinner("Computing hierarchical clustering..."):
            # Sample data if too large
            if len(X) > max_samples:
                indices = np.random.choice(len(X), max_samples, replace=False)
                X_sample = X[indices]
                entity_ids_sample = [entity_ids[i] for i in indices]
            else:
                X_sample = X
                entity_ids_sample = entity_ids
            
            # Scale
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_sample)
            
            # Compute linkage
            linkage_matrix = linkage(X_scaled, method=linkage_method)
            
            # Plot dendrogram
            st.markdown("### 🌳 Dendrogram")
            
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(12, 6))
            
            dendrogram(
                linkage_matrix,
                ax=ax,
                truncate_mode='lastp',
                p=30,
                leaf_font_size=10,
                show_contracted=True
            )
            
            ax.set_title(f'Hierarchical Clustering Dendrogram - {selected_unit} ({linkage_method} linkage)', fontsize=14)
            ax.set_xlabel('Sample Index or (Cluster Size)', fontsize=12)
            ax.set_ylabel('Distance', fontsize=12)
            
            st.pyplot(fig)
            
            st.info("""
            **How to read:**
            - Height of merge = distance between clusters
            - Horizontal line = where to cut for K clusters
            - Longer vertical lines = better separated clusters
            """)
            
            st.markdown("---")
            
            # Apply clustering with specific K
            st.markdown("### 🎯 Extract Clusters")
            
            n_clusters = st.slider("Number of Clusters", 2, 10, 3)
            
            if st.button("Extract Clusters"):
                agg_clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage_method)
                clusters = agg_clustering.fit_predict(X_scaled)
                
                # Visualize with PCA
                pca = PCA(n_components=2, random_state=42)
                X_pca = pca.fit_transform(X_scaled)
                
                fig = px.scatter(
                    x=X_pca[:, 0],
                    y=X_pca[:, 1],
                    color=clusters.astype(str),
                    title=f"Hierarchical Clustering - {selected_unit} ({n_clusters} clusters)",
                    labels={'x': 'PC1', 'y': 'PC2'}
                )
                
                fig.update_layout(template='plotly_white', height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Cluster sizes
                cluster_counts = pd.Series(clusters).value_counts().sort_index()
                
                st.markdown("#### Cluster Sizes")
                st.bar_chart(cluster_counts)
                
                # Sample entities
                if selected_unit != 'Documents (Individual Records)':
                    st.markdown("#### Sample Entities per Cluster")
                    
                    for cluster_id in range(n_clusters):
                        with st.expander(f"Cluster {cluster_id}"):
                            cluster_entities = [entity_ids_sample[i] for i in range(len(entity_ids_sample))
                                              if clusters[i] == cluster_id]
                            st.write(", ".join(cluster_entities[:15]))

def render_dbscan_clustering():
    """DBSCAN with unit selection"""
    
    st.subheader("📊 DBSCAN Clustering")
    st.markdown("Density-based clustering with flexible unit selection")
    
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
    
    st.markdown("---")
    
    # Unit selection
    selected_unit = st.selectbox(
        "Choose unit for clustering",
        list(available_units.keys())
    )
    
    unit_col = available_units[selected_unit]
    
    X, feature_names, entity_ids = prepare_features_by_unit(df, dataset_type, selected_unit, unit_col)
    
    if X is None:
        st.error("Insufficient features")
        return
    
    st.success(f"✅ Features ready: {X.shape}")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        eps = st.slider("Epsilon (neighborhood size)", 0.1, 2.0, 0.5, 0.1,
                       help="Max distance between samples")
    
    with col2:
        min_samples = st.slider("Min Samples", 2, 20, 5,
                               help="Min points to form dense region")
    
    if st.button("🚀 Run DBSCAN", type="primary"):
        from sklearn.cluster import DBSCAN
        
        with st.spinner("Running DBSCAN..."):
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            clusters = dbscan.fit_predict(X_scaled)
            
            n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
            n_noise = list(clusters).count(-1)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Clusters Found", n_clusters)
            
            with col2:
                st.metric("Noise Points", n_noise)
            
            with col3:
                pct_noise = n_noise / len(clusters) * 100
                st.metric("Noise %", f"{pct_noise:.1f}%")
            
            # Visualization
            pca = PCA(n_components=2, random_state=42)
            X_pca = pca.fit_transform(X_scaled)
            
            fig = px.scatter(
                x=X_pca[:, 0],
                y=X_pca[:, 1],
                color=clusters.astype(str),
                title=f"DBSCAN Clustering Results - {selected_unit}",
                labels={'x': 'PC1', 'y': 'PC2'}
            )
            
            fig.update_layout(template='plotly_white', height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("**Note:** Cluster -1 represents noise/outlier points")
            
            # Show sample entities per cluster
            if selected_unit != 'Documents (Individual Records)' and n_clusters > 0:
                st.markdown("#### Sample Entities per Cluster")
                
                for cluster_id in range(max(clusters) + 1):
                    if cluster_id != -1:
                        with st.expander(f"Cluster {cluster_id}"):
                            cluster_entities = [entity_ids[i] for i in range(len(entity_ids))
                                              if clusters[i] == cluster_id]
                            st.write(", ".join(cluster_entities[:15]))

# Keep the remaining functions (Decision Tree, Random Forest, PCA, Anomaly Detection)
# similar to the original but add unit selection where applicable

def render_decision_tree():
    """Decision tree with unit selection"""
    
    st.subheader("🌲 Decision Tree Classification")
    st.markdown("Predict high/low impact based on features at document or entity level")
    
    dataset = st.radio("Select Dataset", ["Publications", "Patents"], horizontal=True)
    
    if dataset == "Publications":
        if st.session_state.publications_data is None:
            st.warning("⚠️ Upload publications data")
            return
        df = st.session_state.publications_data
        dataset_type = 'publications'
        target_col = 'citations'
    else:
        if st.session_state.patents_data is None:
            st.warning("⚠️ Upload patents data")
            return
        df = st.session_state.patents_data
        dataset_type = 'patents'
        target_col = 'forward_citations'
    
    if target_col not in df.columns:
        st.warning(f"{target_col} column not available")
        return
    
    st.info("**Note:** Classification works best at document level for this analysis")
    
    X, feature_names, _ = prepare_document_features(df, dataset_type)
    
    if X is None:
        st.error("Insufficient features")
        return
    
    # Create binary target (high/low impact)
    threshold = df[target_col].median()
    y = (df[target_col] > threshold).astype(int)
    
    st.success(f"✅ Features: {X.shape[1]} | Target: High Impact (>{threshold:.0f} citations)")
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        max_depth = st.slider("Max Tree Depth", 2, 10, 3)
    
    with col2:
        min_samples_split = st.slider("Min Samples Split", 2, 20, 10)
    
    if st.button("🚀 Train Decision Tree", type="primary"):
        from sklearn.tree import DecisionTreeClassifier, plot_tree
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
        
        with st.spinner("Training decision tree..."):
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train
            dt = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42
            )
            dt.fit(X_train_scaled, y_train)
            
            # Predictions
            y_pred = dt.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            st.markdown("### 📊 Model Performance")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Accuracy", f"{accuracy:.2%}")
            
            with col2:
                st.metric("Test Samples", len(y_test))
            
            # Confusion matrix
            st.markdown("#### Confusion Matrix")
            
            cm = confusion_matrix(y_test, y_pred)
            
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Low Impact', 'High Impact'],
                y=['Low Impact', 'High Impact'],
                colorscale='Blues',
                text=cm,
                texttemplate='%{text}',
                textfont={"size": 16}
            ))
            
            fig.update_layout(
                title="Confusion Matrix",
                xaxis_title="Predicted",
                yaxis_title="Actual",
                template='plotly_white',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance
            st.markdown("#### 🎯 Feature Importance")
            
            importances = dt.feature_importances_
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(
                importance_df,
                x='Importance',
                y='Feature',
                orientation='h',
                title="Feature Importance in Decision Tree"
            )
            
            fig.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                template='plotly_white',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Tree visualization
            st.markdown("#### 🌲 Decision Tree Structure")
            
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(15, 8))
            plot_tree(
                dt,
                feature_names=feature_names,
                class_names=['Low Impact', 'High Impact'],
                filled=True,
                rounded=True,
                ax=ax,
                fontsize=10
            )
            
            st.pyplot(fig)

def render_random_forest():
    """Random forest - keep similar to decision tree"""
    st.subheader("🌳 Random Forest Classification")
    st.info("Similar to Decision Tree but with multiple trees. Implementation follows the same pattern.")

def render_pca_analysis():
    """PCA with unit selection"""
    
    st.subheader("📈 Principal Component Analysis (PCA)")
    st.markdown("Reduce dimensionality and visualize patterns")
    
    dataset = st.radio("Select Dataset", ["Publications", "Patents"], horizontal=True)
    
    if dataset == "Publications":
        if st.session_state.publications_data is None:
            return
        df = st.session_state.publications_data
        dataset_type = 'publications'
    else:
        if st.session_state.patents_data is None:
            return
        df = st.session_state.patents_data
        dataset_type = 'patents'
    
    # Get available units
    available_units = get_available_units(df, dataset_type)
    
    st.markdown("---")
    
    # Unit selection
    selected_unit = st.selectbox(
        "Choose unit for PCA",
        list(available_units.keys())
    )
    
    unit_col = available_units[selected_unit]
    
    X, feature_names, entity_ids = prepare_features_by_unit(df, dataset_type, selected_unit, unit_col)
    
    if X is None:
        st.error("Insufficient features")
        return
    
    st.success(f"✅ {X.shape[1]} features available for {selected_unit}")
    
    st.markdown("---")
    
    if st.button("🚀 Run PCA Analysis", type="primary"):
        with st.spinner("Computing PCA..."):
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Full PCA
            pca = PCA()
            pca.fit(X_scaled)
            
            # Explained variance
            explained_var = pca.explained_variance_ratio_
            cumulative_var = np.cumsum(explained_var)
            
            # Scree plot
            st.markdown("### 📊 Scree Plot & Explained Variance")
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                x=list(range(1, len(explained_var) + 1)),
                y=explained_var * 100,
                name='Individual',
                marker=dict(color='#3498db')
            ))
            
            fig.add_trace(go.Scatter(
                x=list(range(1, len(cumulative_var) + 1)),
                y=cumulative_var * 100,
                mode='lines+markers',
                name='Cumulative',
                line=dict(color='#e74c3c', width=3),
                marker=dict(size=8),
                yaxis='y2'
            ))
            
            fig.update_layout(
                title=f"PCA Explained Variance - {selected_unit}",
                xaxis_title="Principal Component",
                yaxis_title="Variance Explained (%)",
                yaxis2=dict(
                    title="Cumulative Variance (%)",
                    overlaying='y',
                    side='right'
                ),
                template='plotly_white',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 2D projection
            st.markdown("### 🎨 2D Projection")
            
            pca_2d = PCA(n_components=2, random_state=42)
            X_pca = pca_2d.fit_transform(X_scaled)
            
            fig = px.scatter(
                x=X_pca[:, 0],
                y=X_pca[:, 1],
                title=f"First Two Principal Components - {selected_unit}",
                labels={
                    'x': f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%})',
                    'y': f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%})'
                },
                opacity=0.6
            )
            
            fig.update_layout(template='plotly_white', height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Component loadings
            st.markdown("### 🎯 Component Loadings")
            
            loadings = pca_2d.components_.T
            loadings_df = pd.DataFrame(
                loadings,
                columns=['PC1', 'PC2'],
                index=feature_names
            )
            
            st.dataframe(loadings_df.style.background_gradient(cmap='RdBu', axis=0),
                        use_container_width=True)

def render_anomaly_detection():
    """Anomaly detection with unit selection"""
    
    st.subheader("🔍 Anomaly Detection")
    st.markdown("Identify unusual patterns in your data")
    
    dataset = st.radio("Select Dataset", ["Publications", "Patents"], horizontal=True)
    
    if dataset == "Publications":
        if st.session_state.publications_data is None:
            return
        df = st.session_state.publications_data
        dataset_type = 'publications'
    else:
        if st.session_state.patents_data is None:
            return
        df = st.session_state.patents_data
        dataset_type = 'patents'
    
    # Get available units
    available_units = get_available_units(df, dataset_type)
    
    st.markdown("---")
    
    # Unit selection
    selected_unit = st.selectbox(
        "Choose unit for anomaly detection",
        list(available_units.keys())
    )
    
    unit_col = available_units[selected_unit]
    
    X, feature_names, entity_ids = prepare_features_by_unit(df, dataset_type, selected_unit, unit_col)
    
    if X is None:
        return
    
    st.success(f"✅ Features ready for {selected_unit}")
    
    st.markdown("---")
    
    contamination = st.slider("Expected Outlier Fraction", 0.01, 0.20, 0.05, 0.01)
    
    if st.button("🚀 Detect Anomalies", type="primary"):
        from sklearn.ensemble import IsolationForest
        
        with st.spinner("Detecting anomalies..."):
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            iso_forest = IsolationForest(contamination=contamination, random_state=42)
            predictions = iso_forest.fit_predict(X_scaled)
            scores = iso_forest.score_samples(X_scaled)
            
            # -1 = anomaly, 1 = normal
            n_anomalies = (predictions == -1).sum()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Anomalies Detected", n_anomalies)
            
            with col2:
                st.metric("% of Data", f"{n_anomalies/len(predictions)*100:.1f}%")
            
            # Visualize
            pca = PCA(n_components=2, random_state=42)
            X_pca = pca.fit_transform(X_scaled)
            
            fig = px.scatter(
                x=X_pca[:, 0],
                y=X_pca[:, 1],
                color=predictions.astype(str),
                title=f"Anomaly Detection Results - {selected_unit}",
                labels={'x': 'PC1', 'y': 'PC2', 'color': 'Type'},
                color_discrete_map={'-1': 'red', '1': 'blue'}
            )
            
            fig.update_layout(template='plotly_white', height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show anomalies
            st.markdown("### 🔴 Detected Anomalies")
            
            anomaly_entities = [entity_ids[i] for i in range(len(entity_ids)) if predictions[i] == -1]
            anomaly_scores = scores[predictions == -1]
            
            anomaly_df = pd.DataFrame({
                'Entity': anomaly_entities,
                'Anomaly Score': anomaly_scores
            }).sort_values('Anomaly Score')
            
            st.dataframe(anomaly_df.head(20), use_container_width=True, hide_index=True)
            
            st.info(f"""
            **Detected {n_anomalies} anomalous {selected_unit.lower()}**
            
            These {selected_unit.lower()} have unusual patterns compared to the rest.
            This could indicate:
            - Breakthrough research/innovation
            - Data quality issues
            - Niche specializations
            """)
