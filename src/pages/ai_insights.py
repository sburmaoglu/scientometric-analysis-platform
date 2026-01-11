"""Machine Learning & Clustering Analysis Page"""

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
    st.markdown("Advanced ML algorithms for pattern discovery and classification")
    
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

def prepare_features(df, dataset_type):
    """Prepare feature matrix for ML algorithms"""
    
    features = []
    feature_names = []
    
    # Temporal features
    if 'year' in df.columns:
        years_normalized = (df['year'] - df['year'].min()) / (df['year'].max() - df['year'].min())
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
    
    # Text length features
    if 'abstract' in df.columns:
        text_lengths = df['abstract'].fillna('').apply(len)
        text_lengths_normalized = text_lengths / text_lengths.max() if text_lengths.max() > 0 else text_lengths
        features.append(text_lengths_normalized.values.reshape(-1, 1))
        feature_names.append('abstract_length')
    
    if len(features) == 0:
        return None, None
    
    # Combine features
    X = np.hstack(features)
    
    return X, feature_names

def render_kmeans_clustering():
    """K-Means clustering with elbow method"""
    
    st.subheader("🎯 K-Means Clustering")
    st.markdown("""
    **K-Means** partitions data into K clusters by minimizing within-cluster variance.
    
    **Use Cases:**
    - Group similar publications/patents
    - Identify research themes
    - Discover citation patterns
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
    
    # Prepare features
    X, feature_names = prepare_features(df, dataset_type)
    
    if X is None:
        st.error("Insufficient features for clustering")
        return
    
    st.success(f"✅ Prepared {X.shape[0]} samples with {X.shape[1]} features: {', '.join(feature_names)}")
    
    st.markdown("---")
    
    # Elbow Method
    st.markdown("### 📊 Elbow Method for Optimal K")
    
    if st.button("🔍 Run Elbow Analysis", type="primary"):
        from sklearn.cluster import KMeans
        
        with st.spinner("Computing elbow curve..."):
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Test different K values
            K_range = range(2, min(11, len(X) // 10))
            inertias = []
            silhouette_scores = []
            
            for k in K_range:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                kmeans.fit(X_scaled)
                inertias.append(kmeans.inertia_)
                
                # Silhouette score
                from sklearn.metrics import silhouette_score
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
            
            # Add clusters to dataframe
            df_clustered = df.copy()
            df_clustered['cluster'] = clusters
            
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
            
            cluster_counts = df_clustered['cluster'].value_counts().sort_index()
            
            fig = px.bar(
                x=[f"Cluster {i}" for i in cluster_counts.index],
                y=cluster_counts.values,
                title="Number of Documents per Cluster",
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
                title="K-Means Clusters (PCA Projection)",
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
                cluster_data = df_clustered[df_clustered['cluster'] == cluster_id]
                
                stats = {'Cluster': f'Cluster {cluster_id}', 'Size': len(cluster_data)}
                
                if 'year' in cluster_data.columns:
                    stats['Avg Year'] = f"{cluster_data['year'].mean():.1f}"
                
                if dataset_type == 'publications' and 'citations' in cluster_data.columns:
                    stats['Avg Citations'] = f"{cluster_data['citations'].mean():.1f}"
                elif 'forward_citations' in cluster_data.columns:
                    stats['Avg Citations'] = f"{cluster_data['forward_citations'].mean():.1f}"
                
                cluster_stats.append(stats)
            
            st.dataframe(pd.DataFrame(cluster_stats), use_container_width=True, hide_index=True)
            
            # Save results
            st.session_state.clustering_results = {
                'clusters': clusters,
                'df_clustered': df_clustered,
                'n_clusters': n_clusters
            }
            
            # Download
            csv = df_clustered.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Clustered Data",
                csv,
                f"kmeans_clusters_{dataset.lower()}.csv",
                "text/csv"
            )

def render_hierarchical_clustering():
    """Hierarchical clustering with dendrogram"""
    
    st.subheader("🌳 Hierarchical Clustering")
    st.markdown("""
    **Hierarchical Clustering** builds a tree of clusters (dendrogram).
    
    **Advantages:**
    - No need to specify K in advance
    - Reveals hierarchical structure
    - Dendrogram visualization
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
    
    # Prepare features
    X, feature_names = prepare_features(df, dataset_type)
    
    if X is None:
        st.error("Insufficient features for clustering")
        return
    
    st.success(f"✅ Prepared {X.shape[0]} samples with {X.shape[1]} features")
    
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
            else:
                X_sample = X
                indices = np.arange(len(X))
            
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
            
            ax.set_title(f'Hierarchical Clustering Dendrogram ({linkage_method} linkage)', fontsize=14)
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
                    title=f"Hierarchical Clustering ({n_clusters} clusters)",
                    labels={'x': 'PC1', 'y': 'PC2'}
                )
                
                fig.update_layout(template='plotly_white', height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Cluster sizes
                cluster_counts = pd.Series(clusters).value_counts().sort_index()
                
                st.markdown("#### Cluster Sizes")
                st.bar_chart(cluster_counts)

def render_dbscan_clustering():
    """DBSCAN density-based clustering"""
    
    st.subheader("📊 DBSCAN Clustering")
    st.markdown("""
    **DBSCAN** (Density-Based Spatial Clustering) finds clusters of arbitrary shape.
    
    **Advantages:**
    - No need to specify K
    - Identifies outliers
    - Finds non-spherical clusters
    """)
    
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
    
    X, feature_names = prepare_features(df, dataset_type)
    
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
                title="DBSCAN Clustering Results",
                labels={'x': 'PC1', 'y': 'PC2'}
            )
            
            fig.update_layout(template='plotly_white', height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("**Note:** Cluster -1 represents noise/outlier points")

def render_decision_tree():
    """Decision tree classification"""
    
    st.subheader("🌲 Decision Tree Classification")
    st.markdown("""
    **Decision Trees** create interpretable classification rules.
    
    **Classification Task:** Predict high/low impact based on features
    """)
    
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
    
    X, feature_names = prepare_features(df, dataset_type)
    
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
    """Random forest classification"""
    
    st.subheader("🌳 Random Forest Classification")
    st.markdown("""
    **Random Forest** combines multiple decision trees for robust predictions.
    
    **Advantages:**
    - More accurate than single trees
    - Reduces overfitting
    - Provides feature importance
    """)
    
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
        st.warning(f"{target_col} not available")
        return
    
    X, feature_names = prepare_features(df, dataset_type)
    
    if X is None:
        st.error("Insufficient features")
        return
    
    # Binary target
    threshold = df[target_col].median()
    y = (df[target_col] > threshold).astype(int)
    
    st.success(f"✅ Ready for classification")
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        n_estimators = st.slider("Number of Trees", 10, 200, 100, 10)
    
    with col2:
        max_depth = st.slider("Max Depth", 2, 15, 5)
    
    with col3:
        min_samples_split = st.slider("Min Samples Split", 2, 20, 10)
    
    if st.button("🚀 Train Random Forest", type="primary"):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split, cross_val_score
        from sklearn.metrics import classification_report, roc_curve, auc
        
        with st.spinner("Training random forest..."):
            # Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            
            # Scale
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train
            rf = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42,
                n_jobs=-1
            )
            
            rf.fit(X_train_scaled, y_train)
            
            # Evaluate
            train_score = rf.score(X_train_scaled, y_train)
            test_score = rf.score(X_test_scaled, y_test)
            
            # Cross-validation
            cv_scores = cross_val_score(rf, X_train_scaled, y_train, cv=5)
            
            st.markdown("### 📊 Model Performance")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Train Accuracy", f"{train_score:.2%}")
            
            with col2:
                st.metric("Test Accuracy", f"{test_score:.2%}")
            
            with col3:
                st.metric("CV Score", f"{cv_scores.mean():.2%} ± {cv_scores.std():.2%}")
            
            # ROC Curve
            st.markdown("#### 📈 ROC Curve")
            
            y_pred_proba = rf.predict_proba(X_test_scaled)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=fpr,
                y=tpr,
                mode='lines',
                name=f'ROC (AUC = {roc_auc:.3f})',
                line=dict(color='#3498db', width=3)
            ))
            
            fig.add_trace(go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                name='Random',
                line=dict(color='gray', dash='dash')
            ))
            
            fig.update_layout(
                title="Receiver Operating Characteristic (ROC) Curve",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                template='plotly_white',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance
            st.markdown("#### 🎯 Feature Importance")
            
            importances = rf.feature_importances_
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances,
                'Std': np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)
            }).sort_values('Importance', ascending=False)
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                y=importance_df['Feature'],
                x=importance_df['Importance'],
                orientation='h',
                error_x=dict(type='data', array=importance_df['Std']),
                marker=dict(color=importance_df['Importance'], colorscale='Viridis')
            ))
            
            fig.update_layout(
                title="Feature Importance (with std dev across trees)",
                xaxis_title="Importance",
                yaxis={'categoryorder': 'total ascending'},
                template='plotly_white',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)

def render_pca_analysis():
    """PCA dimensionality reduction"""
    
    st.subheader("📈 Principal Component Analysis (PCA)")
    st.markdown("""
    **PCA** reduces dimensionality while preserving variance.
    
    **Use Cases:**
    - Visualize high-dimensional data
    - Remove noise
    - Feature extraction
    """)
    
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
    
    X, feature_names = prepare_features(df, dataset_type)
    
    if X is None:
        st.error("Insufficient features")
        return
    
    st.success(f"✅ {X.shape[1]} features available")
    
    st.markdown("---")
    
    if st.button("🚀 Run PCA Analysis", type="primary"):
        from sklearn.decomposition import PCA
        
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
                title="PCA Explained Variance",
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
                title="First Two Principal Components",
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
    """Anomaly detection using Isolation Forest"""
    
    st.subheader("🔍 Anomaly Detection")
    st.markdown("""
    **Isolation Forest** identifies unusual/outlier patterns.
    
    **Use Cases:**
    - Detect highly unusual research
    - Identify data quality issues
    - Find breakthrough innovations
    """)
    
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
    
    X, feature_names = prepare_features(df, dataset_type)
    
    if X is None:
        return
    
    st.success(f"✅ Features ready")
    
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
                st.metric("% of Data", f"{n_anomalies/len(df)*100:.1f}%")
            
            # Visualize
            pca = PCA(n_components=2, random_state=42)
            X_pca = pca.fit_transform(X_scaled)
            
            colors = ['red' if p == -1 else 'blue' for p in predictions]
            
            fig = px.scatter(
                x=X_pca[:, 0],
                y=X_pca[:, 1],
                color=predictions.astype(str),
                title="Anomaly Detection Results",
                labels={'x': 'PC1', 'y': 'PC2', 'color': 'Type'},
                color_discrete_map={'-1': 'red', '1': 'blue'}
            )
            
            fig.update_layout(template='plotly_white', height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Show anomalies
            st.markdown("### 🔴 Detected Anomalies")
            
            df_anomalies = df[predictions == -1].copy()
            df_anomalies['anomaly_score'] = scores[predictions == -1]
            df_anomalies = df_anomalies.sort_values('anomaly_score')
            
            if 'title' in df_anomalies.columns:
                st.dataframe(
                    df_anomalies[['title', 'anomaly_score']].head(10),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.write(f"Found {len(df_anomalies)} anomalies")
