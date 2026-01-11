"""Technology Roadmapping - Interactive Pipeline Report Generator"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import re
from collections import Counter
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

def render():
    """Render technology roadmapping page"""
    
    st.title("🗺️ Technology Roadmapping")
    st.markdown("Build comprehensive technology roadmaps with guided analysis pipeline")
    
    # Initialize session state for roadmap
    if 'roadmap_step' not in st.session_state:
        st.session_state.roadmap_step = 1
    
    if 'roadmap_pipeline' not in st.session_state:
        st.session_state.roadmap_pipeline = []
    
    if 'roadmap_results' not in st.session_state:
        st.session_state.roadmap_results = {}
    
    if 'roadmap_config' not in st.session_state:
        st.session_state.roadmap_config = {
            'dataset': None,
            'unit': None,
            'title': 'Technology Roadmap Report',
            'time_horizon': 'historical',
            'style': 'academic'
        }
    
    # Progress indicator
    show_progress_bar()
    
    st.markdown("---")
    
    # Render current step
    current_step = st.session_state.roadmap_step
    
    if current_step == 1:
        render_data_selection()
    elif current_step == 2:
        render_unit_selection()
    elif current_step == 3:
        render_pipeline_builder()
    elif current_step == 4:
        render_configuration()
    elif current_step == 5:
        render_report_generator()

def show_progress_bar():
    """Display progress indicator"""
    
    current_step = st.session_state.roadmap_step
    
    st.markdown("### 📊 Progress")
    
    # Progress bar
    progress = current_step / 5
    st.progress(progress)
    
    # Step indicators
    cols = st.columns(5)
    
    steps = [
        ("1️⃣", "Data"),
        ("2️⃣", "Unit"),
        ("3️⃣", "Pipeline"),
        ("4️⃣", "Config"),
        ("5️⃣", "Generate")
    ]
    
    for i, (emoji, label) in enumerate(steps, 1):
        with cols[i-1]:
            if i < current_step:
                st.success(f"{emoji} {label} ✓")
            elif i == current_step:
                st.info(f"{emoji} **{label}**")
            else:
                st.write(f"{emoji} {label}")

def render_data_selection():
    """Step 1: Data selection"""
    
    st.subheader("1️⃣ Data Selection")
    st.markdown("Select which dataset(s) to use for the technology roadmap")
    
    # Check available data
    has_pubs = st.session_state.publications_data is not None
    has_pats = st.session_state.patents_data is not None
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Publications", "✅ Available" if has_pubs else "❌ Not loaded")
    
    with col2:
        st.metric("Patents", "✅ Available" if has_pats else "❌ Not loaded")
    
    with col3:
        st.metric("Status", "Ready" if (has_pubs or has_pats) else "Upload Data")
    
    st.markdown("---")
    
    if not has_pubs and not has_pats:
        st.warning("⚠️ Please upload data in the Data Upload page first")
        return
    
    # Dataset selection
    st.markdown("### 📊 Choose Dataset")
    
    options = []
    if has_pubs:
        options.append("Publications")
    if has_pats:
        options.append("Patents")
    if has_pubs and has_pats:
        options.append("Both (Comparative)")
    
    dataset_choice = st.radio(
        "Select dataset for roadmap",
        options,
        index=0 if st.session_state.roadmap_config['dataset'] is None else 
              options.index(st.session_state.roadmap_config['dataset'])
    )
    
    # Preview data
    if dataset_choice == "Publications":
        df = st.session_state.publications_data
        show_dataset_preview(df, 'publications')
    
    elif dataset_choice == "Patents":
        df = st.session_state.patents_data
        show_dataset_preview(df, 'patents')
    
    else:  # Both
        pubs_df = st.session_state.publications_data
        pats_df = st.session_state.patents_data
        st.success(f"✅ Selected: {len(pubs_df):,} publications + {len(pats_df):,} patents")
    
    st.markdown("---")
    
    # Navigation buttons
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col3:
        if st.button("Next: Unit Selection →", type="primary", use_container_width=True):
            st.session_state.roadmap_config['dataset'] = dataset_choice
            st.session_state.roadmap_step = 2
            st.rerun()

def show_dataset_preview(df, dataset_type):
    """Show dataset preview with key metrics"""
    
    st.success(f"✅ Selected: {len(df):,} {dataset_type}")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    
    with col2:
        if 'year' in df.columns:
            years = df['year'].dropna()
            st.metric("Year Range", f"{int(years.min())}-{int(years.max())}")
    
    with col3:
        citation_col = 'citations' if dataset_type == 'publications' else 'forward_citations'
        if citation_col in df.columns:
            total_cites = df[citation_col].sum()
            st.metric("Total Citations", f"{int(total_cites):,}")
    
    with col4:
        entity_col = 'author' if dataset_type == 'publications' else 'assignee'
        if entity_col in df.columns:
            all_entities = []
            for e_str in df[entity_col].dropna()[:1000]:  # Sample for speed
                all_entities.extend(re.split(r'[;,]', str(e_str)))
            st.metric("Unique Entities", f"{len(set(all_entities)):,}+")

def render_unit_selection():
    """Step 2: Unit selection"""
    
    st.subheader("2️⃣ Unit Selection")
    st.markdown("Select the primary unit of analysis for your technology roadmap")
    
    # Check if data is selected
    if st.session_state.roadmap_config['dataset'] is None:
        st.warning("⚠️ Please complete Step 1: Data Selection first")
        
        if st.button("← Back to Data Selection"):
            st.session_state.roadmap_step = 1
            st.rerun()
        return
    
    dataset_choice = st.session_state.roadmap_config['dataset']
    
    st.info(f"**Selected Dataset:** {dataset_choice}")
    
    st.markdown("---")
    
    # Get available units
    available_units = get_available_units_for_roadmap(dataset_choice)
    
    if not available_units:
        st.error("No analysis units available for this dataset")
        return
    
    # Unit selection with descriptions
    st.markdown("### 🎯 Available Analysis Units")
    
    unit_descriptions = {
        'Authors': '👥 Track research evolution through author/inventor networks',
        'Inventors': '👥 Track innovation through inventor networks',
        'Organizations': '🏢 Monitor institutional technology development',
        'Countries': '🌍 Analyze geographic technology leadership',
        'Jurisdictions': '🌍 Analyze patent jurisdiction patterns',
        'Keywords': '🏷️ Follow topic evolution and emergence',
        'IPC Classes': '🔬 Map patent technology classification (IPC)',
        'CPC Classes': '🔬 Map patent technology classification (CPC)',
        'Technology Areas': '📊 Broad technology domain analysis',
        'Journals': '📚 Track publication venue trends',
        'Documents': '📄 Document-level granular analysis'
    }
    
    selected_unit = st.radio(
        "Choose your primary unit of analysis",
        list(available_units.keys()),
        format_func=lambda x: f"{unit_descriptions.get(x, x)}",
        index=0 if st.session_state.roadmap_config['unit'] is None else
              list(available_units.keys()).index(st.session_state.roadmap_config['unit'])
              if st.session_state.roadmap_config['unit'] in available_units else 0
    )
    
    st.markdown("---")
    
    # Unit-specific configuration
    st.markdown("### ⚙️ Unit Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if selected_unit in ['IPC Classes', 'CPC Classes', 'Keywords']:
            focus_top_n = st.slider("Focus on top N entities", 5, 50, 10,
                                   help="Limit analysis to most frequent entities")
        else:
            focus_top_n = st.slider("Focus on top N entities", 5, 100, 20)
    
    with col2:
        time_granularity = st.selectbox(
            "Time Granularity",
            ["Yearly", "3-Year Periods", "5-Year Periods"],
            help="How to aggregate temporal data"
        )
    
    # Preview unit data
    st.markdown("### 📊 Unit Preview")
    
    preview_data = get_unit_preview(dataset_choice, selected_unit, focus_top_n)
    
    if preview_data is not None:
        st.dataframe(preview_data.head(10), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Navigation buttons
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("← Back", use_container_width=True):
            st.session_state.roadmap_step = 1
            st.rerun()
    
    with col3:
        if st.button("Next: Pipeline →", type="primary", use_container_width=True):
            st.session_state.roadmap_config['unit'] = selected_unit
            st.session_state.roadmap_config['unit_column'] = available_units[selected_unit]
            st.session_state.roadmap_config['focus_top_n'] = focus_top_n
            st.session_state.roadmap_config['time_granularity'] = time_granularity
            st.session_state.roadmap_step = 3
            st.rerun()

def render_pipeline_builder():
    """Step 3: Build analysis pipeline"""
    
    st.subheader("3️⃣ Analysis Pipeline Builder")
    st.markdown("Select and configure analyses to include in your roadmap")
    
    # Check prerequisites
    if st.session_state.roadmap_config['unit'] is None:
        st.warning("⚠️ Please complete Step 2: Unit Selection first")
        
        if st.button("← Back to Unit Selection"):
            st.session_state.roadmap_step = 2
            st.rerun()
        return
    
    config = st.session_state.roadmap_config
    st.info(f"**Dataset:** {config['dataset']} | **Unit:** {config['unit']}")
    
    st.markdown("---")
    
    # Available analysis modules
    st.markdown("### 📦 Select Analysis Modules")
    
    analysis_modules = {
        'temporal_trends': {
            'name': '📈 Temporal Trends',
            'description': 'Growth patterns, trends, and inflection points',
            'required': True
        },
        'diversity_entropy': {
            'name': '📊 Diversity Analysis',
            'description': 'Shannon entropy and concentration metrics',
            'required': False
        },
        'impact_analysis': {
            'name': '💥 Impact Assessment',
            'description': 'Citation analysis and research impact',
            'required': False
        },
        'clustering': {
            'name': '🎯 Clustering Analysis',
            'description': 'Group similar entities/technologies',
            'required': False
        },
        'geographic_evolution': {
            'name': '🌍 Geographic Evolution',
            'description': 'Spatial development and diffusion',
            'required': False
        },
        'emerging_topics': {
            'name': '🌟 Emerging Topics',
            'description': 'Identify growing research areas',
            'required': False
        }
    }
    
    # Display modules with checkboxes
    selected_analyses = ['temporal_trends']  # Always include
    
    for module_id, module_info in analysis_modules.items():
        if module_info['required']:
            st.checkbox(
                f"{module_info['name']} - {module_info['description']}",
                value=True,
                disabled=True,
                key=f"check_{module_id}"
            )
        else:
            is_selected = st.checkbox(
                f"{module_info['name']} - {module_info['description']}",
                value=module_id in st.session_state.roadmap_pipeline,
                key=f"check_{module_id}"
            )
            
            if is_selected:
                selected_analyses.append(module_id)
    
    st.markdown("---")
    
    # Pipeline summary
    st.markdown("### 📋 Pipeline Summary")
    
    total_selected = len(selected_analyses)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Analyses", total_selected)
    
    with col2:
        estimated_time = total_selected * 2  # 2 seconds per analysis
        st.metric("Est. Time", f"~{estimated_time}s")
    
    with col3:
        st.metric("Report Sections", total_selected + 2)  # +2 for intro and summary
    
    st.markdown("---")
    
    # Navigation buttons
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("← Back", use_container_width=True):
            st.session_state.roadmap_step = 2
            st.rerun()
    
    with col3:
        if st.button("Next: Configure →", type="primary", use_container_width=True):
            st.session_state.roadmap_pipeline = selected_analyses
            st.session_state.roadmap_step = 4
            st.rerun()

def render_configuration():
    """Step 4: Report configuration"""
    
    st.subheader("4️⃣ Report Configuration")
    st.markdown("Configure report settings and style")
    
    # Check prerequisites
    if not st.session_state.roadmap_pipeline:
        st.warning("⚠️ Please complete Step 3: Analysis Pipeline first")
        
        if st.button("← Back to Pipeline"):
            st.session_state.roadmap_step = 3
            st.rerun()
        return
    
    st.info(f"**Pipeline:** {len(st.session_state.roadmap_pipeline)} analyses configured")
    
    st.markdown("---")
    
    # Report metadata
    st.markdown("### 📄 Report Metadata")
    
    col1, col2 = st.columns(2)
    
    with col1:
        report_title = st.text_input(
            "Report Title",
            value=st.session_state.roadmap_config.get('title', 'Technology Roadmap Report')
        )
        
        report_subtitle = st.text_input(
            "Subtitle (optional)",
            value=st.session_state.roadmap_config.get('subtitle', '')
        )
    
    with col2:
        author = st.text_input(
            "Author/Organization",
            value=st.session_state.roadmap_config.get('author', '')
        )
        
        report_date = st.date_input(
            "Report Date",
            value=datetime.now()
        )
    
    st.markdown("---")
    
    # Report style
    st.markdown("### 🎨 Report Style")
    
    col1, col2 = st.columns(2)
    
    with col1:
        report_style = st.selectbox(
            "Report Style",
            ["Academic", "Executive", "Technical"],
            help="Academic: Detailed. Executive: High-level. Technical: Implementation focus"
        )
    
    with col2:
        include_recommendations = st.checkbox("Include Strategic Recommendations", value=True)
    
    st.markdown("---")
    
    # Navigation buttons
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("← Back", use_container_width=True):
            st.session_state.roadmap_step = 3
            st.rerun()
    
    with col3:
        if st.button("Generate Report →", type="primary", use_container_width=True):
            st.session_state.roadmap_config.update({
                'title': report_title,
                'subtitle': report_subtitle,
                'author': author,
                'report_date': report_date.strftime('%Y-%m-%d'),
                'style': report_style.lower(),
                'include_recommendations': include_recommendations
            })
            st.session_state.roadmap_step = 5
            st.rerun()

def render_report_generator():
    """Step 5: Generate and display report"""
    
    st.subheader("5️⃣ Technology Roadmap Report")
    
    # Check if report already generated
    if not st.session_state.roadmap_results:
        # Generate report
        generate_roadmap_report()
    
    # Display report
    display_roadmap_report()
    
    st.markdown("---")
    
    # Action buttons
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔄 Regenerate", use_container_width=True):
            st.session_state.roadmap_results = {}
            st.rerun()
    
    with col2:
        if st.button("📄 Export PDF", use_container_width=True):
            st.info("PDF export functionality coming soon")
    
    with col3:
        if st.button("📊 Export Data", use_container_width=True):
            export_roadmap_data()
    
    with col4:
        if st.button("🆕 New Roadmap", use_container_width=True):
            # Reset everything
            st.session_state.roadmap_step = 1
            st.session_state.roadmap_pipeline = []
            st.session_state.roadmap_results = {}
            st.session_state.roadmap_config = {
                'dataset': None,
                'unit': None,
                'title': 'Technology Roadmap Report',
                'time_horizon': 'historical',
                'style': 'academic'
            }
            st.rerun()

def generate_roadmap_report():
    """Execute pipeline and generate report"""
    
    st.markdown("### 🔄 Generating Report...")
    
    config = st.session_state.roadmap_config
    pipeline = st.session_state.roadmap_pipeline
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = {}
    
    # Get data
    dataset_choice = config['dataset']
    
    if dataset_choice == "Publications":
        df = st.session_state.publications_data
        dataset_type = 'publications'
    elif dataset_choice == "Patents":
        df = st.session_state.patents_data
        dataset_type = 'patents'
    else:
        df = None
        dataset_type = 'both'
    
    # Execute each analysis
    for i, module_id in enumerate(pipeline):
        progress = (i + 1) / len(pipeline)
        progress_bar.progress(progress)
        status_text.text(f"Running: {module_id.replace('_', ' ').title()}... ({i+1}/{len(pipeline)})")
        
        try:
            if module_id == 'temporal_trends':
                results[module_id] = run_temporal_trends(df, config, dataset_type)
            
            elif module_id == 'diversity_entropy':
                results[module_id] = run_diversity_analysis(df, config, dataset_type)
            
            elif module_id == 'impact_analysis':
                results[module_id] = run_impact_analysis(df, config, dataset_type)
            
            elif module_id == 'clustering':
                results[module_id] = run_clustering_analysis(df, config, dataset_type)
            
            elif module_id == 'geographic_evolution':
                results[module_id] = run_geographic_evolution(df, config, dataset_type)
            
            elif module_id == 'emerging_topics':
                results[module_id] = run_emerging_topics(df, config, dataset_type)
            
            results[module_id]['status'] = 'success'
        
        except Exception as e:
            results[module_id] = {
                'status': 'error',
                'error': str(e)
            }
    
    progress_bar.progress(1.0)
    status_text.text("✅ Report generation complete!")
    
    # Save results
    st.session_state.roadmap_results = results
    
    st.success("🎉 Technology roadmap generated successfully!")

def run_temporal_trends(df, config, dataset_type):
    """Execute temporal trends analysis"""
    
    if 'year' not in df.columns:
        return {'status': 'error', 'error': 'Year column not available'}
    
    yearly = df.groupby('year').size().reset_index(name='count')
    
    # Calculate growth metrics
    if len(yearly) > 1:
        growth_rate = ((yearly['count'].iloc[-1] - yearly['count'].iloc[-2]) / 
                      yearly['count'].iloc[-2] * 100)
        total_growth = ((yearly['count'].iloc[-1] - yearly['count'].iloc[0]) / 
                       yearly['count'].iloc[0] * 100)
        avg_growth = yearly['count'].pct_change().mean() * 100
    else:
        growth_rate = 0
        total_growth = 0
        avg_growth = 0
    
    # Create visualization
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=yearly['year'],
        y=yearly['count'],
        mode='lines+markers',
        fill='tozeroy',
        line=dict(width=3, color='#3498db'),
        marker=dict(size=8),
        name='Count'
    ))
    
    fig.update_layout(
        title=f"Temporal Trends - {config['dataset']}",
        xaxis_title="Year",
        yaxis_title="Count",
        template='plotly_white',
        height=400,
        hovermode='x unified'
    )
    
    return {
        'yearly_data': yearly,
        'growth_rate': growth_rate,
        'total_growth': total_growth,
        'avg_growth': avg_growth,
        'peak_year': int(yearly.loc[yearly['count'].idxmax(), 'year']),
        'peak_count': int(yearly['count'].max()),
        'figure': fig
    }

def run_diversity_analysis(df, config, dataset_type):
    """Execute diversity analysis"""
    
    unit_col = config.get('unit_column')
    
    if unit_col == 'document':
        return {'status': 'skip', 'message': 'Diversity not applicable to documents'}
    
    if unit_col not in df.columns:
        return {'status': 'error', 'error': f'Column {unit_col} not available'}
    
    # Parse entities
    all_entities = []
    for entity_str in df[unit_col].dropna():
        entities = re.split(r'[;,]', str(entity_str))
        all_entities.extend([e.strip().lower() for e in entities if e.strip()])
    
    # Calculate entropy
    entity_counts = pd.Series(all_entities).value_counts(normalize=True)
    shannon_entropy = entropy(entity_counts, base=2)
    
    # Diversity score (normalized)
    max_entropy = np.log2(len(entity_counts))
    diversity_score = shannon_entropy / max_entropy if max_entropy > 0 else 0
    
    # Top entities
    top_entities = pd.Series(all_entities).value_counts().head(10)
    
    # Visualization
    fig = px.bar(
        x=top_entities.index,
        y=top_entities.values,
        title=f"Top 10 {config['unit']}",
        labels={'x': config['unit'], 'y': 'Count'}
    )
    
    fig.update_layout(template='plotly_white', height=400)
    
    return {
        'entropy': shannon_entropy,
        'diversity_score': diversity_score,
        'unique_entities': len(entity_counts),
        'top_entities': top_entities.to_dict(),
        'figure': fig
    }

def run_impact_analysis(df, config, dataset_type):
    """Execute impact analysis"""
    
    citation_col = 'citations' if dataset_type == 'publications' else 'forward_citations'
    
    if citation_col not in df.columns:
        return {'status': 'error', 'error': 'Citation data not available'}
    
    citations = df[citation_col].dropna()
    
    # Calculate metrics
    mean_citations = citations.mean()
    median_citations = citations.median()
    total_citations = citations.sum()
    
    # H-index calculation
    sorted_citations = sorted(citations, reverse=True)
    h_index = 0
    for i, cites in enumerate(sorted_citations, 1):
        if cites >= i:
            h_index = i
        else:
            break
    
    # Citation distribution
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=citations[citations <= citations.quantile(0.95)],  # Remove outliers for viz
        nbinsx=30,
        marker_color='#e74c3c',
        name='Citations'
    ))
    
    fig.update_layout(
        title="Citation Distribution",
        xaxis_title="Citations",
        yaxis_title="Frequency",
        template='plotly_white',
        height=400
    )
    
    return {
        'mean_citations': mean_citations,
        'median_citations': median_citations,
        'total_citations': int(total_citations),
        'h_index': h_index,
        'highly_cited': int((citations >= citations.quantile(0.9)).sum()),
        'figure': fig
    }

def run_clustering_analysis(df, config, dataset_type):
    """Execute clustering analysis"""
    
    # Prepare simple features
    features = []
    feature_names = []
    
    if 'year' in df.columns:
        years = df['year'].fillna(df['year'].mean())
        years_norm = (years - years.min()) / (years.max() - years.min() + 1)
        features.append(years_norm)
        feature_names.append('year_norm')
    
    citation_col = 'citations' if dataset_type == 'publications' else 'forward_citations'
    if citation_col in df.columns:
        citations_log = np.log1p(df[citation_col].fillna(0))
        features.append(citations_log)
        feature_names.append('log_citations')
    
    if len(features) < 2:
        return {'status': 'error', 'error': 'Insufficient features for clustering'}
    
    X = np.column_stack(features)
    
    # Sample if too large
    if len(X) > 1000:
        indices = np.random.choice(len(X), 1000, replace=False)
        X_sample = X[indices]
    else:
        X_sample = X
    
    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sample)
    
    # K-means
    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    # PCA for visualization
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    
    # Visualization
    fig = px.scatter(
        x=X_pca[:, 0],
        y=X_pca[:, 1],
        color=clusters.astype(str),
        title="K-Means Clustering (PCA Projection)",
        labels={'x': f'PC1 ({pca.explained_variance_ratio_[0]:.1%})',
               'y': f'PC2 ({pca.explained_variance_ratio_[1]:.1%})'},
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    
    fig.update_layout(template='plotly_white', height=400)
    
    # Cluster sizes
    cluster_sizes = pd.Series(clusters).value_counts().sort_index()
    
    return {
        'n_clusters': n_clusters,
        'cluster_sizes': cluster_sizes.to_dict(),
        'figure': fig
    }

def run_geographic_evolution(df, config, dataset_type):
    """Execute geographic evolution analysis"""
    
    geo_col = 'country' if dataset_type == 'publications' else 'jurisdiction'
    
    if geo_col not in df.columns:
        return {'status': 'error', 'error': 'Geographic data not available'}
    
    geo_counts = df[geo_col].value_counts().head(10)
    
    # Visualization
    fig = px.pie(
        values=geo_counts.values,
        names=geo_counts.index,
        title=f"Top 10 {geo_col.title()}s",
        hole=0.4
    )
    
    fig.update_layout(template='plotly_white', height=400)
    
    return {
        'top_countries': geo_counts.to_dict(),
        'unique_countries': df[geo_col].nunique(),
        'figure': fig
    }

def run_emerging_topics(df, config, dataset_type):
    """Execute emerging topics detection"""
    
    if 'year' not in df.columns:
        return {'status': 'error', 'error': 'Year data required'}
    
    # Get recent vs older data
    recent_year = df['year'].max()
    cutoff_year = recent_year - 3
    
    recent_df = df[df['year'] >= cutoff_year]
    older_df = df[df['year'] < cutoff_year]
    
    # For keywords/IPC analysis
    keyword_col = config.get('unit_column')
    
    if keyword_col == 'document' or keyword_col not in df.columns:
        return {'status': 'skip', 'message': 'Emerging topics requires keyword/classification data'}
    
    # Parse entities
    def get_entities(dataframe):
        entities = []
        for entity_str in dataframe[keyword_col].dropna():
            ents = re.split(r'[;,]', str(entity_str))
            entities.extend([e.strip().lower() for e in ents if e.strip()])
        return entities
    
    recent_entities = get_entities(recent_df)
    older_entities = get_entities(older_df)
    
    recent_counts = Counter(recent_entities)
    older_counts = Counter(older_entities)
    
    # Calculate growth rates
    emerging = []
    for entity, recent_count in recent_counts.most_common(20):
        older_count = older_counts.get(entity, 0)
        if older_count > 0:
            growth = ((recent_count - older_count) / older_count) * 100
        else:
            growth = 1000  # New topic
        
        emerging.append({
            'entity': entity,
            'recent_count': recent_count,
            'older_count': older_count,
            'growth_rate': growth
        })
    
    # Sort by growth
    emerging = sorted(emerging, key=lambda x: x['growth_rate'], reverse=True)[:10]
    
    # Visualization
    emerging_df = pd.DataFrame(emerging)
    
    fig = px.bar(
        emerging_df,
        x='growth_rate',
        y='entity',
        orientation='h',
        title="Top 10 Emerging Topics (by Growth Rate)",
        labels={'growth_rate': 'Growth Rate (%)', 'entity': config['unit']}
    )
    
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        template='plotly_white',
        height=400
    )
    
    return {
        'emerging_topics': emerging,
        'figure': fig
    }

def display_roadmap_report():
    """Display generated roadmap report"""
    
    results = st.session_state.roadmap_results
    config = st.session_state.roadmap_config
    
    # Report header
    st.markdown(f"# {config.get('title', 'Technology Roadmap Report')}")
    
    if config.get('subtitle'):
        st.markdown(f"*{config.get('subtitle')}*")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Date:** {config.get('report_date', datetime.now().strftime('%Y-%m-%d'))}")
    with col2:
        if config.get('author'):
            st.markdown(f"**Author:** {config.get('author')}")
    
    st.markdown("---")
    
    # Executive summary
    st.markdown("## 📋 Executive Summary")
    
    st.markdown(f"""
    This technology roadmap analyzes **{config.get('dataset')}** data 
    using **{config.get('unit')}** as the primary unit of analysis.
    
    The analysis includes **{len(results)} analytical perspectives**, providing comprehensive 
    insights into technology evolution, patterns, and future directions.
    """)
    
    st.markdown("---")
    
    # Display each analysis result
    for i, (module_id, result) in enumerate(results.items(), 1):
        if result.get('status') == 'success':
            st.markdown(f"## {i}. {module_id.replace('_', ' ').title()}")
            
            # Display module-specific results
            if module_id == 'temporal_trends':
                display_temporal_results(result)
            
            elif module_id == 'diversity_entropy':
                display_diversity_results(result)
            
            elif module_id == 'impact_analysis':
                display_impact_results(result)
            
            elif module_id == 'clustering':
                display_clustering_results(result)
            
            elif module_id == 'geographic_evolution':
                display_geographic_results(result)
            
            elif module_id == 'emerging_topics':
                display_emerging_results(result)
            
            st.markdown("---")
        
        elif result.get('status') == 'skip':
            st.info(f"ℹ️ {module_id.replace('_', ' ').title()}: {result.get('message')}")
        
        elif result.get('status') == 'error':
            st.error(f"❌ {module_id.replace('_', ' ').title()}: {result.get('error')}")
    
    # Recommendations
    if config.get('include_recommendations'):
        st.markdown("## 🎯 Strategic Recommendations")
        generate_recommendations(results, config)

def display_temporal_results(result):
    """Display temporal trends results"""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Recent Growth", f"{result['growth_rate']:.1f}%")
    
    with col2:
        st.metric("Total Growth", f"{result['total_growth']:.1f}%")
    
    with col3:
        st.metric("Peak Year", f"{result['peak_year']}")
    
    with col4:
        st.metric("Peak Count", f"{result['peak_count']:,}")
    
    st.plotly_chart(result['figure'], use_container_width=True)
    
    # Interpretation
    if result['growth_rate'] > 20:
        st.success("📈 **Strong growth trend** - Technology is rapidly expanding")
    elif result['growth_rate'] > 5:
        st.info("📊 **Moderate growth** - Steady development")
    else:
        st.warning("📉 **Slow/declining growth** - May indicate maturity or declining interest")

def display_diversity_results(result):
    """Display diversity analysis results"""
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Shannon Entropy", f"{result['entropy']:.3f}")
    
    with col2:
        st.metric("Diversity Score", f"{result['diversity_score']:.2f}")
    
    with col3:
        st.metric("Unique Entities", f"{result['unique_entities']:,}")
    
    st.plotly_chart(result['figure'], use_container_width=True)
    
    # Interpretation
    if result['diversity_score'] > 0.7:
        st.success("🌈 **High diversity** - Field is highly distributed across entities")
    elif result['diversity_score'] > 0.4:
        st.info("📊 **Moderate diversity** - Some concentration but fairly distributed")
    else:
        st.warning("🎯 **High concentration** - Field dominated by few entities")

def display_impact_results(result):
    """Display impact analysis results"""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Mean Citations", f"{result['mean_citations']:.1f}")
    
    with col2:
        st.metric("Median Citations", f"{result['median_citations']:.0f}")
    
    with col3:
        st.metric("H-Index", f"{result['h_index']}")
    
    with col4:
        st.metric("Highly Cited (Top 10%)", f"{result['highly_cited']}")
    
    st.plotly_chart(result['figure'], use_container_width=True)

def display_clustering_results(result):
    """Display clustering results"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Number of Clusters", result['n_clusters'])
    
    with col2:
        cluster_sizes = result['cluster_sizes']
        avg_size = np.mean(list(cluster_sizes.values()))
        st.metric("Avg Cluster Size", f"{avg_size:.0f}")
    
    st.plotly_chart(result['figure'], use_container_width=True)
    
    # Show cluster sizes
    st.markdown("**Cluster Sizes:**")
    for cluster_id, size in result['cluster_sizes'].items():
        st.write(f"- Cluster {cluster_id}: {size} items")

def display_geographic_results(result):
    """Display geographic results"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Unique Countries", result['unique_countries'])
    
    with col2:
        top_country = list(result['top_countries'].keys())[0]
        st.metric("Leading Country", top_country)
    
    st.plotly_chart(result['figure'], use_container_width=True)

def display_emerging_results(result):
    """Display emerging topics results"""
    
    st.plotly_chart(result['figure'], use_container_width=True)
    
    st.markdown("**Top Emerging Topics:**")
    
    for i, topic in enumerate(result['emerging_topics'][:5], 1):
        st.write(f"{i}. **{topic['entity']}** - Growth: {topic['growth_rate']:.0f}% "
                f"(Recent: {topic['recent_count']}, Older: {topic['older_count']})")

def generate_recommendations(results, config):
    """Generate strategic recommendations based on results"""
    
    recommendations = []
    
    # Based on temporal trends
    if 'temporal_trends' in results:
        growth = results['temporal_trends'].get('growth_rate', 0)
        if growth > 20:
            recommendations.append("✅ **High growth area** - Consider increased investment and resource allocation")
        elif growth < 0:
            recommendations.append("⚠️ **Declining trend** - Reassess strategic priorities or pivot focus")
    
    # Based on diversity
    if 'diversity_entropy' in results:
        diversity = results['diversity_entropy'].get('diversity_score', 0)
        if diversity < 0.3:
            recommendations.append("🎯 **High concentration** - Consider diversification to reduce dependency risks")
    
    # Based on impact
    if 'impact_analysis' in results:
        h_index = results['impact_analysis'].get('h_index', 0)
        if h_index > 50:
            recommendations.append("🌟 **High impact field** - Leverage visibility for partnerships and funding")
    
    # Based on emerging topics
    if 'emerging_topics' in results:
        recommendations.append("🚀 **Monitor emerging topics** - Early adoption may provide competitive advantage")
    
    # Display recommendations
    for rec in recommendations:
        st.markdown(f"- {rec}")
    
    if not recommendations:
        st.info("Continue monitoring trends and adapting strategies based on evolving patterns")

def export_roadmap_data():
    """Export roadmap data as CSV"""
    
    results = st.session_state.roadmap_results
    
    # Create summary dataframe
    summary_data = []
    
    for module_id, result in results.items():
        if result.get('status') == 'success':
            summary_data.append({
                'Analysis': module_id.replace('_', ' ').title(),
                'Status': 'Success',
                'Key Metrics': str(result.get('summary', 'See detailed report'))
            })
    
    summary_df = pd.DataFrame(summary_data)
    
    csv = summary_df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        "📥 Download Summary (CSV)",
        csv,
        f"roadmap_summary_{datetime.now().strftime('%Y%m%d')}.csv",
        "text/csv",
        key='download-csv'
    )

def get_available_units_for_roadmap(dataset_choice):
    """Get available units based on dataset"""
    
    units = {}
    
    if dataset_choice == "Publications":
        df = st.session_state.publications_data
        
        if 'author' in df.columns:
            units['Authors'] = 'author'
        if 'country' in df.columns:
            units['Countries'] = 'country'
        if 'keywords' in df.columns or 'author_keywords' in df.columns:
            units['Keywords'] = 'keywords' if 'keywords' in df.columns else 'author_keywords'
        if 'journal' in df.columns:
            units['Journals'] = 'journal'
    
    elif dataset_choice == "Patents":
        df = st.session_state.patents_data
        
        if 'inventor' in df.columns:
            units['Inventors'] = 'inventor'
        if 'assignee' in df.columns:
            units['Organizations'] = 'assignee'
        if 'jurisdiction' in df.columns:
            units['Jurisdictions'] = 'jurisdiction'
        if 'ipc_class' in df.columns:
            units['IPC Classes'] = 'ipc_class'
        if 'cpc_class' in df.columns:
            units['CPC Classes'] = 'cpc_class'
    
    else:  # Both
        units['Technology Areas'] = 'combined'
    
    units['Documents'] = 'document'
    
    return units

def get_unit_preview(dataset_choice, selected_unit, top_n):
    """Get preview of selected unit"""
    
    if dataset_choice == "Publications":
        df = st.session_state.publications_data
    elif dataset_choice == "Patents":
        df = st.session_state.patents_data
    else:
        return pd.DataFrame({'Info': ['Both datasets selected']})
    
    units = get_available_units_for_roadmap(dataset_choice)
    col_name = units.get(selected_unit)
    
    if col_name == 'document':
        return pd.DataFrame({
            'Unit': ['Documents'],
            'Count': [len(df)],
            'Description': ['Individual records']
        })
    
    if col_name not in df.columns:
        return None
    
    # Parse and count entities
    all_entities = []
    for entity_str in df[col_name].dropna():
        entities = re.split(r'[;,]', str(entity_str))
        all_entities.extend([e.strip() for e in entities if e.strip()])
    
    entity_counts = Counter(all_entities)
    top_entities = entity_counts.most_common(top_n)
    
    return pd.DataFrame(top_entities, columns=['Entity', 'Count'])
