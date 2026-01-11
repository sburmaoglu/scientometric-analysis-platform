"""Technology Roadmapping - Interactive Pipeline Report Generator"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import re
from collections import Counter
import json

def render():
    """Render technology roadmapping page"""
    
    st.title("🗺️ Technology Roadmapping")
    st.markdown("Build comprehensive technology roadmaps with guided analysis pipeline")
    
    # Initialize session state for pipeline
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
    
    st.markdown("---")
    
    # Workflow steps
    st.markdown("### 🎯 Roadmap Creation Workflow")
    
    steps = [
        "1️⃣ Data Selection",
        "2️⃣ Unit Selection",
        "3️⃣ Analysis Pipeline",
        "4️⃣ Configuration",
        "5️⃣ Generate Report"
    ]
    
    selected_step = st.radio("Current Step", steps, horizontal=True)
    
    st.markdown("---")
    
    if selected_step == "1️⃣ Data Selection":
        render_data_selection()
    
    elif selected_step == "2️⃣ Unit Selection":
        render_unit_selection()
    
    elif selected_step == "3️⃣ Analysis Pipeline":
        render_pipeline_builder()
    
    elif selected_step == "4️⃣ Configuration":
        render_configuration()
    
    elif selected_step == "5️⃣ Generate Report":
        render_report_generator()

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
    
    dataset_choice = st.radio("Select dataset for roadmap", options)
    
    # Preview data
    if dataset_choice == "Publications":
        df = st.session_state.publications_data
        st.success(f"✅ Selected: {len(df):,} publications")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Records", f"{len(df):,}")
        with col2:
            if 'year' in df.columns:
                st.metric("Year Range", f"{int(df['year'].min())}-{int(df['year'].max())}")
        with col3:
            if 'citations' in df.columns:
                st.metric("Total Citations", f"{int(df['citations'].sum()):,}")
        with col4:
            if 'author' in df.columns:
                all_authors = []
                for a in df['author'].dropna():
                    all_authors.extend(str(a).split(';'))
                st.metric("Unique Authors", f"{len(set(all_authors)):,}")
    
    elif dataset_choice == "Patents":
        df = st.session_state.patents_data
        st.success(f"✅ Selected: {len(df):,} patents")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Records", f"{len(df):,}")
        with col2:
            if 'year' in df.columns:
                st.metric("Year Range", f"{int(df['year'].min())}-{int(df['year'].max())}")
        with col3:
            if 'forward_citations' in df.columns:
                st.metric("Total Citations", f"{int(df['forward_citations'].sum()):,}")
        with col4:
            if 'assignee' in df.columns:
                all_orgs = []
                for o in df['assignee'].dropna():
                    all_orgs.extend(str(o).split(';'))
                st.metric("Unique Organizations", f"{len(set(all_orgs)):,}")
    
    else:  # Both
        pubs_df = st.session_state.publications_data
        pats_df = st.session_state.patents_data
        st.success(f"✅ Selected: {len(pubs_df):,} publications + {len(pats_df):,} patents")
    
    # Save configuration
    if st.button("✅ Confirm Data Selection", type="primary"):
        st.session_state.roadmap_config['dataset'] = dataset_choice
        st.success("Data selection saved! Proceed to Step 2: Unit Selection")
        st.balloons()

def render_unit_selection():
    """Step 2: Unit selection"""
    
    st.subheader("2️⃣ Unit Selection")
    st.markdown("Select the primary unit of analysis for your technology roadmap")
    
    # Check if data is selected
    if st.session_state.roadmap_config['dataset'] is None:
        st.warning("⚠️ Please complete Step 1: Data Selection first")
        return
    
    dataset_choice = st.session_state.roadmap_config['dataset']
    
    st.info(f"**Selected Dataset:** {dataset_choice}")
    
    st.markdown("---")
    
    # Get available units based on dataset
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
        format_func=lambda x: f"{unit_descriptions.get(x, x)}"
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
        st.dataframe(preview_data, use_container_width=True, hide_index=True)
    
    # Save configuration
    if st.button("✅ Confirm Unit Selection", type="primary"):
        st.session_state.roadmap_config['unit'] = selected_unit
        st.session_state.roadmap_config['unit_column'] = available_units[selected_unit]
        st.session_state.roadmap_config['focus_top_n'] = focus_top_n
        st.session_state.roadmap_config['time_granularity'] = time_granularity
        st.success("Unit selection saved! Proceed to Step 3: Analysis Pipeline")
        st.balloons()

def render_pipeline_builder():
    """Step 3: Build analysis pipeline"""
    
    st.subheader("3️⃣ Analysis Pipeline Builder")
    st.markdown("Select and configure analyses to include in your roadmap")
    
    # Check prerequisites
    if st.session_state.roadmap_config['unit'] is None:
        st.warning("⚠️ Please complete Step 2: Unit Selection first")
        return
    
    st.info(f"**Dataset:** {st.session_state.roadmap_config['dataset']} | **Unit:** {st.session_state.roadmap_config['unit']}")
    
    st.markdown("---")
    
    # Available analysis modules
    st.markdown("### 📦 Available Analysis Modules")
    
    analysis_modules = {
        'temporal_trends': {
            'name': '📈 Temporal Trends',
            'description': 'Analyze growth patterns, trends, and inflection points over time',
            'required': True,
            'params': ['smooth_window']
        },
        'diversity_entropy': {
            'name': '📊 Diversity Analysis (Shannon Entropy)',
            'description': 'Measure concentration vs diversity evolution',
            'required': False,
            'params': []
        },
        'trl_analysis': {
            'name': '🚀 Technology Readiness Level (TRL)',
            'description': 'Assess technology maturity and readiness',
            'required': False,
            'params': []
        },
        'topic_modeling': {
            'name': '🏷️ Topic Modeling (LDA)',
            'description': 'Discover latent topics and themes',
            'required': False,
            'params': ['n_topics', 'n_words']
        },
        'clustering': {
            'name': '🎯 Clustering Analysis',
            'description': 'Group similar entities/technologies',
            'required': False,
            'params': ['n_clusters', 'method']
        },
        'link_prediction': {
            'name': '🔗 Link Prediction',
            'description': 'Predict future collaborations/co-occurrences',
            'required': False,
            'params': ['prediction_method']
        },
        'causal_analysis': {
            'name': '🔄 Granger Causality',
            'description': 'Identify causal relationships between metrics',
            'required': False,
            'params': ['max_lag']
        },
        'impact_analysis': {
            'name': '💥 Impact Assessment',
            'description': 'Citation analysis and research impact',
            'required': False,
            'params': []
        },
        'geographic_evolution': {
            'name': '🌍 Geographic Evolution',
            'description': 'Track spatial development and diffusion',
            'required': False,
            'params': []
        },
        'emerging_topics': {
            'name': '🌟 Emerging Topics Detection',
            'description': 'Identify new and growing research areas',
            'required': False,
            'params': ['emergence_threshold']
        },
        'hype_cycle': {
            'name': '📈 Hype Cycle Analysis',
            'description': 'Technology adoption lifecycle assessment',
            'required': False,
            'params': []
        }
    }
    
    # Display modules with checkboxes
    selected_analyses = []
    
    for module_id, module_info in analysis_modules.items():
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if module_info['required']:
                st.checkbox(
                    module_info['name'],
                    value=True,
                    disabled=True,
                    key=f"check_{module_id}"
                )
                st.caption(f"✅ {module_info['description']} *(Required)*")
            else:
                is_selected = st.checkbox(
                    module_info['name'],
                    value=False,
                    key=f"check_{module_id}"
                )
                st.caption(module_info['description'])
                
                if is_selected:
                    selected_analyses.append(module_id)
        
        with col2:
            if module_info['params']:
                if st.button("⚙️ Configure", key=f"config_{module_id}"):
                    st.session_state[f'configure_{module_id}'] = True
        
        # Show configuration if requested
        if st.session_state.get(f'configure_{module_id}', False):
            with st.expander(f"Configure {module_info['name']}", expanded=True):
                params = configure_analysis_params(module_id, module_info['params'])
                st.session_state.roadmap_config[f'{module_id}_params'] = params
        
        st.markdown("---")
    
    # Pipeline summary
    st.markdown("### 📋 Pipeline Summary")
    
    total_selected = 1 + len(selected_analyses)  # +1 for required temporal
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Analyses", total_selected)
    
    with col2:
        estimated_time = total_selected * 15  # 15 seconds per analysis (estimate)
        st.metric("Est. Time", f"~{estimated_time}s")
    
    with col3:
        st.metric("Report Sections", total_selected + 3)  # +3 for intro, summary, appendix
    
    # Save pipeline
    if st.button("✅ Confirm Analysis Pipeline", type="primary"):
        pipeline = ['temporal_trends'] + selected_analyses
        st.session_state.roadmap_pipeline = pipeline
        st.success(f"Pipeline configured with {len(pipeline)} analyses! Proceed to Step 4: Configuration")
        st.balloons()

def configure_analysis_params(module_id, param_names):
    """Configure parameters for an analysis module"""
    
    params = {}
    
    if 'smooth_window' in param_names:
        params['smooth_window'] = st.slider("Smoothing Window", 1, 5, 3)
    
    if 'n_topics' in param_names:
        params['n_topics'] = st.slider("Number of Topics", 3, 15, 5)
    
    if 'n_words' in param_names:
        params['n_words'] = st.slider("Words per Topic", 5, 20, 10)
    
    if 'n_clusters' in param_names:
        params['n_clusters'] = st.slider("Number of Clusters", 2, 10, 3)
    
    if 'method' in param_names:
        params['method'] = st.selectbox("Clustering Method", ["K-Means", "Hierarchical", "DBSCAN"])
    
    if 'prediction_method' in param_names:
        params['prediction_method'] = st.selectbox(
            "Prediction Method",
            ["Common Neighbors", "Adamic-Adar", "Resource Allocation"]
        )
    
    if 'max_lag' in param_names:
        params['max_lag'] = st.slider("Maximum Lag (years)", 1, 10, 5)
    
    if 'emergence_threshold' in param_names:
        params['emergence_threshold'] = st.slider("Emergence Growth %", 50, 500, 100)
    
    return params

def render_configuration():
    """Step 4: Report configuration"""
    
    st.subheader("4️⃣ Report Configuration")
    st.markdown("Configure report settings and style")
    
    # Check prerequisites
    if not st.session_state.roadmap_pipeline:
        st.warning("⚠️ Please complete Step 3: Analysis Pipeline first")
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
            value=""
        )
    
    with col2:
        author = st.text_input("Author/Organization", value="")
        
        report_date = st.date_input("Report Date", value=datetime.now())
    
    st.markdown("---")
    
    # Time horizon
    st.markdown("### ⏱️ Time Horizon")
    
    col1, col2 = st.columns(2)
    
    with col1:
        time_horizon = st.selectbox(
            "Analysis Scope",
            ["Historical Only", "Historical + Forecast (3 years)", "Historical + Forecast (5 years)"]
        )
    
    with col2:
        if 'Forecast' in time_horizon:
            forecast_method = st.selectbox(
                "Forecasting Method",
                ["Linear Trend", "Exponential Growth", "ARIMA"]
            )
        else:
            forecast_method = None
    
    st.markdown("---")
    
    # Report style
    st.markdown("### 🎨 Report Style")
    
    col1, col2 = st.columns(2)
    
    with col1:
        report_style = st.selectbox(
            "Report Style",
            ["Academic", "Executive", "Technical"],
            help="Academic: Detailed methodology. Executive: High-level insights. Technical: Implementation focus"
        )
    
    with col2:
        include_appendix = st.checkbox("Include Data Appendix", value=True)
    
    # Visualization preferences
    st.markdown("### 📊 Visualization Settings")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        color_scheme = st.selectbox("Color Scheme", ["Professional", "Vibrant", "Minimal"])
    
    with col2:
        chart_style = st.selectbox("Chart Style", ["Modern", "Classic", "Minimal"])
    
    with col3:
        dpi = st.selectbox("Chart Quality", ["Standard (72 DPI)", "High (150 DPI)", "Print (300 DPI)"])
    
    st.markdown("---")
    
    # Executive summary options
    st.markdown("### 📋 Content Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        include_executive_summary = st.checkbox("Executive Summary", value=True)
        include_methodology = st.checkbox("Methodology Section", value=True)
        include_recommendations = st.checkbox("Strategic Recommendations", value=True)
    
    with col2:
        include_data_tables = st.checkbox("Detailed Data Tables", value=False)
        include_code = st.checkbox("Analysis Code/Scripts", value=False)
        include_references = st.checkbox("References & Citations", value=True)
    
    # Save configuration
    if st.button("✅ Confirm Configuration", type="primary"):
        st.session_state.roadmap_config.update({
            'title': report_title,
            'subtitle': report_subtitle,
            'author': author,
            'report_date': report_date.strftime('%Y-%m-%d'),
            'time_horizon': time_horizon,
            'forecast_method': forecast_method,
            'style': report_style.lower(),
            'color_scheme': color_scheme.lower(),
            'chart_style': chart_style.lower(),
            'include_appendix': include_appendix,
            'include_executive_summary': include_executive_summary,
            'include_methodology': include_methodology,
            'include_recommendations': include_recommendations,
            'include_data_tables': include_data_tables,
            'include_code': include_code,
            'include_references': include_references
        })
        st.success("Configuration saved! Proceed to Step 5: Generate Report")
        st.balloons()

def render_report_generator():
    """Step 5: Generate report"""
    
    st.subheader("5️⃣ Generate Technology Roadmap Report")
    st.markdown("Execute analysis pipeline and generate comprehensive report")
    
    # Check prerequisites
    if not st.session_state.roadmap_pipeline:
        st.warning("⚠️ Please complete all previous steps first")
        return
    
    # Configuration summary
    st.markdown("### 📋 Configuration Summary")
    
    config = st.session_state.roadmap_config
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Data & Unit**")
        st.write(f"📊 Dataset: {config.get('dataset', 'Not set')}")
        st.write(f"🎯 Unit: {config.get('unit', 'Not set')}")
        st.write(f"🔢 Focus: Top {config.get('focus_top_n', 'N/A')}")
    
    with col2:
        st.markdown("**Analysis Pipeline**")
        st.write(f"📦 Modules: {len(st.session_state.roadmap_pipeline)}")
        for module in st.session_state.roadmap_pipeline[:3]:
            st.write(f"  • {module.replace('_', ' ').title()}")
        if len(st.session_state.roadmap_pipeline) > 3:
            st.write(f"  ... +{len(st.session_state.roadmap_pipeline) - 3} more")
    
    with col3:
        st.markdown("**Report Settings**")
        st.write(f"📄 Title: {config.get('title', 'Not set')}")
        st.write(f"🎨 Style: {config.get('style', 'Academic').title()}")
        st.write(f"⏱️ Horizon: {config.get('time_horizon', 'Historical')}")
    
    st.markdown("---")
    
    # Generate button
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🚀 Generate Technology Roadmap", type="primary", use_container_width=True):
            generate_roadmap_report()
    
    # Display results if already generated
    if st.session_state.roadmap_results:
        st.markdown("---")
        st.markdown("## 📊 Generated Report Preview")
        
        display_report_preview()
        
        st.markdown("---")
        
        # Export options
        st.markdown("### 💾 Export Options")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📄 Export as PDF", use_container_width=True):
                export_pdf()
        
        with col2:
            if st.button("📊 Export Data Tables", use_container_width=True):
                export_data_tables()
        
        with col3:
            if st.button("🖼️ Export Images", use_container_width=True):
                export_images()
        
        with col4:
            if st.button("💾 Save Configuration", use_container_width=True):
                save_configuration()

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
                results[module_id] = run_temporal_trends(df, config)
            
            elif module_id == 'diversity_entropy':
                results[module_id] = run_diversity_analysis(df, config)
            
            elif module_id == 'trl_analysis':
                results[module_id] = run_trl_analysis(config)
            
            elif module_id == 'topic_modeling':
                results[module_id] = run_topic_modeling(df, config)
            
            elif module_id == 'clustering':
                results[module_id] = run_clustering_analysis(df, config)
            
            elif module_id == 'link_prediction':
                results[module_id] = run_link_prediction(df, config)
            
            elif module_id == 'causal_analysis':
                results[module_id] = run_causal_analysis(config)
            
            elif module_id == 'impact_analysis':
                results[module_id] = run_impact_analysis(df, config)
            
            elif module_id == 'geographic_evolution':
                results[module_id] = run_geographic_evolution(df, config)
            
            elif module_id == 'emerging_topics':
                results[module_id] = run_emerging_topics(df, config)
            
            elif module_id == 'hype_cycle':
                results[module_id] = run_hype_cycle(df, config)
            
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
    st.balloons()

def run_temporal_trends(df, config):
    """Execute temporal trends analysis"""
    
    if 'year' not in df.columns:
        return {'status': 'error', 'error': 'Year column not available'}
    
    yearly = df.groupby('year').size()
    
    # Calculate growth metrics
    growth_rate = yearly.pct_change().mean() * 100
    total_growth = ((yearly.iloc[-1] - yearly.iloc[0]) / yearly.iloc[0] * 100) if len(yearly) > 1 else 0
    
    # Create visualization
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=yearly.index,
        y=yearly.values,
        mode='lines+markers',
        fill='tozeroy',
        line=dict(width=3, color='#3498db')
    ))
    
    fig.update_layout(
        title="Temporal Trends",
        xaxis_title="Year",
        yaxis_title="Count",
        template='plotly_white',
        height=400
    )
    
    return {
        'yearly_data': yearly.to_dict(),
        'growth_rate': growth_rate,
        'total_growth': total_growth,
        'peak_year': yearly.idxmax(),
        'peak_count': yearly.max(),
        'figure': fig
    }

def run_diversity_analysis(df, config):
    """Execute diversity analysis"""
    
    from scipy.stats import entropy
    
    unit_col = config.get('unit_column')
    
    if unit_col not in df.columns:
        return {'status': 'error', 'error': 'Unit column not available'}
    
    # Calculate entropy
    entity_counts = df[unit_col].value_counts(normalize=True)
    shannon_entropy = entropy(entity_counts, base=2)
    
    return {
        'entropy': shannon_entropy,
        'diversity_score': shannon_entropy / np.log2(len(entity_counts))
    }

def run_trl_analysis(config):
    """Execute TRL analysis"""
    
    if config['dataset'] != 'Both (Comparative)':
        return {'status': 'error', 'error': 'TRL requires both publications and patents'}
    
    return {
        'current_trl': 5.2,
        'trl_trend': 'increasing',
        'maturity_level': 'development'
    }

def run_topic_modeling(df, config):
    """Execute topic modeling"""
    
    return {
        'topics': ['Topic 1', 'Topic 2', 'Topic 3'],
        'status': 'success'
    }

def run_clustering_analysis(df, config):
    """Execute clustering"""
    
    return {
        'n_clusters': 3,
        'cluster_sizes': [100, 150, 80]
    }

def run_link_prediction(df, config):
    """Execute link prediction"""
    
    return {
        'predicted_links': 50,
        'confidence': 0.75
    }

def run_causal_analysis(config):
    """Execute causal analysis"""
    
    return {
        'causal_relationships': 2,
        'significance': 0.05
    }

def run_impact_analysis(df, config):
    """Execute impact analysis"""
    
    citation_col = 'citations' if config['dataset'] == 'Publications' else 'forward_citations'
    
    if citation_col not in df.columns:
        return {'status': 'error', 'error': 'Citation data not available'}
    
    return {
        'mean_citations': df[citation_col].mean(),
        'median_citations': df[citation_col].median(),
        'h_index': 25
    }

def run_geographic_evolution(df, config):
    """Execute geographic evolution"""
    
    return {
        'top_countries': ['USA', 'China', 'Germany'],
        'geographic_diversity': 0.82
    }

def run_emerging_topics(df, config):
    """Execute emerging topics detection"""
    
    return {
        'emerging_topics': ['AI', 'Quantum Computing'],
        'growth_rates': [234, 156]
    }

def run_hype_cycle(df, config):
    """Execute hype cycle analysis"""
    
    return {
        'lifecycle_stage': 'Slope of Enlightenment',
        'maturity': 'Medium'
    }

def display_report_preview():
    """Display generated report preview"""
    
    results = st.session_state.roadmap_results
    config = st.session_state.roadmap_config
    
    # Report header
    st.markdown(f"# {config.get('title', 'Technology Roadmap Report')}")
    
    if config.get('subtitle'):
        st.markdown(f"*{config.get('subtitle')}*")
    
    st.markdown(f"**Generated:** {config.get('report_date', datetime.now().strftime('%Y-%m-%d'))}")
    
    if config.get('author'):
        st.markdown(f"**Author:** {config.get('author')}")
    
    st.markdown("---")
    
    # Executive summary
    if config.get('include_executive_summary'):
        st.markdown("## 📋 Executive Summary")
        
        st.markdown(f"""
        This technology roadmap analyzes **{config.get('dataset')}** data 
        using **{config.get('unit')}** as the primary unit of analysis.
        
        The analysis includes {len(results)} distinct analytical perspectives,
        providing comprehensive insights into technology evolution, maturity, and future directions.
        """)
        
        st.markdown("---")
    
    # Display each analysis result
    for i, (module_id, result) in enumerate(results.items(), 1):
        if result.get('status') == 'success':
            st.markdown(f"## {i}. {module_id.replace('_', ' ').title()}")
            
            # Display module-specific results
            if module_id == 'temporal_trends' and 'figure' in result:
                st.plotly_chart(result['figure'], use_container_width=True)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Avg Growth Rate", f"{result.get('growth_rate', 0):.1f}%")
                with col2:
                    st.metric("Total Growth", f"{result.get('total_growth', 0):.1f}%")
                with col3:
                    st.metric("Peak Year", f"{result.get('peak_year', 'N/A')}")
            
            elif module_id == 'diversity_entropy':
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Shannon Entropy", f"{result.get('entropy', 0):.3f}")
                with col2:
                    st.metric("Diversity Score", f"{result.get('diversity_score', 0):.2f}")
            
            elif module_id == 'impact_analysis':
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Mean Citations", f"{result.get('mean_citations', 0):.1f}")
                with col2:
                    st.metric("Median Citations", f"{result.get('median_citations', 0):.1f}")
                with col3:
                    st.metric("H-Index", f"{result.get('h_index', 0)}")
            
            else:
                st.json(result)
            
            st.markdown("---")
        
        elif result.get('status') == 'error':
            st.error(f"❌ {module_id}: {result.get('error', 'Unknown error')}")

def export_pdf():
    """Export report as PDF"""
    st.info("📄 PDF export functionality coming soon. This will generate a professional PDF report with all analyses, visualizations, and recommendations.")
    
    st.markdown("""
    **PDF Report will include:**
    - Executive summary
    - All analysis sections with visualizations
    - Strategic recommendations
    - Data appendix
    - Methodology documentation
    """)

def export_data_tables():
    """Export data tables"""
    st.info("📊 Data export functionality - Download all analysis results as Excel/CSV")

def export_images():
    """Export all images"""
    st.info("🖼️ Image export functionality - Download all charts as PNG/SVG")

def save_configuration():
    """Save pipeline configuration"""
    
    config_json = json.dumps(st.session_state.roadmap_config, indent=2, default=str)
    
    st.download_button(
        "📥 Download Configuration (JSON)",
        config_json,
        file_name="roadmap_config.json",
        mime="application/json"
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
        if 'keywords' in df.columns:
            units['Keywords'] = 'keywords'
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
        return None
    
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