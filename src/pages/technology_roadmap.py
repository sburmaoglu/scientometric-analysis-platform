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
              options.index(st.session_state.roadmap_config['dataset']) if st.session_state.roadmap_config['dataset'] in options else 0
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
            if len(years) > 0:
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
            for e_str in df[entity_col].dropna()[:1000]:
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
        'Authors': '👥 Track research evolution through author networks',
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
    st.markdown("Select analyses to include in your roadmap")
    
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
        estimated_time = total_selected * 2
        st.metric("Est. Time", f"~{estimated_time}s")
    
    with col3:
        st.metric("Report Sections", total_selected + 2)
    
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
    
    # Display report only if results exist
    if st.session_state.roadmap_results:
        display_roadmap_report()
        
        st.markdown("---")
        
        # Export section
        render_export_section()
        
        st.markdown("---")
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Regenerate Report", use_container_width=True):
                st.session_state.roadmap_results = {}
                st.rerun()
        
        with col2:
            if st.button("⚙️ Modify Configuration", use_container_width=True):
                st.session_state.roadmap_step = 4
                st.rerun()
        
        with col3:
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

def render_export_section():
    """Render export options section"""
    
    st.markdown("## 💾 Export Options")
    
    results = st.session_state.roadmap_results
    config = st.session_state.roadmap_config
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📄 Report Summary")
        
        # Create summary text
        summary_text = f"""# {config.get('title', 'Technology Roadmap Report')}

**Date:** {config.get('report_date', datetime.now().strftime('%Y-%m-%d'))}
**Author:** {config.get('author', 'N/A')}
**Dataset:** {config.get('dataset', 'N/A')}
**Unit:** {config.get('unit', 'N/A')}

## Executive Summary

This technology roadmap analyzes {config.get('dataset')} data using {config.get('unit')} as the primary unit of analysis.

Analyses included: {len(results)}

## Key Findings

"""
        
        # Add findings from each analysis
        for module_id, result in results.items():
            if result.get('status') == 'success':
                summary_text += f"\n### {module_id.replace('_', ' ').title()}\n\n"
                
                if module_id == 'temporal_trends':
                    if result.get('is_comparative'):
                        summary_text += f"- Publications Growth: {result.get('pubs_growth', 0):.1f}%\n"
                        summary_text += f"- Patents Growth: {result.get('pats_growth', 0):.1f}%\n"
                    else:
                        summary_text += f"- Growth Rate: {result.get('growth_rate', 0):.1f}%\n"
                        summary_text += f"- Peak Year: {result.get('peak_year', 'N/A')}\n"
                
                elif module_id == 'diversity_entropy':
                    summary_text += f"- Shannon Entropy: {result.get('entropy', 0):.3f}\n"
                    summary_text += f"- Diversity Score: {result.get('diversity_score', 0):.2f}\n"
                
                elif module_id == 'impact_analysis':
                    if result.get('is_comparative'):
                        summary_text += f"- Pubs Mean Citations: {result.get('pubs_mean', 0):.1f}\n"
                        summary_text += f"- Patents Mean Citations: {result.get('pats_mean', 0):.1f}\n"
                    else:
                        summary_text += f"- Mean Citations: {result.get('mean_citations', 0):.1f}\n"
                        summary_text += f"- H-Index: {result.get('h_index', 0)}\n"
        
        # Download as text
        st.download_button(
            "📝 Download Summary (TXT)",
            summary_text,
            f"roadmap_summary_{datetime.now().strftime('%Y%m%d')}.txt",
            "text/plain",
            use_container_width=True
        )
    
    with col2:
        st.markdown("### 📊 Data Tables")
        
        # Create comprehensive data export
        export_data = []
        
        for module_id, result in results.items():
            if result.get('status') == 'success':
                row = {
                    'Analysis': module_id.replace('_', ' ').title(),
                    'Status': 'Success'
                }
                
                # Add module-specific metrics
                if module_id == 'temporal_trends':
                    if result.get('is_comparative'):
                        row['Pubs_Growth'] = f"{result.get('pubs_growth', 0):.1f}%"
                        row['Pats_Growth'] = f"{result.get('pats_growth', 0):.1f}%"
                    else:
                        row['Growth_Rate'] = f"{result.get('growth_rate', 0):.1f}%"
                        row['Peak_Year'] = result.get('peak_year', 'N/A')
                
                elif module_id == 'diversity_entropy':
                    row['Entropy'] = f"{result.get('entropy', 0):.3f}"
                    row['Diversity_Score'] = f"{result.get('diversity_score', 0):.2f}"
                
                elif module_id == 'impact_analysis':
                    if result.get('is_comparative'):
                        row['Pubs_Mean_Citations'] = f"{result.get('pubs_mean', 0):.1f}"
                        row['Pats_Mean_Citations'] = f"{result.get('pats_mean', 0):.1f}"
                    else:
                        row['Mean_Citations'] = f"{result.get('mean_citations', 0):.1f}"
                        row['H_Index'] = result.get('h_index', 0)
                
                export_data.append(row)
        
        export_df = pd.DataFrame(export_data)
        csv = export_df.to_csv(index=False).encode('utf-8')
        
        st.download_button(
            "📥 Download Data (CSV)",
            csv,
            f"roadmap_data_{datetime.now().strftime('%Y%m%d')}.csv",
            "text/csv",
            use_container_width=True
        )
    
    with col3:
        st.markdown("### 📋 Configuration")
        
        # Export configuration as JSON
        import json
        
        config_export = {
            'title': config.get('title'),
            'subtitle': config.get('subtitle'),
            'author': config.get('author'),
            'date': config.get('report_date'),
            'dataset': config.get('dataset'),
            'unit': config.get('unit'),
            'unit_column': config.get('unit_column'),
            'focus_top_n': config.get('focus_top_n'),
            'time_granularity': config.get('time_granularity'),
            'style': config.get('style'),
            'pipeline': st.session_state.roadmap_pipeline
        }
        
        config_json = json.dumps(config_export, indent=2)
        
        st.download_button(
            "⚙️ Download Config (JSON)",
            config_json,
            f"roadmap_config_{datetime.now().strftime('%Y%m%d')}.json",
            "application/json",
            use_container_width=True
        )
    
    st.info("💡 **Tip:** Save the configuration file to reproduce this exact roadmap in the future!")

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
        df_secondary = None
        dataset_type = 'publications'
    elif dataset_choice == "Patents":
        df = st.session_state.patents_data
        df_secondary = None
        dataset_type = 'patents'
    else:  # Both
        df = st.session_state.publications_data
        df_secondary = st.session_state.patents_data
        dataset_type = 'both'
    
    # Validate data
    if df is None:
        st.error("❌ No data available. Please upload data first.")
        st.session_state.roadmap_results = {}
        return
    
    # Execute each analysis
    for i, module_id in enumerate(pipeline):
        progress = (i + 1) / len(pipeline)
        progress_bar.progress(progress)
        status_text.text(f"Running: {module_id.replace('_', ' ').title()}... ({i+1}/{len(pipeline)})")
        
        try:
            if module_id == 'temporal_trends':
                if dataset_type == 'both' and df_secondary is not None:
                    results[module_id] = run_temporal_trends_both(df, df_secondary, config)
                else:
                    results[module_id] = run_temporal_trends(df, config, dataset_type)
            
            elif module_id == 'diversity_entropy':
                results[module_id] = run_diversity_analysis(df, config, dataset_type)
            
            elif module_id == 'impact_analysis':
                if dataset_type == 'both' and df_secondary is not None:
                    results[module_id] = run_impact_analysis_both(df, df_secondary, config)
                else:
                    results[module_id] = run_impact_analysis(df, config, dataset_type)
            
            elif module_id == 'clustering':
                results[module_id] = run_clustering_analysis(df, config, dataset_type)
            
            elif module_id == 'geographic_evolution':
                if dataset_type == 'both' and df_secondary is not None:
                    results[module_id] = run_geographic_evolution_both(df, df_secondary, config)
                else:
                    results[module_id] = run_geographic_evolution(df, config, dataset_type)
            
            elif module_id == 'emerging_topics':
                results[module_id] = run_emerging_topics(df, config, dataset_type)
            
            if 'status' not in results[module_id]:
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
