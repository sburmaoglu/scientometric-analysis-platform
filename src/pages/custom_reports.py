"""Custom Reports & Dashboards - User-Configurable Analytics"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import re
from collections import Counter

def render():
    """Render custom reports & dashboards page"""
    
    st.title("📊 Custom Reports & Dashboards")
    st.markdown("Build personalized dashboards with flexible visualizations and units")
    
    if st.session_state.publications_data is None and st.session_state.patents_data is None:
        st.warning("⚠️ Please upload data first")
        return
    
    st.markdown("---")
    
    # Dashboard mode selection
    mode = st.selectbox(
        "Select Mode",
        [
            "🎨 Dashboard Builder",
            "📈 Pre-built Templates",
            "📄 Executive Summary",
            "🔬 Research Impact Report",
            "💡 Innovation Landscape",
            "📊 Comparative Dashboard"
        ]
    )
    
    st.markdown("---")
    
    if mode == "🎨 Dashboard Builder":
        render_dashboard_builder()
    
    elif mode == "📈 Pre-built Templates":
        render_templates()
    
    elif mode == "📄 Executive Summary":
        render_executive_summary()
    
    elif mode == "🔬 Research Impact Report":
        render_research_impact()
    
    elif mode == "💡 Innovation Landscape":
        render_innovation_landscape()
    
    elif mode == "📊 Comparative Dashboard":
        render_comparative_dashboard()

def get_available_units(df, dataset_type):
    """Get available analysis units"""
    
    units = {}
    
    # Entity-based
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
    keyword_candidates = ['keywords', 'author_keywords', 'Keywords', 'ipc_class', 'cpc_class']
    for col in keyword_candidates:
        if col in df.columns:
            units['Keywords/Classes'] = col
            break
    
    # Geographic
    geo_col = 'country' if dataset_type == 'publications' else 'jurisdiction'
    if geo_col in df.columns:
        units['Geographic'] = geo_col
    
    # Temporal
    if 'year' in df.columns:
        units['Temporal'] = 'year'
    
    return units

def render_dashboard_builder():
    """Interactive dashboard builder"""
    
    st.subheader("🎨 Dashboard Builder")
    st.markdown("Create your custom dashboard by selecting dataset, units, and visualizations")
    
    # Dataset selection
    dataset = st.radio("Select Dataset", ["Publications", "Patents", "Both"], horizontal=True)
    
    if dataset == "Publications":
        if st.session_state.publications_data is None:
            st.warning("⚠️ Upload publications data")
            return
        df = st.session_state.publications_data
        dataset_type = 'publications'
    elif dataset == "Patents":
        if st.session_state.patents_data is None:
            st.warning("⚠️ Upload patents data")
            return
        df = st.session_state.patents_data
        dataset_type = 'patents'
    else:
        # Both datasets
        if st.session_state.publications_data is None or st.session_state.patents_data is None:
            st.warning("⚠️ Upload both datasets")
            return
        df = None  # Will handle separately
        dataset_type = 'both'
    
    st.markdown("---")
    
    # Dashboard configuration
    st.markdown("### ⚙️ Dashboard Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        dashboard_title = st.text_input("Dashboard Title", "My Custom Dashboard")
    
    with col2:
        layout = st.selectbox("Layout", ["2 Columns", "3 Columns", "1 Column"])
    
    st.markdown("---")
    
    # Widget selection
    st.markdown("### 📊 Select Visualizations")
    
    available_widgets = [
        "📈 Time Series Trend",
        "🏆 Top Entities Ranking",
        "🌍 Geographic Distribution",
        "📊 Citation Distribution",
        "🎯 Keyword Cloud",
        "📉 Growth Rate Analysis",
        "🔥 Heatmap (Entity x Year)",
        "🌐 Network Graph",
        "📊 Sankey Diagram",
        "🎨 Sunburst Chart",
        "📈 Area Chart (Cumulative)",
        "🎯 Gauge Chart (KPIs)",
        "📊 Correlation Matrix",
        "🌊 Stream Graph"
    ]
    
    selected_widgets = st.multiselect(
        "Choose up to 6 visualizations",
        available_widgets,
        default=available_widgets[:3],
        max_selections=6
    )
    
    if not selected_widgets:
        st.warning("Please select at least one visualization")
        return
    
    st.markdown("---")
    
    # Generate dashboard
    if st.button("🚀 Generate Dashboard", type="primary"):
        generate_custom_dashboard(df, dataset_type, dashboard_title, layout, selected_widgets)

def generate_custom_dashboard(df, dataset_type, title, layout, widgets):
    """Generate custom dashboard with selected widgets"""
    
    st.markdown("---")
    st.markdown(f"# {title}")
    st.markdown(f"*Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}*")
    
    # Determine column layout
    if layout == "2 Columns":
        cols_per_row = 2
    elif layout == "3 Columns":
        cols_per_row = 3
    else:
        cols_per_row = 1
    
    # Render widgets
    for i in range(0, len(widgets), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for j, col in enumerate(cols):
            if i + j < len(widgets):
                with col:
                    render_widget(widgets[i + j], df, dataset_type)
    
    # Export button
    st.markdown("---")
    st.markdown("### 💾 Export Dashboard")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Export as PDF"):
            st.info("PDF export feature coming soon")
    
    with col2:
        if st.button("📊 Export Data"):
            st.info("Data export feature coming soon")
    
    with col3:
        if st.button("🖼️ Save as Image"):
            st.info("Image export feature coming soon")

def render_widget(widget_name, df, dataset_type):
    """Render individual widget"""
    
    st.markdown(f"**{widget_name}**")
    
    try:
        if "Time Series" in widget_name:
            render_time_series_widget(df, dataset_type)
        
        elif "Top Entities" in widget_name:
            render_top_entities_widget(df, dataset_type)
        
        elif "Geographic" in widget_name:
            render_geographic_widget(df, dataset_type)
        
        elif "Citation Distribution" in widget_name:
            render_citation_widget(df, dataset_type)
        
        elif "Keyword Cloud" in widget_name:
            render_keyword_cloud_widget(df, dataset_type)
        
        elif "Growth Rate" in widget_name:
            render_growth_rate_widget(df, dataset_type)
        
        elif "Heatmap" in widget_name:
            render_heatmap_widget(df, dataset_type)
        
        elif "Network Graph" in widget_name:
            render_network_widget(df, dataset_type)
        
        elif "Sankey" in widget_name:
            render_sankey_widget(df, dataset_type)
        
        elif "Sunburst" in widget_name:
            render_sunburst_widget(df, dataset_type)
        
        elif "Area Chart" in widget_name:
            render_area_chart_widget(df, dataset_type)
        
        elif "Gauge" in widget_name:
            render_gauge_widget(df, dataset_type)
        
        elif "Correlation" in widget_name:
            render_correlation_widget(df, dataset_type)
        
        elif "Stream Graph" in widget_name:
            render_stream_graph_widget(df, dataset_type)
        
        else:
            st.info(f"Widget {widget_name} under development")
    
    except Exception as e:
        st.error(f"Error rendering widget: {str(e)}")

def render_time_series_widget(df, dataset_type):
    """Time series trend widget"""
    
    if 'year' not in df.columns:
        st.warning("Year data not available")
        return
    
    yearly = df.groupby('year').size()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=yearly.index,
        y=yearly.values,
        mode='lines',
        fill='tozeroy',
        line=dict(width=2, color='#3498db'),
        fillcolor='rgba(52, 152, 219, 0.2)'
    ))
    
    fig.update_layout(
        title="Publications Over Time" if dataset_type == 'publications' else "Patents Over Time",
        showlegend=False,
        height=300,
        margin=dict(l=0, r=0, t=40, b=0),
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_top_entities_widget(df, dataset_type):
    """Top entities ranking widget"""
    
    entity_col = 'author' if dataset_type == 'publications' else 'assignee'
    
    if entity_col not in df.columns:
        st.warning(f"{entity_col} not available")
        return
    
    # Parse entities
    all_entities = []
    for entities_str in df[entity_col].dropna():
        entities = re.split(r'[;,]', str(entities_str))
        all_entities.extend([e.strip() for e in entities if e.strip()])
    
    entity_counts = Counter(all_entities)
    top_entities = pd.DataFrame(
        entity_counts.most_common(10),
        columns=['Entity', 'Count']
    )
    
    fig = px.bar(
        top_entities,
        x='Count',
        y='Entity',
        orientation='h',
        color='Count',
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        title=f"Top {'Authors' if dataset_type == 'publications' else 'Organizations'}",
        yaxis={'categoryorder': 'total ascending'},
        height=300,
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_geographic_widget(df, dataset_type):
    """Geographic distribution widget"""
    
    geo_col = 'country' if dataset_type == 'publications' else 'jurisdiction'
    
    if geo_col not in df.columns:
        st.warning("Geographic data not available")
        return
    
    geo_counts = df[geo_col].value_counts().head(10)
    
    fig = px.pie(
        values=geo_counts.values,
        names=geo_counts.index,
        hole=0.4
    )
    
    fig.update_layout(
        title="Geographic Distribution",
        height=300,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_citation_widget(df, dataset_type):
    """Citation distribution widget"""
    
    citation_col = 'citations' if dataset_type == 'publications' else 'forward_citations'
    
    if citation_col not in df.columns:
        st.warning("Citation data not available")
        return
    
    citations = df[citation_col].dropna()
    
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=citations,
        nbinsx=30,
        marker_color='#e74c3c',
        opacity=0.7
    ))
    
    fig.update_layout(
        title="Citation Distribution",
        xaxis_title="Citations",
        yaxis_title="Count",
        height=300,
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_keyword_cloud_widget(df, dataset_type):
    """Keyword cloud widget"""
    
    keyword_col = 'keywords' if dataset_type == 'publications' else 'ipc_class'
    
    if keyword_col not in df.columns:
        st.info("Keyword data not available - showing placeholder")
        return
    
    # Parse keywords
    all_keywords = []
    for keywords_str in df[keyword_col].dropna():
        keywords = re.split(r'[;,]', str(keywords_str))
        keywords = [k.strip().lower() for k in keywords if k.strip()]
        all_keywords.extend(keywords[:5])  # Limit per document
    
    keyword_counts = Counter(all_keywords)
    top_keywords = keyword_counts.most_common(20)
    
    # Create bubble chart as word cloud alternative
    keywords_df = pd.DataFrame(top_keywords, columns=['keyword', 'count'])
    keywords_df['size'] = keywords_df['count'] / keywords_df['count'].max() * 100
    
    fig = px.scatter(
        keywords_df.head(15),
        x=np.random.rand(15),
        y=np.random.rand(15),
        size='size',
        text='keyword',
        color='count',
        color_continuous_scale='Viridis'
    )
    
    fig.update_traces(textposition='middle center', textfont_size=10)
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    
    fig.update_layout(
        title="Keyword Landscape",
        height=300,
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_growth_rate_widget(df, dataset_type):
    """Growth rate analysis widget"""
    
    if 'year' not in df.columns:
        st.warning("Year data not available")
        return
    
    yearly = df.groupby('year').size()
    growth_rate = yearly.pct_change() * 100
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=growth_rate.index,
        y=growth_rate.values,
        marker_color=np.where(growth_rate.values >= 0, '#2ecc71', '#e74c3c')
    ))
    
    fig.update_layout(
        title="Year-over-Year Growth Rate (%)",
        yaxis_title="Growth %",
        height=300,
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_heatmap_widget(df, dataset_type):
    """Entity x Year heatmap widget"""
    
    if 'year' not in df.columns:
        st.warning("Year data not available")
        return
    
    entity_col = 'author' if dataset_type == 'publications' else 'assignee'
    
    if entity_col not in df.columns:
        st.warning(f"{entity_col} not available")
        return
    
    # Parse entities
    entity_year_data = []
    
    for idx, row in df.iterrows():
        if pd.notna(row.get(entity_col)) and pd.notna(row.get('year')):
            entities = re.split(r'[;,]', str(row[entity_col]))
            for entity in entities:
                entity = entity.strip()
                if entity:
                    entity_year_data.append({
                        'entity': entity,
                        'year': row['year']
                    })
    
    if not entity_year_data:
        st.warning("No entity-year data")
        return
    
    entity_year_df = pd.DataFrame(entity_year_data)
    
    # Get top entities
    top_entities = entity_year_df['entity'].value_counts().head(10).index
    
    # Create pivot table
    heatmap_data = entity_year_df[entity_year_df['entity'].isin(top_entities)]
    pivot = heatmap_data.groupby(['entity', 'year']).size().reset_index(name='count')
    pivot = pivot.pivot(index='entity', columns='year', values='count').fillna(0)
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale='Blues'
    ))
    
    fig.update_layout(
        title="Entity Activity Heatmap",
        height=300,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_network_widget(df, dataset_type):
    """Network graph widget"""
    
    st.info("🌐 Interactive network graph - requires additional computation")
    
    # Placeholder for network visualization
    st.markdown("**Network visualization available in Advanced Analytics > Link Prediction**")

def render_sankey_widget(df, dataset_type):
    """Sankey diagram widget"""
    
    if 'year' not in df.columns:
        st.warning("Year data not available")
        return
    
    # Create period-based flow
    df['period'] = pd.cut(df['year'], bins=3, labels=['Early', 'Middle', 'Recent'])
    
    geo_col = 'country' if dataset_type == 'publications' else 'jurisdiction'
    
    if geo_col not in df.columns:
        st.warning("Geographic data not available")
        return
    
    # Get top regions per period
    top_regions = df[geo_col].value_counts().head(5).index
    
    flow_data = df[df[geo_col].isin(top_regions)].groupby(['period', geo_col]).size().reset_index(name='value')
    
    # Create Sankey
    periods = ['Early', 'Middle', 'Recent']
    all_labels = list(periods) + list(top_regions)
    
    source = []
    target = []
    value = []
    
    for i, period in enumerate(periods[:-1]):
        period_data = flow_data[flow_data['period'] == period]
        for _, row in period_data.iterrows():
            source.append(all_labels.index(period))
            target.append(all_labels.index(row[geo_col]))
            value.append(row['value'])
    
    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            label=all_labels,
            color='lightblue'
        ),
        link=dict(
            source=source,
            target=target,
            value=value
        )
    )])
    
    fig.update_layout(
        title="Temporal-Geographic Flow",
        height=300,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_sunburst_widget(df, dataset_type):
    """Sunburst chart widget"""
    
    if 'year' not in df.columns:
        st.warning("Year data not available")
        return
    
    # Create hierarchical data: decade > year > category
    df['decade'] = (df['year'] // 10) * 10
    
    geo_col = 'country' if dataset_type == 'publications' else 'jurisdiction'
    
    if geo_col in df.columns:
        sunburst_data = df.groupby(['decade', 'year', geo_col]).size().reset_index(name='value')
        
        fig = px.sunburst(
            sunburst_data.head(100),
            path=['decade', 'year', geo_col],
            values='value',
            color='value',
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            title="Hierarchical Distribution",
            height=300,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Geographic data needed for sunburst")

def render_area_chart_widget(df, dataset_type):
    """Cumulative area chart widget"""
    
    if 'year' not in df.columns:
        st.warning("Year data not available")
        return
    
    yearly = df.groupby('year').size()
    cumulative = yearly.cumsum()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=cumulative.index,
        y=cumulative.values,
        mode='lines',
        fill='tozeroy',
        line=dict(width=0),
        fillcolor='rgba(52, 152, 219, 0.5)'
    ))
    
    fig.update_layout(
        title="Cumulative Growth",
        yaxis_title="Total Count",
        height=300,
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_gauge_widget(df, dataset_type):
    """KPI gauge chart widget"""
    
    # Calculate growth rate
    if 'year' in df.columns:
        yearly = df.groupby('year').size()
        if len(yearly) > 1:
            recent_growth = ((yearly.iloc[-1] - yearly.iloc[-2]) / yearly.iloc[-2] * 100)
        else:
            recent_growth = 0
    else:
        recent_growth = 0
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=recent_growth,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Annual Growth Rate"},
        delta={'reference': 0, 'suffix': '%'},
        gauge={
            'axis': {'range': [-50, 50]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [-50, 0], 'color': "lightgray"},
                {'range': [0, 50], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 25
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_correlation_widget(df, dataset_type):
    """Correlation matrix widget"""
    
    st.info("Correlation analysis requires multiple numeric features")
    
    # Prepare features
    features = []
    feature_names = []
    
    if 'year' in df.columns:
        features.append(df['year'].fillna(df['year'].mean()))
        feature_names.append('Year')
    
    citation_col = 'citations' if dataset_type == 'publications' else 'forward_citations'
    if citation_col in df.columns:
        features.append(np.log1p(df[citation_col].fillna(0)))
        feature_names.append('Log Citations')
    
    if len(features) >= 2:
        feature_matrix = np.column_stack(features)
        corr_matrix = np.corrcoef(feature_matrix.T)
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix,
            x=feature_names,
            y=feature_names,
            colorscale='RdBu',
            zmid=0
        ))
        
        fig.update_layout(
            title="Feature Correlations",
            height=250,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        st.plotly_chart(fig, use_container_width=True)

def render_stream_graph_widget(df, dataset_type):
    """Stream graph widget"""
    
    if 'year' not in df.columns:
        st.warning("Year data not available")
        return
    
    geo_col = 'country' if dataset_type == 'publications' else 'jurisdiction'
    
    if geo_col not in df.columns:
        st.warning("Geographic data not available")
        return
    
    # Get top regions
    top_regions = df[geo_col].value_counts().head(5).index
    
    # Create data for stream graph
    stream_data = df[df[geo_col].isin(top_regions)].groupby(['year', geo_col]).size().reset_index(name='count')
    
    fig = px.area(
        stream_data,
        x='year',
        y='count',
        color=geo_col,
        line_group=geo_col
    )
    
    fig.update_layout(
        title="Regional Trends (Stream Graph)",
        height=300,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_templates():
    """Pre-built dashboard templates"""
    
    st.subheader("📈 Pre-built Templates")
    st.markdown("Choose from professionally designed dashboard templates")
    
    templates = {
        "📊 Overview Dashboard": "Comprehensive overview with key metrics",
        "📈 Trend Analysis": "Focus on temporal trends and growth",
        "🌍 Geographic Analysis": "Geographic distribution and collaboration",
        "👥 Collaboration Network": "Author/inventor collaboration patterns",
        "💡 Innovation Metrics": "Patent-specific innovation indicators",
        "🎯 Impact Assessment": "Citation analysis and research impact"
    }
    
    selected_template = st.selectbox("Select Template", list(templates.keys()))
    
    st.info(templates[selected_template])
    
    if st.button("🚀 Load Template", type="primary"):
        st.success(f"Loading {selected_template}...")
        
        # Load appropriate template
        if "Overview" in selected_template:
            render_overview_template()
        elif "Trend" in selected_template:
            render_trend_template()
        elif "Geographic" in selected_template:
            render_geographic_template()
        else:
            st.info("Template loading...")

def render_overview_template():
    """Overview dashboard template"""
    
    st.markdown("## 📊 Overview Dashboard")
    
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
    
    # KPIs
    st.markdown("### 📊 Key Performance Indicators")
    
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
            for e_str in df[entity_col].dropna():
                all_entities.extend(re.split(r'[;,]', str(e_str)))
            st.metric("Unique Entities", f"{len(set(all_entities)):,}")
    
    st.markdown("---")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        render_time_series_widget(df, dataset_type)
    
    with col2:
        render_top_entities_widget(df, dataset_type)
    
    col1, col2 = st.columns(2)
    
    with col1:
        render_geographic_widget(df, dataset_type)
    
    with col2:
        render_citation_widget(df, dataset_type)

def render_trend_template():
    """Trend analysis template"""
    st.info("Trend template - showing temporal patterns")

def render_geographic_template():
    """Geographic analysis template"""
    st.info("Geographic template - showing spatial patterns")

def render_executive_summary():
    """Executive summary report"""
    
    st.subheader("📄 Executive Summary")
    st.markdown("High-level overview for decision makers")
    
    # Check available data
    has_pubs = st.session_state.publications_data is not None
    has_pats = st.session_state.patents_data is not None
    
    if not has_pubs and not has_pats:
        st.warning("No data available")
        return
    
    st.markdown("---")
    
    # Executive summary content
    st.markdown("## 🎯 Key Findings")
    
    if has_pubs:
        pubs_df = st.session_state.publications_data
        
        st.markdown("### 📚 Publications")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Publications", f"{len(pubs_df):,}")
        
        with col2:
            if 'citations' in pubs_df.columns:
                avg_cites = pubs_df['citations'].mean()
                st.metric("Avg Citations", f"{avg_cites:.1f}")
        
        with col3:
            if 'year' in pubs_df.columns:
                yearly = pubs_df.groupby('year').size()
                if len(yearly) > 1:
                    growth = ((yearly.iloc[-1] - yearly.iloc[0]) / yearly.iloc[0] * 100)
                    st.metric("Total Growth", f"{growth:+.1f}%")
    
    if has_pats:
        st.markdown("---")
        pats_df = st.session_state.patents_data
        
        st.markdown("### 💡 Patents")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Patents", f"{len(pats_df):,}")
        
        with col2:
            if 'forward_citations' in pats_df.columns:
                avg_cites = pats_df['forward_citations'].mean()
                st.metric("Avg Citations", f"{avg_cites:.1f}")
        
        with col3:
            if 'year' in pats_df.columns:
                yearly = pats_df.groupby('year').size()
                if len(yearly) > 1:
                    growth = ((yearly.iloc[-1] - yearly.iloc[0]) / yearly.iloc[0] * 100)
                    st.metric("Total Growth", f"{growth:+.1f}%")
    
    st.markdown("---")
    st.markdown("## 📈 Trends and Insights")
    
    # Add visualizations
    if has_pubs and has_pats:
        # Comparative view
        pubs_yearly = pubs_df.groupby('year').size()
        pats_yearly = pats_df.groupby('year').size()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=pubs_yearly.index,
            y=pubs_yearly.values,
            mode='lines+markers',
            name='Publications',
            line=dict(color='#3498db', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=pats_yearly.index,
            y=pats_yearly.values,
            mode='lines+markers',
            name='Patents',
            line=dict(color='#e74c3c', width=3)
        ))
        
        fig.update_layout(
            title="Publications vs Patents Over Time",
            template='plotly_white',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

def render_research_impact():
    """Research impact report"""
    st.subheader("🔬 Research Impact Report")
    st.info("Detailed research impact analysis - under development")

def render_innovation_landscape():
    """Innovation landscape report"""
    st.subheader("💡 Innovation Landscape")
    st.info("Technology innovation landscape - under development")

def render_comparative_dashboard():
    """Comparative dashboard"""
    st.subheader("📊 Comparative Dashboard")
    st.info("Side-by-side comparison dashboard - under development")