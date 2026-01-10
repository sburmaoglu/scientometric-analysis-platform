"""Patents Analysis Page"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

def render():
    """Render patents analysis page"""
    
    st.title("💡 Patents Analysis")
    st.markdown("Comprehensive analysis of patent data")
    
    if st.session_state.patents_data is None:
        st.warning("⚠️ Please upload patents data first")
        with st.expander("📤 How to upload"):
            st.markdown("""
            1. Go to **Data Upload** in the sidebar
            2. Select the **Patents** tab
            3. Upload your CSV/Excel file
            4. Return to this page
            """)
        return
    
    df = st.session_state.patents_data
    
    st.markdown("---")
    
    # Overview Metrics
    st.subheader("📊 Overview Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Patents", f"{len(df):,}")
    
    with col2:
        if 'year' in df.columns:
            years = df['year'].dropna()
            if len(years) > 0:
                st.metric("Year Range", f"{int(years.min())}-{int(years.max())}")
    
    with col3:
        if 'forward_citations' in df.columns:
            total_cites = df['forward_citations'].sum()
            if pd.notna(total_cites):
                st.metric("Total Citations", f"{int(total_cites):,}")
    
    with col4:
        if 'family_size' in df.columns:
            avg_family = df['family_size'].mean()
            if pd.notna(avg_family):
                st.metric("Avg Family Size", f"{avg_family:.1f}")
    
    st.markdown("---")
    
    # Analysis Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Temporal",
        "🏢 Organizations",
        "🗺️ Geographic",
        "🔬 Technology"
    ])
    
    with tab1:
        render_temporal_tab(df)
    
    with tab2:
        render_organizations_tab(df)
    
    with tab3:
        render_geographic_tab(df)
    
    with tab4:
        render_technology_tab(df)

def render_temporal_tab(df):
    """Temporal analysis tab"""
    st.subheader("📈 Patents Over Time")
    
    if 'year' not in df.columns:
        st.warning("Year column not available")
        return
    
    yearly = df.groupby('year').size().reset_index(name='count')
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=yearly['year'],
        y=yearly['count'],
        mode='lines+markers',
        marker=dict(size=8, color='#e74c3c', line=dict(width=2, color='white')),
        line=dict(width=3, color='#c0392b'),
        fill='tozeroy',
        fillcolor='rgba(231, 76, 60, 0.1)'
    ))
    
    fig.update_layout(
        title="Patent Filings per Year",
        xaxis_title="Year",
        yaxis_title="Count",
        template='plotly_white',
        height=450
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Avg per Year", f"{yearly['count'].mean():.1f}")
    with col2:
        st.metric("Peak Year", f"{yearly.loc[yearly['count'].idxmax(), 'year']:.0f}")
    with col3:
        st.metric("Peak Count", f"{yearly['count'].max():.0f}")

def render_organizations_tab(df):
    """Organizations analysis tab"""
    st.subheader("🏢 Top Patent Holders")
    
    if 'assignee' not in df.columns:
        st.warning("Assignee column not available")
        return
    
    # Parse organizations
    all_orgs = []
    for org_str in df['assignee'].dropna():
        if pd.notna(org_str):
            orgs = str(org_str).split(';')
            all_orgs.extend([o.strip() for o in orgs if o.strip()])
    
    if not all_orgs:
        st.info("No organization data available")
        return
    
    # Count and create dataframe
    org_counts = Counter(all_orgs)
    top_orgs = pd.DataFrame(
        org_counts.most_common(15),
        columns=['Organization', 'Patents']
    )
    
    # Horizontal bar chart
    fig = px.bar(
        top_orgs,
        x='Patents',
        y='Organization',
        orientation='h',
        title="Top 15 Patent Holders",
        color='Patents',
        color_continuous_scale='Reds'
    )
    
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        template='plotly_white',
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Unique Organizations", f"{len(org_counts):,}")
    
    with col2:
        top_pct = (top_orgs.iloc[0]['Patents'] / len(df)) * 100
        st.metric("Top Org Share", f"{top_pct:.1f}%")
    
    with col3:
        top10_pct = (top_orgs.head(10)['Patents'].sum() / len(df)) * 100
        st.metric("Top 10 Share", f"{top10_pct:.1f}%")
    
    # Full table
    with st.expander("📊 View Complete Table"):
        st.dataframe(top_orgs, use_container_width=True, hide_index=True)

def render_geographic_tab(df):
    """Geographic distribution tab"""
    st.subheader("🗺️ Geographic Distribution")
    
    if 'jurisdiction' not in df.columns:
        st.warning("Jurisdiction column not available")
        return
    
    # Count by jurisdiction
    juris_counts = df['jurisdiction'].value_counts().head(20)
    
    geo_df = pd.DataFrame({
        'Jurisdiction': juris_counts.index,
        'Count': juris_counts.values
    })
    
    # Bar chart
    fig = px.bar(
        geo_df.head(15),
        x='Jurisdiction',
        y='Count',
        title="Patents by Jurisdiction (Top 15)",
        color='Count',
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        template='plotly_white',
        height=450
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Jurisdictions", len(df['jurisdiction'].unique()))
    
    with col2:
        st.metric("Top Jurisdiction", geo_df.iloc[0]['Jurisdiction'])
    
    with col3:
        top_pct = (geo_df.iloc[0]['Count'] / len(df)) * 100
        st.metric("Top Share", f"{top_pct:.1f}%")
    
    # Full table
    with st.expander("📊 All Jurisdictions"):
        st.dataframe(geo_df, use_container_width=True, hide_index=True)

def render_technology_tab(df):
    """Technology classification tab"""
    st.subheader("🔬 Technology Classification")
    
    # Check available classifications
    has_ipc = 'ipc_class' in df.columns
    has_cpc = 'cpc_class' in df.columns
    
    if not has_ipc and not has_cpc:
        st.warning("No classification data available")
        return
    
    # Choose classification
    class_col = 'ipc_class' if has_ipc else 'cpc_class'
    class_name = 'IPC' if has_ipc else 'CPC'
    
    st.markdown(f"**Analyzing {class_name} Classifications**")
    
    # Parse classifications
    all_classes = []
    for class_str in df[class_col].dropna():
        if pd.notna(class_str):
            classes = str(class_str).split(';')
            main_classes = [c.strip()[:4] for c in classes if c.strip()]
            all_classes.extend(main_classes)
    
    if not all_classes:
        st.info("No classification data found")
        return
    
    # Count
    class_counts = Counter(all_classes)
    top_classes = pd.DataFrame(
        class_counts.most_common(15),
        columns=['Classification', 'Count']
    )
    
    # Pie chart
    fig = px.pie(
        top_classes.head(10),
        values='Count',
        names='Classification',
        title=f"Top 10 {class_name} Classifications",
        hole=0.3
    )
    
    fig.update_traces(textposition='inside', textinfo='label+percent')
    fig.update_layout(template='plotly_white', height=500)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Metrics
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Unique Classifications", f"{len(class_counts):,}")
    
    with col2:
        st.metric("Most Common", top_classes.iloc[0]['Classification'])
    
    # Table
    with st.expander("📊 Complete Classification List"):
        st.dataframe(top_classes, use_container_width=True, hide_index=True)