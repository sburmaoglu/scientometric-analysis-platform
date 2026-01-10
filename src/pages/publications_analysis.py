"""Publications Analysis Page"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go

def render():
    """Render publications analysis page"""
    
    st.title("📚 Publications Analysis")
    st.markdown("Comprehensive analysis of publication data")
    
    if st.session_state.publications_data is None:
        st.warning("⚠️ Please upload publications data first")
        with st.expander("📤 How to upload"):
            st.markdown("""
            1. Go to **Data Upload** in the sidebar
            2. Select the **Publications** tab
            3. Upload your CSV/Excel file
            4. Return to this page
            """)
        return
    
    df = st.session_state.publications_data
    
    st.markdown("---")
    
    # Overview
    st.subheader("📊 Overview Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Publications", f"{len(df):,}")
    
    with col2:
        if 'year' in df.columns:
            years = df['year'].dropna()
            if len(years) > 0:
                st.metric("Year Range", f"{int(years.min())}-{int(years.max())}")
    
    with col3:
        if 'citations' in df.columns:
            total_cites = df['citations'].sum()
            if pd.notna(total_cites):
                st.metric("Total Citations", f"{int(total_cites):,}")
    
    with col4:
        if 'journal' in df.columns:
            st.metric("Unique Journals", df['journal'].nunique())
    
    st.markdown("---")
    
    # Analysis tabs
    tab1, tab2, tab3 = st.tabs(["📈 Temporal Trends", "📊 Citation Analysis", "🔍 Data View"])
    
    with tab1:
        render_temporal_tab(df)
    
    with tab2:
        render_citation_tab(df)
    
    with tab3:
        render_data_view_tab(df)

def render_temporal_tab(df):
    """Render temporal analysis tab"""
    st.subheader("📈 Publications Over Time")
    
    if 'year' not in df.columns:
        st.warning("Year column not available")
        return
    
    yearly = df.groupby('year').size().reset_index(name='count')
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=yearly['year'],
        y=yearly['count'],
        mode='lines+markers',
        marker=dict(size=8, color='#3498db', line=dict(width=2, color='white')),
        line=dict(width=3, color='#2980b9'),
        fill='tozeroy',
        fillcolor='rgba(52, 152, 219, 0.1)'
    ))
    
    fig.update_layout(
        title="Publications per Year",
        xaxis_title="Year",
        yaxis_title="Number of Publications",
        template='plotly_white',
        height=450
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Avg per Year", f"{yearly['count'].mean():.1f}")
    with col2:
        peak_idx = yearly['count'].idxmax()
        st.metric("Peak Year", f"{yearly.loc[peak_idx, 'year']:.0f}")
    with col3:
        st.metric("Peak Count", f"{yearly['count'].max():.0f}")

def render_citation_tab(df):
    """Render citation analysis tab"""
    st.subheader("📊 Citation Analysis")
    
    if 'citations' not in df.columns:
        st.warning("Citations column not available")
        return
    
    # Top cited
    st.markdown("#### 🏆 Top Cited Publications")
    top_cited = df.nlargest(10, 'citations')[['title', 'year', 'citations']]
    st.dataframe(top_cited, use_container_width=True, hide_index=True)
    
    # Citation statistics
    st.markdown("#### 📈 Citation Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Mean", f"{df['citations'].mean():.2f}")
    with col2:
        st.metric("Median", f"{df['citations'].median():.0f}")
    with col3:
        st.metric("Max", f"{df['citations'].max():.0f}")

def render_data_view_tab(df):
    """Render data view tab"""
    st.subheader("🔍 Data View")
    
    # Search
    search = st.text_input("🔍 Search in titles", "")
    
    if search:
        mask = df['title'].str.contains(search, case=False, na=False)
        filtered_df = df[mask]
        st.write(f"Found {len(filtered_df)} results")
        st.dataframe(filtered_df.head(50), use_container_width=True)
    else:
        st.dataframe(df.head(50), use_container_width=True)
