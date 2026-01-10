"""
Scientometric Analysis Platform - Simplified Version
All analyses in one file for reliability
"""

import streamlit as st
import sys
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from config.settings import PAGE_CONFIG, THEME_CONFIG, CUSTOM_CSS
from utils.session_state import initialize_session_state

st.set_page_config(**PAGE_CONFIG)
initialize_session_state()
st.markdown(THEME_CONFIG, unsafe_allow_html=True)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def main():
    """Main application"""
    
    with st.sidebar:
        st.markdown("""
        <div style='text-align: center; padding: 1.5rem 0;'>
            <h1 style='
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-size: 2.2rem;
                font-weight: 700;
            '>🔬 ScientoMetrics</h1>
            <p style='color: #666; font-size: 0.95rem;'>Advanced Analysis Platform</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navigation
        page = st.radio(
            "Navigation",
            ["🏠 Home", "📤 Data Upload", "📚 Publications Analysis", 
             "💡 Patents Analysis", "🔄 Comparative Analysis"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Data Status
        st.markdown("### 📊 Data Status")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.session_state.get('publications_data') is not None:
                pub_count = len(st.session_state.publications_data)
                st.metric("📚 Publications", f"{pub_count:,}")
            else:
                st.info("📚 No Data")
        
        with col2:
            if st.session_state.get('patents_data') is not None:
                pat_count = len(st.session_state.patents_data)
                st.metric("💡 Patents", f"{pat_count:,}")
            else:
                st.info("💡 No Data")
    
    # Route to selected page
    if page == "🏠 Home":
        render_home()
    elif page == "📤 Data Upload":
        render_data_upload()
    elif page == "📚 Publications Analysis":
        render_publications_analysis()
    elif page == "💡 Patents Analysis":
        render_patents_analysis()
    elif page == "🔄 Comparative Analysis":
        render_comparative_analysis()

# ==================== HOME PAGE ====================
def render_home():
    """Render home page"""
    from datetime import datetime
    
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1 style='font-size: 3rem;'>🔬 ScientoMetrics</h1>
        <p style='font-size: 1.3rem; color: #666;'>
            Advanced Scientometric Analysis Platform
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        pub_count = len(st.session_state.publications_data) if st.session_state.publications_data is not None else 0
        st.metric("📚 Publications", f"{pub_count:,}")
    
    with col2:
        pat_count = len(st.session_state.patents_data) if st.session_state.patents_data is not None else 0
        st.metric("💡 Patents", f"{pat_count:,}")
    
    with col3:
        st.metric("📊 Analyses", len(st.session_state.analysis_cache))
    
    with col4:
        st.metric("🕒 Time", datetime.now().strftime("%H:%M"))
    
    st.markdown("---")
    
    if st.session_state.publications_data is None and st.session_state.patents_data is None:
        st.info("👉 **Ready to start?** Upload your data in the **Data Upload** page!")

# ==================== DATA UPLOAD PAGE ====================
def render_data_upload():
    """Render data upload page"""
    from pages import data_upload
    data_upload.render()

# ==================== PUBLICATIONS ANALYSIS ====================
def render_publications_analysis():
    """Publications analysis"""
    
    st.title("📚 Publications Analysis")
    
    if st.session_state.publications_data is None:
        st.warning("⚠️ Please upload publications data first")
        return
    
    df = st.session_state.publications_data
    
    st.markdown("---")
    
    # Overview
    st.subheader("📊 Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total", f"{len(df):,}")
    
    with col2:
        if 'year' in df.columns:
            years = df['year'].dropna()
            if len(years) > 0:
                st.metric("Year Range", f"{int(years.min())}-{int(years.max())}")
    
    with col3:
        if 'citations' in df.columns:
            st.metric("Total Citations", f"{int(df['citations'].sum()):,}")
    
    with col4:
        if 'journal' in df.columns:
            st.metric("Unique Journals", df['journal'].nunique())
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📈 Temporal", "📊 Citations", "🔍 Details"])
    
    with tab1:
        st.subheader("Publications Over Time")
        
        if 'year' in df.columns:
            yearly = df.groupby('year').size().reset_index(name='count')
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=yearly['year'],
                y=yearly['count'],
                mode='lines+markers',
                marker=dict(size=8, color='#3498db'),
                line=dict(width=3)
            ))
            
            fig.update_layout(
                title="Publications per Year",
                xaxis_title="Year",
                yaxis_title="Count",
                template='plotly_white',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Citation Analysis")
        
        if 'citations' in df.columns:
            top_cited = df.nlargest(10, 'citations')[['title', 'year', 'citations']]
            st.dataframe(top_cited, use_container_width=True, hide_index=True)
    
    with tab3:
        st.subheader("Data Details")
        st.dataframe(df.head(20), use_container_width=True)

# ==================== PATENTS ANALYSIS ====================
def render_patents_analysis():
    """Patents analysis"""
    
    st.title("💡 Patents Analysis")
    
    if st.session_state.patents_data is None:
        st.warning("⚠️ Please upload patents data first")
        return
    
    df = st.session_state.patents_data
    
    st.markdown("---")
    
    # Overview
    st.subheader("📊 Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total", f"{len(df):,}")
    
    with col2:
        if 'year' in df.columns:
            years = df['year'].dropna()
            if len(years) > 0:
                st.metric("Year Range", f"{int(years.min())}-{int(years.max())}")
    
    with col3:
        if 'forward_citations' in df.columns:
            st.metric("Total Citations", f"{int(df['forward_citations'].sum()):,}")
    
    with col4:
        if 'jurisdiction' in df.columns:
            st.metric("Jurisdictions", df['jurisdiction'].nunique())
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Temporal", "🏢 Organizations", "🗺️ Geographic", "🔬 Technology"])
    
    with tab1:
        st.subheader("Patents Over Time")
        
        if 'year' in df.columns:
            yearly = df.groupby('year').size().reset_index(name='count')
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=yearly['year'],
                y=yearly['count'],
                mode='lines+markers',
                marker=dict(size=8, color='#e74c3c'),
                line=dict(width=3)
            ))
            
            fig.update_layout(
                title="Patents per Year",
                xaxis_title="Year",
                yaxis_title="Count",
                template='plotly_white',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Top Organizations")
        
        if 'assignee' in df.columns:
            all_orgs = []
            for orgs_str in df['assignee'].dropna():
                if pd.notna(orgs_str):
                    orgs = str(orgs_str).split(';')
                    all_orgs.extend([o.strip() for o in orgs if o.strip()])
            
            if all_orgs:
                org_counts = Counter(all_orgs)
                top_orgs = pd.DataFrame(
                    org_counts.most_common(15),
                    columns=['Organization', 'Patents']
                )
                
                fig = px.bar(
                    top_orgs,
                    x='Patents',
                    y='Organization',
                    orientation='h',
                    title="Top 15 Patent Holders"
                )
                
                fig.update_layout(
                    yaxis={'categoryorder': 'total ascending'},
                    template='plotly_white',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Geographic Distribution")
        
        if 'jurisdiction' in df.columns:
            juris_counts = df['jurisdiction'].value_counts().head(15)
            
            fig = px.bar(
                x=juris_counts.index,
                y=juris_counts.values,
                title="Patents by Jurisdiction",
                labels={'x': 'Jurisdiction', 'y': 'Count'}
            )
            
            fig.update_layout(template='plotly_white', height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Technology Classification")
        
        if 'ipc_class' in df.columns:
            all_classes = []
            for class_str in df['ipc_class'].dropna():
                if pd.notna(class_str):
                    classes = str(class_str).split(';')
                    all_classes.extend([c.strip()[:4] for c in classes if c.strip()])
            
            if all_classes:
                class_counts = Counter(all_classes)
                top_classes = pd.DataFrame(
                    class_counts.most_common(10),
                    columns=['Classification', 'Count']
                )
                
                fig = px.pie(
                    top_classes,
                    values='Count',
                    names='Classification',
                    title="Top 10 IPC Classifications"
                )
                
                st.plotly_chart(fig, use_container_width=True)

# ==================== COMPARATIVE ANALYSIS ====================
def render_comparative_analysis():
    """Comparative analysis"""
    
    st.title("🔄 Comparative Analysis")
    
    has_pubs = st.session_state.publications_data is not None
    has_pats = st.session_state.patents_data is not None
    
    if not has_pubs or not has_pats:
        st.warning("⚠️ Please upload both publications and patents data")
        return
    
    pubs_df = st.session_state.publications_data
    pats_df = st.session_state.patents_data
    
    st.markdown("---")
    
    # Overview
    st.subheader("📊 Dataset Comparison")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Publications", f"{len(pubs_df):,}")
        st.metric("Patents", f"{len(pats_df):,}")
    
    with col2:
        if 'year' in pubs_df.columns:
            pub_years = f"{pubs_df['year'].min():.0f}-{pubs_df['year'].max():.0f}"
            st.metric("Pub Years", pub_years)
        if 'year' in pats_df.columns:
            pat_years = f"{pats_df['year'].min():.0f}-{pats_df['year'].max():.0f}"
            st.metric("Pat Years", pat_years)
    
    with col3:
        if 'citations' in pubs_df.columns:
            st.metric("Pub Citations", f"{pubs_df['citations'].sum():,.0f}")
        if 'forward_citations' in pats_df.columns:
            st.metric("Pat Citations", f"{pats_df['forward_citations'].sum():,.0f}")
    
    st.markdown("---")
    
    # Temporal comparison
    st.subheader("📈 Temporal Trends Comparison")
    
    if 'year' in pubs_df.columns and 'year' in pats_df.columns:
        pub_yearly = pubs_df.groupby('year').size()
        pat_yearly = pats_df.groupby('year').size()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pub_yearly.index,
            y=pub_yearly.values,
            mode='lines+markers',
            name='Publications',
            line=dict(color='#3498db', width=3)
        ))
        fig.add_trace(go.Scatter(
            x=pat_yearly.index,
            y=pat_yearly.values,
            mode='lines+markers',
            name='Patents',
            line=dict(color='#e74c3c', width=3)
        ))
        
        fig.update_layout(
            title="Publications vs Patents Over Time",
            xaxis_title="Year",
            yaxis_title="Count",
            template='plotly_white',
            height=450
        )
        
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
