"""
Scientometric Analysis Platform - Clean Single File Version
"""

import streamlit as st
import sys
from pathlib import Path

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
                margin: 0;
            '>🔬 ScientoMetrics</h1>
            <p style='color: #666; font-size: 0.95rem; margin: 0.5rem 0 0 0;'>
                Advanced Analysis Platform
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 📑 Navigation")
        
        # Simple radio button navigation
        page = st.radio(
            "Select Page",
            [
                "🏠 Home",
                "📤 Data Upload",
                "📚 Publications Analysis",
                "💡 Patents Analysis",
                "🔄 Comparative Analysis",
                "📈 Temporal Analysis",
                "🗺️ Geographic Analysis"
            ],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### 📊 Data Status")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.session_state.get('publications_data') is not None:
                pub_count = len(st.session_state.publications_data)
                st.metric("📚 Pubs", f"{pub_count:,}")
            else:
                st.info("📚 No Pubs")
        
        with col2:
            if st.session_state.get('patents_data') is not None:
                pat_count = len(st.session_state.patents_data)
                st.metric("💡 Pats", f"{pat_count:,}")
            else:
                st.info("💡 No Pats")
        
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; font-size: 0.7rem; color: #999;'>
            v1.0.0 | Built for Researchers
        </div>
        """, unsafe_allow_html=True)
    
    # Route to pages
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
    elif page == "📈 Temporal Analysis":
        render_temporal_analysis()
    elif page == "🗺️ Geographic Analysis":
        render_geographic_analysis()
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 1rem;'>
        <p style='color: #666; font-size: 0.85rem;'>
            <b>ScientoMetrics</b> | Advanced Scientometric Analysis
        </p>
    </div>
    """, unsafe_allow_html=True)

# ==================== HOME PAGE ====================
def render_home():
    """Home page"""
    from datetime import datetime
    
    st.markdown("""
    <div style='text-align: center; padding: 3rem 0 2rem 0;'>
        <h1 style='font-size: 3.5rem; margin-bottom: 1rem;'>🔬 ScientoMetrics</h1>
        <p style='font-size: 1.4rem; color: #666; margin-bottom: 0.5rem;'>
            Advanced Scientometric Analysis Platform
        </p>
        <p style='font-size: 1rem; color: #999;'>
            Publication-Ready Statistical Analysis
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
        st.metric("📊 Analyses Available", "7")
    
    with col4:
        st.metric("🕒 Session Time", datetime.now().strftime("%H:%M"))
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Key Features
        
        ✨ **Publication-Ready Analysis**
        - Rigorous statistical testing
        - Complete methodology documentation
        - Professional visualizations
        
        📊 **Advanced Analytics**
        - Temporal trend analysis
        - Geographic distribution
        - Citation analysis
        - Technology classification
        """)
    
    with col2:
        st.markdown("""
        ### 🚀 Getting Started
        
        **1. Upload Data**
        Go to "Data Upload" and upload your files
        
        **2. Explore**
        Choose an analysis from the sidebar
        
        **3. Analyze**
        View interactive charts and statistics
        
        **4. Export**
        Download results and visualizations
        """)
    
    st.markdown("---")
    
    if st.session_state.publications_data is None and st.session_state.patents_data is None:
        st.info("👉 **Ready to start?** Upload your data in the **Data Upload** page!")
    else:
        st.success("✅ **Data loaded!** Explore the analysis pages in the sidebar.")

# ==================== DATA UPLOAD ====================
def render_data_upload():
    """Data upload page"""
    from pages import data_upload
    data_upload.render()

# ==================== PUBLICATIONS ANALYSIS ====================
def render_publications_analysis():
    """Publications analysis page"""
    import pandas as pd
    import plotly.graph_objects as go
    from collections import Counter
    
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
        st.subheader("📈 Publications Over Time")
        
        if 'year' in df.columns:
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
        else:
            st.warning("Year column not available")
    
    with tab2:
        st.subheader("📊 Citation Analysis")
        
        if 'citations' in df.columns:
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
        else:
            st.warning("Citations column not available")
    
    with tab3:
        st.subheader("🔍 Data View")
        
        # Show data with search
        search = st.text_input("🔍 Search in titles", "")
        
        if search:
            mask = df['title'].str.contains(search, case=False, na=False)
            filtered_df = df[mask]
            st.write(f"Found {len(filtered_df)} results")
            st.dataframe(filtered_df.head(50), use_container_width=True)
        else:
            st.dataframe(df.head(50), use_container_width=True)

# ==================== PATENTS ANALYSIS ====================
def render_patents_analysis():
    """Patents analysis page"""
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    from collections import Counter
    
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
    
    # Overview
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
    
    # Analysis tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Temporal", "🏢 Organizations", "🗺️ Geographic", "🔬 Technology"])
    
    with tab1:
        st.subheader("📈 Patents Over Time")
        
        if 'year' in df.columns:
            yearly = df.groupby('year').size().reset_index(name='count')
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=yearly['year'],
                y=yearly['count'],
                mode='lines+markers',
                marker=dict(size=8, color='#e74c3c'),
                line=dict(width=3, color='#c0392b'),
                fill='tozeroy',
                fillcolor='rgba(231, 76, 60, 0.1)'
            ))
            
            fig.update_layout(
                title="Patent Filings per Year",
                xaxis_title="Year",
                yaxis_title="Count",
                template='plotly_white',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg per Year", f"{yearly['count'].mean():.1f}")
            with col2:
                st.metric("Peak Year", f"{yearly.loc[yearly['count'].idxmax(), 'year']:.0f}")
            with col3:
                st.metric("Peak Count", f"{yearly['count'].max():.0f}")
    
    with tab2:
        st.subheader("🏢 Top Organizations")
        
        if 'assignee' in df.columns:
            all_orgs = []
            for org_str in df['assignee'].dropna():
                if pd.notna(org_str):
                    orgs = str(org_str).split(';')
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
                    title="Top 15 Patent Holders",
                    color='Patents',
                    color_continuous_scale='Reds'
                )
                
                fig.update_layout(
                    yaxis={'categoryorder': 'total ascending'},
                    template='plotly_white',
                    height=500
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(top_orgs, use_container_width=True, hide_index=True)
    
    with tab3:
        st.subheader("🗺️ Geographic Distribution")
        
        if 'jurisdiction' in df.columns:
            juris_counts = df['jurisdiction'].value_counts().head(15)
            
            geo_df = pd.DataFrame({
                'Jurisdiction': juris_counts.index,
                'Count': juris_counts.values
            })
            
            fig = px.bar(
                geo_df,
                x='Jurisdiction',
                y='Count',
                title="Patents by Jurisdiction",
                color='Count',
                color_continuous_scale='Blues'
            )
            
            fig.update_layout(template='plotly_white', height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Jurisdictions", len(df['jurisdiction'].unique()))
            with col2:
                top_pct = (geo_df.iloc[0]['Count'] / len(df)) * 100
                st.metric("Top Jurisdiction Share", f"{top_pct:.1f}%")
    
    with tab4:
        st.subheader("🔬 Technology Classification")
        
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
                
                fig.update_layout(template='plotly_white', height=450)
                st.plotly_chart(fig, use_container_width=True)

# ==================== COMPARATIVE ANALYSIS ====================
def render_comparative_analysis():
    """Comparative analysis"""
    import plotly.graph_objects as go
    
    st.title("🔄 Comparative Analysis")
    st.markdown("Compare publications and patents data")
    
    has_pubs = st.session_state.publications_data is not None
    has_pats = st.session_state.patents_data is not None
    
    if not has_pubs or not has_pats:
        st.warning("⚠️ Both publications and patents data required")
        return
    
    pubs_df = st.session_state.publications_data
    pats_df = st.session_state.patents_data
    
    st.markdown("---")
    
    # Overview
    st.subheader("📊 Dataset Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("📚 Publications", f"{len(pubs_df):,}")
        if 'year' in pubs_df.columns:
            years = pubs_df['year'].dropna()
            if len(years) > 0:
                st.metric("Pub Year Range", f"{int(years.min())}-{int(years.max())}")
    
    with col2:
        st.metric("💡 Patents", f"{len(pats_df):,}")
        if 'year' in pats_df.columns:
            years = pats_df['year'].dropna()
            if len(years) > 0:
                st.metric("Pat Year Range", f"{int(years.min())}-{int(years.max())}")
    
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
            line=dict(color='#3498db', width=3),
            marker=dict(size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=pat_yearly.index,
            y=pat_yearly.values,
            mode='lines+markers',
            name='Patents',
            line=dict(color='#e74c3c', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title="Publications vs Patents Over Time",
            xaxis_title="Year",
            yaxis_title="Count",
            template='plotly_white',
            height=450,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)

# ==================== TEMPORAL ANALYSIS ====================
def render_temporal_analysis():
    """Advanced temporal analysis"""
    st.title("📈 Temporal Analysis")
    st.markdown("Advanced time-series analysis")
    
    if st.session_state.publications_data is None and st.session_state.patents_data is None:
        st.warning("⚠️ Please upload data first")
        return
    
    st.info("🚧 Advanced temporal forecasting features coming soon!")

# ==================== GEOGRAPHIC ANALYSIS ====================
def render_geographic_analysis():
    """Geographic analysis"""
    st.title("🗺️ Geographic Analysis")
    st.markdown("Geographic distribution and collaboration")
    
    if st.session_state.publications_data is None and st.session_state.patents_data is None:
        st.warning("⚠️ Please upload data first")
        return
    
    st.info("🚧 Interactive maps and geospatial features coming soon!")

if __name__ == "__main__":
    main()
