"""Home Page"""

import streamlit as st
from datetime import datetime

def render():
    """Render home page"""
    
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