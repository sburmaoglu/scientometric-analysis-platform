"""Home Page"""

import streamlit as st
from datetime import datetime

def render():
    """Render home page"""

    # --- Header ---
    st.markdown("""
    <div style='text-align: center; padding: 3rem 0 2rem 0;'>
        <h1 style='font-size: 3.6rem; margin-bottom: 0.6rem;'>🔬 ScientoMetrics</h1>
        <p style='font-size: 1.4rem; color: #555; margin-bottom: 0.3rem;'>
            Advanced Scientometric Analysis Platform for Technology Foresight
        </p>
        <p style='font-size: 1rem; color: #888; max-width: 720px; margin: 0 auto;'>
            Transforming publication and patent data into publication-ready insights
            using rigorous statistical, network, and AI-driven methodologies.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # --- Key Metrics ---
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        pub_count = len(st.session_state.publications_data) if st.session_state.publications_data is not None else 0
        st.metric("📚 Publications", f"{pub_count:,}")

    with col2:
        pat_count = len(st.session_state.patents_data) if st.session_state.patents_data is not None else 0
        st.metric("💡 Patents", f"{pat_count:,}")

    with col3:
        st.metric("🧠 Analytical Modules", "40+")

    with col4:
        st.metric("🕒 Session Time", datetime.now().strftime("%H:%M"))

    st.markdown("---")

    # --- Core Value Proposition ---
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 🎯 What Makes ScientoMetrics Different?

        **Scientometrics × Technology Foresight**
        - Integrated analysis of **publications and patents**
        - Evidence-based support for **research strategy and policy**
        - Designed for **academics, R&D managers, and decision makers**

        **Publication-Ready by Design**
        - Statistically sound methodologies
        - Transparent assumptions and metrics
        - Professional, exportable visualizations
        """)

    with col2:
        st.markdown("""
        ### 🔬 Analytical Capabilities

        - 📈 Temporal trend & growth analysis  
        - 🌍 Geographic & institutional diversity (entropy-based)  
        - 🧠 Topic modeling (LDA, STM) & evolution tracking  
        - 🔗 Network analysis & link prediction  
        - 🔄 Causal inference (Granger causality, cross-correlation)  
        - 🗺️ Technology roadmapping & TRL assessment  
        """)

    st.markdown("---")

    # --- Getting Started ---
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 🚀 Getting Started

        **1. Upload Data**  
        Import publication or patent data (Lens.org CSV format)

        **2. Select Analysis**  
        Choose from scientometric, AI, or foresight modules

        **3. Generate Insights**  
        Explore interactive charts and statistical results

        **4. Build Roadmaps**  
        Combine analyses into comprehensive foresight pipelines
        """)

    with col2:
        st.markdown("""
        ### 📊 Supported Data

        - **Publications:** Scholarly works metadata  
        - **Patents:** IPC/CPC classifications, citations  
        - **Native Lens.org support**
        - Graceful handling of missing or partial data

        *Other sources (Scopus, WoS, PubMed) can be mapped to the standard schema.*
        """)

    st.markdown("---")

    # --- Status Message ---
    if st.session_state.publications_data is None and st.session_state.patents_data is None:
        st.info(
            "👉 **Ready to begin?** Upload your publication or patent data from the **Data Upload** page "
            "to start a scientometric and technology foresight analysis."
        )
    else:
        st.success(
            "✅ **Data successfully loaded!** Navigate through the analysis modules from the sidebar "
            "to explore trends, networks, topics, and foresight insights."
        )
