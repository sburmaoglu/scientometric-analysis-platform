"""Home Page"""

import streamlit as st
from datetime import datetime

def render():
    """Render home page"""

    st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1 style='font-size: 3rem; margin-bottom: 1rem;'>
            🔬 ScientoMETRICS
        </h1>
        <p style='font-size: 1.3rem; color: #666; margin-bottom: 2rem;'>
            Advanced Scientometric Analysis Platform
        </p>
        <p style='font-size: 1rem; color: #999;'>
            Designed for Academic Research with Rigorous Statistical Methods
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "📚 Publications",
            len(st.session_state.publications_data) if st.session_state.publications_data is not None else 0,
            delta="Loaded" if st.session_state.publications_data is not None else "Not loaded"
        )

    with col2:
        st.metric(
            "💡 Patents",
            len(st.session_state.patents_data) if st.session_state.patents_data is not None else 0,
            delta="Loaded" if st.session_state.patents_data is not None else "Not loaded"
        )

    with col3:
        st.metric(
            "📊 Analyses",
            len(st.session_state.analysis_cache),
            delta="Cached"
        )

    with col4:
        st.metric(
            "🕒 Session",
            datetime.now().strftime("%H:%M"),
            delta=datetime.now().strftime("%Y-%m-%d")
        )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### 🎯 Key Features

        ✨ **Publication-Ready Analysis**
        - Rigorous statistical testing
        - Complete methodology documentation
        - APA/MLA citation formats

        📊 **Advanced Visualizations**
        - Interactive Plotly charts
        - Network analysis graphs
        - Temporal trend analysis

        🔧 **Smart Preprocessing**
        - Named Entity Recognition
        - Keyword extraction
        - Text normalization
        """)

    with col2:
        st.markdown("""
        ### 🚀 Getting Started

        **1. Upload Data**
        Navigate to "Data Upload" and upload your files

        **2. Preprocess**
        Apply NLP and cleaning operations

        **3. Analyze**
        Choose from multiple analysis modules

        **4. Export**
        Generate publication-ready reports

        ---

        💡 **Tip:** All analyses include statistical validation
        suitable for peer-reviewed publications.
        """)

    st.markdown("---")

    if st.session_state.publications_data is None and st.session_state.patents_data is None:
        st.info("""
        👉 **Ready to start?** Upload your data in the **Data Upload** page!

        We support multiple formats: CSV, Excel, JSON, BibTeX, RIS, and XML
        """)
    else:
        st.success("""
        ✅ **Data loaded!** You can now:
        - Explore analysis modules in the sidebar
        - View data statistics
        - Generate visualizations
        - Export results
        """)

    with st.expander("📖 About This Platform"):
        st.markdown("""
        **Scientometrics** is a comprehensive platform for scientometric analysis
        designed specifically for researchers who need rigorous, publication-ready results.

        **Statistical Rigor:**
        - All hypothesis tests include p-values, confidence intervals, and effect sizes
        - Assumption validation (normality, homogeneity)
        - Multiple testing corrections (Bonferroni, FDR)
        - Power analysis

        **Citation:**
```
        Serhat Burmaoglu. (2026). Scientometrics: Advanced Scientometric Analysis Platform
        (Version 1.0.0) [Software]. GitHub.
        https://github.com/yourusername/scientometric-analysis-platform
```

        **License:** MIT License
        """)
