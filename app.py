"""
Advanced Scientometric Analysis Platform
Main Application Entry Point
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

from config.settings import PAGE_CONFIG, THEME_CONFIG, CUSTOM_CSS
from utils.session_state import initialize_session_state
from core.module_loader import load_available_modules, get_module_instance

st.set_page_config(**PAGE_CONFIG)
initialize_session_state()
st.markdown(THEME_CONFIG, unsafe_allow_html=True)
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def main():
    """Main application controller"""

    available_modules = load_available_modules()

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

        core_pages = [
            ("🏠", "Home", "home"),
            ("📤", "Data Upload", "data_upload"),
        ]

        analysis_pages = []
        for module_name, module_info in sorted(available_modules.items()):
            if module_info.get('enabled', True):
                analysis_pages.append((
                    module_info.get('icon', '📊'),
                    module_info.get('display_name', module_name),
                    module_name
                ))

        all_pages = core_pages + analysis_pages
        page_options = [f"{icon} {name}" for icon, name, _ in all_pages]
        page_keys = {f"{icon} {name}": key for icon, name, key in all_pages}

        selected_page = st.radio("Select Page", page_options, label_visibility="collapsed")
        selected_key = page_keys[selected_page]

        st.markdown("---")
        st.markdown("### 📊 Data Status")

        col1, col2 = st.columns(2)

        with col1:
            if st.session_state.get('publications_data') is not None:
                pub_count = len(st.session_state.publications_data)
                processed = st.session_state.preprocessing_done.get('publications', False)
                st.metric("📚 Publications", f"{pub_count:,}",
                         delta="✓ Processed" if processed else "Raw")
            else:
                st.info("📚 No Publications")

        with col2:
            if st.session_state.get('patents_data') is not None:
                pat_count = len(st.session_state.patents_data)
                processed = st.session_state.preprocessing_done.get('patents', False)
                st.metric("💡 Patents", f"{pat_count:,}",
                         delta="✓ Processed" if processed else "Raw")
            else:
                st.info("💡 No Patents")

        st.markdown("---")

        with st.expander("⚙️ Settings", expanded=False):
            st.session_state.theme = st.selectbox("Color Theme",
                ["Professional", "Dark", "Academic", "Vibrant"], index=0)
            st.session_state.chart_style = st.selectbox("Chart Style",
                ["plotly_white", "plotly", "plotly_dark"], index=0)
            st.session_state.export_dpi = st.selectbox("Export DPI",
                [300, 600, 1200], index=0)
            st.session_state.statistical_alpha = st.slider("Significance (α)",
                0.01, 0.10, 0.05, 0.01)

        with st.expander("ℹ️ Help", expanded=False):
            st.markdown("""
            **Quick Start:**
            1. Upload data
            2. Preprocess
            3. Analyze
            4. Export

            All analyses include rigorous statistical validation.
            """)

        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; font-size: 0.7rem; color: #999;'>
            v1.0.0 | Built for Researchers
        </div>
        """, unsafe_allow_html=True)

    if selected_key in ['home', 'data_upload']:
        if selected_key == 'home':
            from pages import home
            home.render()
        elif selected_key == 'data_upload':
            from pages import data_upload
            data_upload.render()
    else:
        module_instance = get_module_instance(selected_key)
        if module_instance:
            try:
                module_instance.render()
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.error(f"Module '{selected_key}' not found")

    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 1rem;'>
        <p style='color: #666; font-size: 0.85rem;'>
            <b>ScientoMetrics</b> | Designed for Academic Research
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
