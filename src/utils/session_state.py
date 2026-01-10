"""Session State Management"""

import streamlit as st
from datetime import datetime

def initialize_session_state():
    """Initialize all session state variables"""

    if 'publications_data' not in st.session_state:
        st.session_state.publications_data = None

    if 'patents_data' not in st.session_state:
        st.session_state.patents_data = None

    if 'processed_publications' not in st.session_state:
        st.session_state.processed_publications = None

    if 'processed_patents' not in st.session_state:
        st.session_state.processed_patents = None

    if 'preprocessing_done' not in st.session_state:
        st.session_state.preprocessing_done = {
            'publications': False,
            'patents': False
        }

    if 'analysis_cache' not in st.session_state:
        st.session_state.analysis_cache = {}

    if 'networks' not in st.session_state:
        st.session_state.networks = {}

    if 'topic_models' not in st.session_state:
        st.session_state.topic_models = {}

    if 'theme' not in st.session_state:
        st.session_state.theme = 'Professional'

    if 'chart_style' not in st.session_state:
        st.session_state.chart_style = 'plotly_white'

    if 'export_dpi' not in st.session_state:
        st.session_state.export_dpi = 300

    if 'statistical_alpha' not in st.session_state:
        st.session_state.statistical_alpha = 0.05

    if 'filters' not in st.session_state:
        st.session_state.filters = {}

    if 'last_update' not in st.session_state:
        st.session_state.last_update = datetime.now()

def update_timestamp():
    """Update last modification timestamp"""
    st.session_state.last_update = datetime.now()
