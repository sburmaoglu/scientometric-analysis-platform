"""Data Upload Page"""

import streamlit as st
import pandas as pd
from utils.data_loader import load_publications_data, load_patents_data, validate_data, get_data_summary
from utils.preprocessing import preprocess_pipeline
from config.settings import UPLOAD_CONFIG

def render():
    """Render data upload page"""

    st.title("📤 Data Upload & Preprocessing")
    st.markdown("Upload publications and patents data for analysis")

    st.markdown("---")

    tab1, tab2 = st.tabs(["📚 Publications", "💡 Patents"])

    # PUBLICATIONS TAB
    with tab1:
        st.header("📚 Publications Data")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("""
            **Supported Formats:** CSV, Excel, JSON, BibTeX, RIS

            **Required Fields:**
            - `title` - Publication title
            - `year` - Publication year

            **Optional Fields:**
            - `author`, `abstract`, `keywords`, `citations`, `doi`, `journal`
            """)

        with col2:
            if st.session_state.publications_data is not None:
                st.success(f"✅ {len(st.session_state.publications_data)} records")
                if st.button("🗑️ Clear Data", key="clear_pub"):
                    st.session_state.publications_data = None
                    st.session_state.processed_publications = None
                    st.session_state.preprocessing_done['publications'] = False
                    st.rerun()

        pub_file = st.file_uploader(
            "Upload Publications",
            type=['csv', 'xlsx', 'json', 'bib', 'ris'],
            key="pub_upload"
        )

        if pub_file:
            try:
                with st.spinner("Loading..."):
                    df = load_publications_data(pub_file)
                    is_valid, msg = validate_data(df, UPLOAD_CONFIG['required_columns']['publications'])

                    if is_valid:
                        st.session_state.publications_data = df
                        st.success(f"✅ Loaded {len(df)} publications!")

                        with st.expander("📋 Preview", expanded=True):
                            st.dataframe(df.head(10), use_container_width=True)

                        with st.expander("📊 Summary"):
                            summary = get_data_summary(df, 'publications')
                            cols = st.columns(3)
                            with cols[0]:
                                st.metric("Records", summary['total_records'])
                            with cols[1]:
                                st.metric("Columns", len(summary['columns']))
                            with cols[2]:
                                if 'year_range' in summary:
                                    st.metric("Years", summary['year_range'])

                        st.markdown("---")
                        st.subheader("🔧 Preprocessing")

                        col1, col2 = st.columns(2)
                        with col1:
                            opts = {
                                'text_cleaning': st.checkbox("Text Cleaning", True),
                                'ner': st.checkbox("Named Entity Recognition", True)
                            }

                        if st.button("🚀 Start Preprocessing", type="primary"):
                            with st.spinner("Processing..."):
                                processed = preprocess_pipeline(df, 'publications', opts)
                                st.session_state.processed_publications = processed
                                st.session_state.preprocessing_done['publications'] = True
                                st.success("✅ Preprocessing complete!")
                    else:
                        st.error(f"❌ {msg}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    # PATENTS TAB
    with tab2:
        st.header("💡 Patents Data")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("""
            **Supported Formats:** CSV, Excel, JSON, XML

            **Required Fields:**
            - `title` - Patent title
            - `application_date` - Application date

            **Optional Fields:**
            - `inventor`, `assignee`, `abstract`, `claims`, `ipc_class`
            """)

        with col2:
            if st.session_state.patents_data is not None:
                st.success(f"✅ {len(st.session_state.patents_data)} records")
                if st.button("🗑️ Clear Data", key="clear_pat"):
                    st.session_state.patents_data = None
                    st.session_state.processed_patents = None
                    st.session_state.preprocessing_done['patents'] = False
                    st.rerun()

        pat_file = st.file_uploader(
            "Upload Patents",
            type=['csv', 'xlsx', 'json', 'xml'],
            key="pat_upload"
        )

        if pat_file:
            try:
                with st.spinner("Loading..."):
                    df = load_patents_data(pat_file)
                    is_valid, msg = validate_data(df, UPLOAD_CONFIG['required_columns']['patents'])

                    if is_valid:
                        st.session_state.patents_data = df
                        st.success(f"✅ Loaded {len(df)} patents!")

                        with st.expander("📋 Preview", expanded=True):
                            st.dataframe(df.head(10), use_container_width=True)

                        with st.expander("📊 Summary"):
                            summary = get_data_summary(df, 'patents')
                            cols = st.columns(3)
                            with cols[0]:
                                st.metric("Records", summary['total_records'])
                            with cols[1]:
                                st.metric("Columns", len(summary['columns']))

                        st.markdown("---")
                        st.subheader("🔧 Preprocessing")

                        opts = {'text_cleaning': st.checkbox("Text Cleaning", True, key="pat_clean")}

                        if st.button("🚀 Start Preprocessing", type="primary", key="pat_process"):
                            with st.spinner("Processing..."):
                                processed = preprocess_pipeline(df, 'patents', opts)
                                st.session_state.processed_patents = processed
                                st.session_state.preprocessing_done['patents'] = True
                                st.success("✅ Preprocessing complete!")
                    else:
                        st.error(f"❌ {msg}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    st.markdown("---")
    st.subheader("📊 Overall Status")

    col1, col2 = st.columns(2)
    with col1:
        if st.session_state.publications_data is not None:
            status = "✅ Loaded"
            if st.session_state.preprocessing_done['publications']:
                status += " & Processed"
            st.success(f"**Publications:** {status}")
        else:
            st.info("**Publications:** Not loaded")

    with col2:
        if st.session_state.patents_data is not None:
            status = "✅ Loaded"
            if st.session_state.preprocessing_done['patents']:
                status += " & Processed"
            st.success(f"**Patents:** {status}")
        else:
            st.info("**Patents:** Not loaded")
