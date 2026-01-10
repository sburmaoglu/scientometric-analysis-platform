"""Data Upload Page"""

import streamlit as st
from utils.data_loader import load_publications_data, load_patents_data, validate_data, get_data_summary
from utils.preprocessing import preprocess_pipeline
from config.settings import UPLOAD_CONFIG

def render():
    """Render data upload page"""
    
    st.title("📤 Data Upload & Preprocessing")
    st.markdown("Upload your publications and patents data for analysis")
    
    st.markdown("---")
    
    # Create tabs for publications and patents
    tab1, tab2 = st.tabs(["📚 Publications", "💡 Patents"])
    
    # ==================== PUBLICATIONS TAB ====================
    with tab1:
        st.header("📚 Publications Data")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            **Supported Formats:** CSV, Excel, JSON, BibTeX, RIS
            
            **Required Fields:**
            - `Article Title` - Publication title
            - `Year` - Publication year
            
            **Optional Fields:**
            - `Authors` - Author names (semicolon separated)
            - `Abstract` - Publication abstract
            - `Author Keywords` - Keywords (semicolon separated)
            - `Cited by` - Citation count
            - `DOI` - Digital Object Identifier
            - `Source Title` - Journal/Conference name
            - `Volume`, `Issue`, `Page Start`, `Page End`
            
            **Example Format:** Use data exported from Scopus, Web of Science, or similar databases
            """)
        
        with col2:
            if st.session_state.publications_data is not None:
                st.success(f"✅ {len(st.session_state.publications_data)} records")
                if st.button("🗑️ Clear Data", key="clear_pub"):
                    st.session_state.publications_data = None
                    st.session_state.processed_publications = None
                    st.session_state.preprocessing_done['publications'] = False
                    st.rerun()
        
        st.markdown("---")
        
        # File uploader
        pub_file = st.file_uploader(
            "Upload Publications Data",
            type=['csv', 'xlsx', 'json', 'bib', 'ris'],
            key="pub_upload",
            help="Upload your publications data file"
        )
        
        if pub_file:
            try:
                with st.spinner("Loading publications data..."):
                    df = load_publications_data(pub_file)
                    
                    # Validate data
                    is_valid, msg = validate_data(df, UPLOAD_CONFIG['required_columns']['publications'])
                    
                    if is_valid:
                        st.session_state.publications_data = df
                        st.success(f"✅ Successfully loaded {len(df)} publications!")
                        
                        # Show preview
                        with st.expander("📋 Data Preview", expanded=True):
                            st.dataframe(df.head(10), use_container_width=True)
                        
                        # Show summary
                        with st.expander("📊 Data Summary"):
                            summary = get_data_summary(df, 'publications')
                            
                            cols = st.columns(4)
                            with cols[0]:
                                st.metric("Total Records", f"{summary['total_records']:,}")
                            with cols[1]:
                                if 'year_range' in summary:
                                    st.metric("Year Range", summary['year_range'])
                            with cols[2]:
                                if 'unique_authors' in summary:
                                    st.metric("Unique Authors", f"{summary['unique_authors']:,}")
                            with cols[3]:
                                if 'total_citations' in summary:
                                    st.metric("Total Citations", f"{summary['total_citations']:,}")
                            
                            st.markdown("**Columns Found:**")
                            st.write(", ".join(summary['columns']))
                            
                            # Show missing values
                            missing_df = df.isnull().sum()
                            missing_df = missing_df[missing_df > 0]
                            if len(missing_df) > 0:
                                st.markdown("**Missing Values:**")
                                st.dataframe(missing_df, use_container_width=True)
                        
                        # Preprocessing options
                        st.markdown("---")
                        st.subheader("🔧 Preprocessing Options")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            opts = {
                                'text_cleaning': st.checkbox("Text Cleaning & Normalization", True, key="pub_clean"),
                                'ner': st.checkbox("Named Entity Recognition", True, key="pub_ner"),
                                'keyword_extraction': st.checkbox("Keyword Extraction", False, key="pub_keywords")
                            }
                        
                        with col2:
                            st.info("""
                            **Preprocessing will:**
                            - Clean and normalize text
                            - Extract entities (organizations, locations)
                            - Generate keywords from abstracts
                            """)
                        
                        if st.button("🚀 Start Preprocessing", type="primary", key="pub_preprocess_btn"):
                            with st.spinner("Processing... This may take a few minutes."):
                                try:
                                    processed = preprocess_pipeline(df, 'publications', opts)
                                    st.session_state.processed_publications = processed
                                    st.session_state.preprocessing_done['publications'] = True
                                    st.success("✅ Preprocessing completed successfully!")
                                    
                                    # Show what was added
                                    new_cols = set(processed.columns) - set(df.columns)
                                    if new_cols:
                                        st.info(f"**New columns added:** {', '.join(new_cols)}")
                                
                                except Exception as e:
                                    st.error(f"❌ Error during preprocessing: {str(e)}")
                    
                    else:
                        st.error(f"❌ Data validation failed: {msg}")
                        st.info("Please ensure your file contains the required columns")
            
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
                st.info("Please check that your file is in the correct format")
    
    # ==================== PATENTS TAB ====================
    with tab2:
        st.header("💡 Patents Data")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            **Supported Formats:** CSV, Excel, JSON, XML
            
            **Required Fields:**
            - `Title` - Patent title
            - `Publication Date` - Publication date
            
            **Optional Fields:**
            - `Inventors` - Inventor names (semicolon separated)
            - `Applicants` - Applicant/assignee organizations
            - `Abstract` - Patent abstract
            - `Jurisdiction` - Patent jurisdiction (US, EP, etc.)
            - `Simple Family Size` - Patent family size
            - `IPC Classifications` - Technology classifications
            
            **Example Format:** Use data exported from lens.org or similar patent databases
            """)
        
        with col2:
            if st.session_state.patents_data is not None:
                st.success(f"✅ {len(st.session_state.patents_data)} records")
                if st.button("🗑️ Clear Data", key="clear_pat"):
                    st.session_state.patents_data = None
                    st.session_state.processed_patents = None
                    st.session_state.preprocessing_done['patents'] = False
                    st.rerun()
        
        st.markdown("---")
        
        # File uploader
        pat_file = st.file_uploader(
            "Upload Patents Data",
            type=['csv', 'xlsx', 'json', 'xml'],
            key="pat_upload",
            help="Upload your patents data file"
        )
        
        if pat_file:
            try:
                with st.spinner("Loading patents data..."):
                    df = load_patents_data(pat_file)
                    
                    # Validate data
                    is_valid, msg = validate_data(df, UPLOAD_CONFIG['required_columns']['patents'])
                    
                    if is_valid:
                        st.session_state.patents_data = df
                        st.success(f"✅ Successfully loaded {len(df)} patents!")
                        
                        # Show preview
                        with st.expander("📋 Data Preview", expanded=True):
                            st.dataframe(df.head(10), use_container_width=True)
                        
                        # Show summary
                        with st.expander("📊 Data Summary"):
                            summary = get_data_summary(df, 'patents')
                            
                            cols = st.columns(4)
                            with cols[0]:
                                st.metric("Total Records", f"{summary['total_records']:,}")
                            with cols[1]:
                                if 'year_range' in summary:
                                    st.metric("Year Range", summary['year_range'])
                            with cols[2]:
                                if 'unique_inventors' in summary:
                                    st.metric("Unique Inventors", f"{summary['unique_inventors']:,}")
                            with cols[3]:
                                if 'unique_assignees' in summary:
                                    st.metric("Unique Assignees", f"{summary['unique_assignees']:,}")
                            
                            st.markdown("**Columns Found:**")
                            st.write(", ".join(summary['columns']))
                            
                            # Show missing values
                            missing_df = df.isnull().sum()
                            missing_df = missing_df[missing_df > 0]
                            if len(missing_df) > 0:
                                st.markdown("**Missing Values:**")
                                st.dataframe(missing_df, use_container_width=True)
                        
                        # Preprocessing options
                        st.markdown("---")
                        st.subheader("🔧 Preprocessing Options")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            opts = {
                                'text_cleaning': st.checkbox("Text Cleaning & Normalization", True, key="pat_clean"),
                                'ner': st.checkbox("Named Entity Recognition", True, key="pat_ner")
                            }
                        
                        with col2:
                            st.info("""
                            **Preprocessing will:**
                            - Clean and normalize text
                            - Extract entities from abstracts
                            - Standardize organization names
                            """)
                        
                        if st.button("🚀 Start Preprocessing", type="primary", key="pat_preprocess_btn"):
                            with st.spinner("Processing... This may take a few minutes."):
                                try:
                                    processed = preprocess_pipeline(df, 'patents', opts)
                                    st.session_state.processed_patents = processed
                                    st.session_state.preprocessing_done['patents'] = True
                                    st.success("✅ Preprocessing completed successfully!")
                                    
                                    # Show what was added
                                    new_cols = set(processed.columns) - set(df.columns)
                                    if new_cols:
                                        st.info(f"**New columns added:** {', '.join(new_cols)}")
                                
                                except Exception as e:
                                    st.error(f"❌ Error during preprocessing: {str(e)}")
                    
                    else:
                        st.error(f"❌ Data validation failed: {msg}")
                        st.info("Please ensure your file contains the required columns")
            
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
                st.info("Please check that your file is in the correct format")
    
    # ==================== OVERALL STATUS ====================
    st.markdown("---")
    st.subheader("📊 Overall Data Status")
    
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
