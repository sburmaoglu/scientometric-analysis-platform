"""
Scientometric Analysis Platform
Main Application Router
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
    """Main application controller"""
    
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
        
        # Navigation menu
        page = st.radio(
   		 "Select Page",
  		  [
       			 "🏠 Home",
      			  "📤 Data Upload",
     			   "📚 Publications Analysis",
     			   "💡 Patents Analysis",
     			   "🔄 Comparative Analysis",
     			   "📈 Temporal Analysis",
     			   "🗺️ Geographic Analysis",
     			   "🔬 Advanced Analytics",      # NEW
     			   "🏷️ Topic Modeling"           # NEW
   		 ],
    		label_visibility="collapsed"
	)

# In the routing section, add:
elif page == "🔬 Advanced Analytics":
    import pages.advanced_analytics as advanced_analytics
    advanced_analytics.render()

elif page == "🏷️ Topic Modeling":
    import pages.topic_modeling as topic_modeling
    topic_modeling.render()

        st.markdown("---")
        st.markdown("### 📊 Data Status")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.session_state.get('publications_data') is not None:
                pub_count = len(st.session_state.publications_data)
                st.metric("📚 Pubs", f"{pub_count:,}")
            else:
                st.info("📚 No Data")
        
        with col2:
            if st.session_state.get('patents_data') is not None:
                pat_count = len(st.session_state.patents_data)
                st.metric("💡 Pats", f"{pat_count:,}")
            else:
                st.info("💡 No Data")
        
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; font-size: 0.7rem; color: #999;'>
            v1.0.0 | Built for Researchers
        </div>
        """, unsafe_allow_html=True)
    
    # Route to pages - Simple imports!
    if page == "🏠 Home":
        from pages import home
        home.render()
    
    elif page == "📤 Data Upload":
        from pages import data_upload
        data_upload.render()
    
    elif page == "📚 Publications Analysis":
        from pages import publications_analysis
        publications_analysis.render()
    
    elif page == "💡 Patents Analysis":
        from pages import patents_analysis
        patents_analysis.render()
    
    elif page == "🔄 Comparative Analysis":
        from pages import comparative_analysis
        comparative_analysis.render()
    
    elif page == "📈 Temporal Analysis":
        from pages import temporal_analysis
        temporal_analysis.render()
    
    elif page == "🗺️ Geographic Analysis":
        from pages import geographic_analysis
        geographic_analysis.render()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 1rem;'>
        <p style='color: #666; font-size: 0.85rem;'>
            <b>ScientoMetrics</b> | Advanced Scientometric Analysis
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()