"""Base Module Class"""

from abc import ABC, abstractmethod
import streamlit as st

class BaseModule(ABC):
    """Base class for all analysis modules"""

    def __init__(self):
        self.name = self.__class__.__name__

    @abstractmethod
    def render(self):
        """Render the module interface"""
        pass

    def check_data_availability(self, data_type: str = None):
        """Check if required data is available"""
        if data_type == 'publications':
            return st.session_state.get('publications_data') is not None
        elif data_type == 'patents':
            return st.session_state.get('patents_data') is not None
        else:
            return (st.session_state.get('publications_data') is not None or
                   st.session_state.get('patents_data') is not None)

    def show_data_required_message(self):
        """Show message when data is not available"""
        st.warning("⚠️ Please upload data first in the Data Upload page")

        with st.expander("📤 Quick Upload Guide"):
            st.markdown("""
            1. Go to **Data Upload** in the sidebar
            2. Upload your publications or patents data
            3. Complete preprocessing
            4. Return to this analysis
            """)
