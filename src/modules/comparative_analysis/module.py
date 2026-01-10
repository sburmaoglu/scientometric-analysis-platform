"""Comparative Analysis Module"""

from core.base_module import BaseModule
import streamlit as st
import pandas as pd

class ComparativeAnalysisModule(BaseModule):
    """Module for comparative analysis between publications and patents"""
    
    def render(self):
        st.title("🔄 Comparative Analysis")
        st.markdown("Compare and integrate publications and patents data")
        
        # Check if both datasets are available
        has_pubs = self.check_data_availability('publications')
        has_pats = self.check_data_availability('patents')
        
        if not has_pubs and not has_pats:
            self.show_data_required_message()
            return
        
        if not has_pubs:
            st.warning("⚠️ Publications data not loaded. Upload publications for full comparative analysis.")
        
        if not has_pats:
            st.warning("⚠️ Patents data not loaded. Upload patents for full comparative analysis.")
        
        st.markdown("---")
        
        # If both are available
        if has_pubs and has_pats:
            pubs_df = st.session_state.publications_data
            pats_df = st.session_state.patents_data
            
            st.subheader("📊 Dataset Comparison")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Publications", f"{len(pubs_df):,}")
                st.metric("Patents", f"{len(pats_df):,}")
            
            with col2:
                if 'year' in pubs_df.columns:
                    pub_years = f"{pubs_df['year'].min():.0f}-{pubs_df['year'].max():.0f}"
                    st.metric("Publications Years", pub_years)
                if 'year' in pats_df.columns:
                    pat_years = f"{pats_df['year'].min():.0f}-{pats_df['year'].max():.0f}"
                    st.metric("Patents Years", pat_years)
            
            with col3:
                if 'citations' in pubs_df.columns:
                    st.metric("Pub Citations", f"{pubs_df['citations'].sum():,.0f}")
                if 'forward_citations' in pats_df.columns:
                    st.metric("Patent Citations", f"{pats_df['forward_citations'].sum():,.0f}")
            
            st.markdown("---")
            
            # Temporal comparison
            st.subheader("📈 Temporal Trends Comparison")
            
            if 'year' in pubs_df.columns and 'year' in pats_df.columns:
                pub_yearly = pubs_df.groupby('year').size()
                pat_yearly = pats_df.groupby('year').size()
                
                import plotly.graph_objects as go
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=pub_yearly.index, y=pub_yearly.values,
                                        mode='lines+markers', name='Publications',
                                        line=dict(color='#3498db', width=3)))
                fig.add_trace(go.Scatter(x=pat_yearly.index, y=pat_yearly.values,
                                        mode='lines+markers', name='Patents',
                                        line=dict(color='#e74c3c', width=3)))
                
                fig.update_layout(
                    title="Publications vs Patents Over Time",
                    xaxis_title="Year",
                    yaxis_title="Count",
                    template='plotly_white',
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            st.info("🚧 Additional comparative analyses (knowledge transfer, cross-domain influence, etc.) coming soon!")
        
        else:
            st.info("""
            ### 📊 Available Analyses
            
            Comparative analysis requires both publications and patents data.
            
            **Upload both datasets to enable:**
            - Knowledge transfer time lag analysis
            - Cross-domain influence metrics
            - Actor & institution comparison
            - Technology lifecycle comparison
            - Innovation ecosystem mapping
            """)
