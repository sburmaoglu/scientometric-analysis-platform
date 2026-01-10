"""Geographic Analysis Page"""

import streamlit as st
import pandas as pd
import plotly.express as px

def render():
    """Render geographic analysis page"""
    
    st.title("🗺️ Geographic Analysis")
    st.markdown("Analyze geographic distribution and international collaboration")
    
    if st.session_state.publications_data is None and st.session_state.patents_data is None:
        st.warning("⚠️ Please upload data first")
        return
    
    # Choose dataset
    dataset_option = st.radio(
        "Select Dataset",
        ["Publications", "Patents"],
        horizontal=True
    )
    
    if dataset_option == "Publications":
        if st.session_state.publications_data is None:
            st.warning("⚠️ Please upload publications data first")
            return
        df = st.session_state.publications_data
        geo_col = 'country'
    else:
        if st.session_state.patents_data is None:
            st.warning("⚠️ Please upload patents data first")
            return
        df = st.session_state.patents_data
        geo_col = 'jurisdiction'
    
    if geo_col not in df.columns:
        st.warning(f"{geo_col} column not available")
        return
    
    st.markdown("---")
    
    # Geographic distribution
    st.subheader("🌍 Geographic Distribution")
    
    geo_counts = df[geo_col].value_counts().head(20)
    
    geo_df = pd.DataFrame({
        'Location': geo_counts.index,
        'Count': geo_counts.values
    })
    
    # Bar chart
    fig = px.bar(
        geo_df.head(15),
        x='Location',
        y='Count',
        title=f"Top 15 {geo_col.title()}s",
        color='Count',
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        template='plotly_white',
        height=450,
        xaxis_tickangle=-45
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Locations", len(df[geo_col].unique()))
    
    with col2:
        st.metric("Top Location", geo_df.iloc[0]['Location'])
    
    with col3:
        top_pct = (geo_df.iloc[0]['Count'] / len(df)) * 100
        st.metric("Top Share", f"{top_pct:.1f}%")
    
    # Full table
    st.markdown("---")
    st.subheader("📊 Complete Geographic Distribution")
    
    st.dataframe(geo_df, use_container_width=True, hide_index=True)