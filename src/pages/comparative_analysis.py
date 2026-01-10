"""Comparative Analysis Page"""

import streamlit as st
import plotly.graph_objects as go

def render():
    """Render comparative analysis page"""
    
    st.title("🔄 Comparative Analysis")
    st.markdown("Compare publications and patents data side-by-side")
    
    has_pubs = st.session_state.publications_data is not None
    has_pats = st.session_state.patents_data is not None
    
    if not has_pubs or not has_pats:
        st.warning("⚠️ Both publications and patents data required for comparison")
        
        col1, col2 = st.columns(2)
        with col1:
            if not has_pubs:
                st.error("❌ Publications data not loaded")
            else:
                st.success("✅ Publications data loaded")
        
        with col2:
            if not has_pats:
                st.error("❌ Patents data not loaded")
            else:
                st.success("✅ Patents data loaded")
        
        return
    
    pubs_df = st.session_state.publications_data
    pats_df = st.session_state.patents_data
    
    st.markdown("---")
    
    # Overview Comparison
    st.subheader("📊 Dataset Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📚 Publications")
        st.metric("Total Records", f"{len(pubs_df):,}")
        
        if 'year' in pubs_df.columns:
            years = pubs_df['year'].dropna()
            if len(years) > 0:
                st.metric("Year Range", f"{int(years.min())}-{int(years.max())}")
        
        if 'citations' in pubs_df.columns:
            total = pubs_df['citations'].sum()
            if pd.notna(total):
                st.metric("Total Citations", f"{int(total):,}")
    
    with col2:
        st.markdown("### 💡 Patents")
        st.metric("Total Records", f"{len(pats_df):,}")
        
        if 'year' in pats_df.columns:
            years = pats_df['year'].dropna()
            if len(years) > 0:
                st.metric("Year Range", f"{int(years.min())}-{int(years.max())}")
        
        if 'forward_citations' in pats_df.columns:
            total = pats_df['forward_citations'].sum()
            if pd.notna(total):
                st.metric("Total Citations", f"{int(total):,}")
    
    st.markdown("---")
    
    # Temporal Comparison
    st.subheader("📈 Temporal Trends Comparison")
    
    if 'year' in pubs_df.columns and 'year' in pats_df.columns:
        pub_yearly = pubs_df.groupby('year').size()
        pat_yearly = pats_df.groupby('year').size()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=pub_yearly.index,
            y=pub_yearly.values,
            mode='lines+markers',
            name='Publications',
            line=dict(color='#3498db', width=3),
            marker=dict(size=8)
        ))
        
        fig.add_trace(go.Scatter(
            x=pat_yearly.index,
            y=pat_yearly.values,
            mode='lines+markers',
            name='Patents',
            line=dict(color='#e74c3c', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title="Publications vs Patents Over Time",
            xaxis_title="Year",
            yaxis_title="Count",
            template='plotly_white',
            height=450,
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Growth comparison
        st.markdown("#### 📊 Growth Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if len(pub_yearly) > 1:
                pub_growth = ((pub_yearly.iloc[-1] - pub_yearly.iloc[0]) / pub_yearly.iloc[0] * 100)
                st.metric("Publications Growth", f"{pub_growth:+.1f}%")
        
        with col2:
            if len(pat_yearly) > 1:
                pat_growth = ((pat_yearly.iloc[-1] - pat_yearly.iloc[0]) / pat_yearly.iloc[0] * 100)
                st.metric("Patents Growth", f"{pat_growth:+.1f}%")
    else:
        st.warning("Year data not available in both datasets")