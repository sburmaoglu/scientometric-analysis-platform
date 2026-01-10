"""Temporal Analysis Page"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

def render():
    """Render temporal analysis page"""
    
    st.title("📈 Temporal Analysis")
    st.markdown("Advanced time-series analysis and trend forecasting")
    
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
        data_name = "Publications"
    else:
        if st.session_state.patents_data is None:
            st.warning("⚠️ Please upload patents data first")
            return
        df = st.session_state.patents_data
        data_name = "Patents"
    
    if 'year' not in df.columns:
        st.warning("Year column not available")
        return
    
    st.markdown("---")
    
    # Time series plot
    st.subheader(f"📈 {data_name} Over Time")
    
    yearly = df.groupby('year').size().reset_index(name='count')
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=yearly['year'],
        y=yearly['count'],
        mode='lines+markers',
        name=data_name,
        marker=dict(size=10, color='#667eea'),
        line=dict(width=3)
    ))
    
    fig.update_layout(
        title=f"{data_name} per Year",
        xaxis_title="Year",
        yaxis_title="Count",
        template='plotly_white',
        height=450
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Growth Analysis
    st.markdown("---")
    st.subheader("📊 Growth Analysis")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if len(yearly) > 1:
            growth = ((yearly['count'].iloc[-1] - yearly['count'].iloc[0]) / yearly['count'].iloc[0] * 100)
            st.metric("Total Growth", f"{growth:+.1f}%")
    
    with col2:
        avg_growth = yearly['count'].pct_change().mean() * 100
        st.metric("Avg Annual Growth", f"{avg_growth:.1f}%")
    
    with col3:
        st.metric("Peak Year", f"{yearly.loc[yearly['count'].idxmax(), 'year']:.0f}")
    
    with col4:
        st.metric("Peak Count", f"{yearly['count'].max():.0f}")
    
    # Moving Average
    st.markdown("---")
    st.subheader("📉 Moving Average Analysis")
    
    window = st.slider("Moving Average Window (years)", 2, 5, 3)
    
    yearly['moving_avg'] = yearly['count'].rolling(window=window, center=True).mean()
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=yearly['year'],
        y=yearly['count'],
        mode='lines+markers',
        name='Actual',
        line=dict(color='lightgray', width=2),
        marker=dict(size=6)
    ))
    
    fig.add_trace(go.Scatter(
        x=yearly['year'],
        y=yearly['moving_avg'],
        mode='lines',
        name=f'{window}-Year Moving Average',
        line=dict(color='#e74c3c', width=3)
    ))
    
    fig.update_layout(
        title=f"{data_name} with {window}-Year Moving Average",
        xaxis_title="Year",
        yaxis_title="Count",
        template='plotly_white',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)