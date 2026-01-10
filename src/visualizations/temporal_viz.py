"""Temporal Visualizations"""

import plotly.graph_objects as go
import pandas as pd

def create_timeline_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str) -> go.Figure:
    """Create timeline visualization"""

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df[x_col],
        y=df[y_col],
        mode='lines+markers',
        marker=dict(size=8, color='#667eea'),
        line=dict(width=2, color='#764ba2')
    ))

    fig.update_layout(
        title=title,
        xaxis_title=x_col.title(),
        yaxis_title=y_col.title(),
        template='plotly_white',
        hovermode='x unified'
    )

    return fig
