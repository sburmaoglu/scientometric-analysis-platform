"""Comparative Visualizations"""

import plotly.graph_objects as go
import pandas as pd

def create_comparison_bar_chart(df: pd.DataFrame, categories: list, values1: list, values2: list,
                                 label1: str, label2: str, title: str) -> go.Figure:
    """Create comparison bar chart"""

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=categories,
        y=values1,
        name=label1,
        marker_color='#667eea'
    ))

    fig.add_trace(go.Bar(
        x=categories,
        y=values2,
        name=label2,
        marker_color='#764ba2'
    ))

    fig.update_layout(
        title=title,
        barmode='group',
        template='plotly_white',
        xaxis_title="Category",
        yaxis_title="Value"
    )

    return fig
