"""Statistical Visualizations"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

def create_distribution_plot(data: pd.Series, title: str) -> go.Figure:
    """Create distribution plot"""

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=data,
        nbinsx=30,
        marker=dict(
            color='#667eea',
            line=dict(color='white', width=1)
        ),
        opacity=0.75
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Value",
        yaxis_title="Frequency",
        template='plotly_white'
    )

    return fig

def create_boxplot(df: pd.DataFrame, x_col: str, y_col: str, title: str) -> go.Figure:
    """Create boxplot"""

    fig = px.box(df, x=x_col, y=y_col, title=title)
    fig.update_traces(marker_color='#667eea')
    fig.update_layout(template='plotly_white')

    return fig
