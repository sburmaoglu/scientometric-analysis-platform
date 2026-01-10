"""Geospatial Visualizations"""

import plotly.express as px
import pandas as pd

def create_choropleth_map(df: pd.DataFrame, location_col: str, value_col: str, title: str) -> px.choropleth:
    """Create choropleth map"""

    fig = px.choropleth(
        df,
        locations=location_col,
        locationmode='country names',
        color=value_col,
        hover_name=location_col,
        title=title,
        color_continuous_scale='Blues'
    )

    fig.update_layout(
        geo=dict(showframe=False, showcoastlines=True)
    )

    return fig
