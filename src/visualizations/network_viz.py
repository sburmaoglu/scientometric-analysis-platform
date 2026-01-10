"""Network Visualizations"""

import plotly.graph_objects as go
import networkx as nx

def create_network_graph(G: nx.Graph, title: str = "Network") -> go.Figure:
    """Create interactive network visualization"""

    pos = nx.spring_layout(G, k=0.5, iterations=50)

    edge_trace = go.Scatter(
        x=[],
        y=[],
        mode='lines',
        line=dict(width=0.5, color='#888'),
        hoverinfo='none'
    )

    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace['x'] += tuple([x0, x1, None])
        edge_trace['y'] += tuple([y0, y1, None])

    node_trace = go.Scatter(
        x=[],
        y=[],
        mode='markers+text',
        hoverinfo='text',
        marker=dict(
            size=10,
            color='#667eea',
            line=dict(width=2, color='white')
        )
    )

    for node in G.nodes():
        x, y = pos[node]
        node_trace['x'] += tuple([x])
        node_trace['y'] += tuple([y])

    node_trace['text'] = list(G.nodes())

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=title,
            showlegend=False,
            hovermode='closest',
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
    )

    return fig
