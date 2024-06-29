from graph_types import ParameterGraph
import networkx as nx
import matplotlib.pyplot as plt
import random
import igraph as ig
# chart studip plotly
import chart_studio.plotly as py
import plotly.graph_objs as go
import numpy as np


def draw_nx_graph(graph: ParameterGraph):
    '''
    Draw the global graph

    Args:
    - graph (MultiDiGraph): Global graph
    '''
    pos = nx.multipartite_layout(graph)
    nx.draw_networkx_nodes(graph, pos)
    nx.draw_networkx_labels(graph, pos)
    i = 0
    for u, v, data in graph.edges(data=True):
        rad = 0.05*i % 1.0 * random.choice([-1, 1])
        nx.draw_networkx_edges(graph, pos, edgelist=[(u, v)], connectionstyle=f'arc3,rad={rad}')
        i += 1
    plt.show()

def draw_3d_graph(graph: ParameterGraph):
    num_links = graph.number_of_edges()
    num_nodes = graph.number_of_nodes()
    print("Number of links: ", num_links)
    print("Number of nodes: ", num_nodes)

    # Create a MultiDiGraph to handle multi-edges
    G = ig.Graph(directed=True)
    G.add_vertices(graph.number_of_nodes())

    # Add edges with their multiplicities
    edge_counts = {}
    for u, v in graph.edges():
        if (u, v) in edge_counts:
            edge_counts[(u, v)] += 1
        else:
            edge_counts[(u, v)] = 1
        G.add_edge(u, v)

    labels = []
    group = []
    for node in graph.nodes(data=True):
        labels.append(str(node[1]['node_obj']))
        group.append(node[1]['subset'])

    layt = G.layout('kk', dim=3)
    
    Xn = [layt[k][0] for k in range(num_nodes)]
    Yn = [layt[k][1] for k in range(num_nodes)]
    Zn = [layt[k][2] for k in range(num_nodes)]
    Xe = []
    Ye = []
    Ze = []

    # Create curved edges for multi-edges
    for (u, v), count in edge_counts.items():
        if count == 1:
            # Single edge
            Xe += [layt[u][0], layt[v][0], None]
            Ye += [layt[u][1], layt[v][1], None]
            Ze += [layt[u][2], layt[v][2], None]
        else:
            # Multi-edge: create curved lines
            for i in range(count):
                # Calculate control point for Bezier curve
                mid_x = (layt[u][0] + layt[v][0]) / 2
                mid_y = (layt[u][1] + layt[v][1]) / 2
                mid_z = (layt[u][2] + layt[v][2]) / 2
                
                # Offset the control point
                offset = 0.1 * (i + 1)  # Adjust this value to change curve intensity
                ctrl_x = mid_x + offset * (layt[v][1] - layt[u][1])
                ctrl_y = mid_y - offset * (layt[v][0] - layt[u][0])
                ctrl_z = mid_z + offset
                
                # Create Bezier curve points
                t = np.linspace(0, 1, 20)
                x = (1-t)**2 * layt[u][0] + 2*(1-t)*t * ctrl_x + t**2 * layt[v][0]
                y = (1-t)**2 * layt[u][1] + 2*(1-t)*t * ctrl_y + t**2 * layt[v][1]
                z = (1-t)**2 * layt[u][2] + 2*(1-t)*t * ctrl_z + t**2 * layt[v][2]
                
                Xe += list(x) + [None]
                Ye += list(y) + [None]
                Ze += list(z) + [None]

    print("adding traces")
    trace1=go.Scatter3d(x=Xe,
               y=Ye,
               z=Ze,
               mode='lines',
               line=dict(color='rgb(125,125,125)', width=1),
               hoverinfo='none'
               )
    print("Made trace 1")
    trace2=go.Scatter3d(x=Xn,
                y=Yn,
                z=Zn,
                mode='markers',
                name='actors',
                marker=dict(symbol='circle',
                                size=6,
                                color=group,
                                colorscale='Viridis',
                                line=dict(color='rgb(50,50,50)', width=0.5)
                                ),
                text=labels,
                hoverinfo='text'
                )
    print("Made trace 2")
    axis=dict(showbackground=False,
            showline=False,
            zeroline=False,
            showgrid=False,
            showticklabels=False,
            title=''
            )
    print("Made axis")
    layout = go.Layout(
         title="",
         width=1000,
         height=1000,
         showlegend=False,
         scene=dict(
             xaxis=dict(axis),
             yaxis=dict(axis),
             zaxis=dict(axis),
        ),
     margin=dict(
        t=100
    ),
    hovermode='closest',
    
        annotations=[])
    print("Made layout")

    data=[trace1, trace2]
    fig=go.Figure(data=data, layout=layout)
    print("Made figure")

    #py.iplot(fig, filename='Les-Miserables')
    fig.show()
    print("Done")



def draw_graph(graph: ParameterGraph, dim: str = '2d'):
    '''
    Draw the global graph

    Args:
    - graph (MultiDiGraph): Global graph
    '''
    if dim == '2d':
        draw_nx_graph(graph)
    elif dim == '3d':
        draw_3d_graph(graph)