from .graph_types import ParameterGraph
import networkx as nx
import matplotlib.pyplot as plt
import random
import igraph as ig
# chart studip plotly
import plotly.express as px
import plotly.graph_objs as go
import numpy as np
import plotly.io as pio
import math


def bezier_curve(p0, p1, p2, num_points=20):
    t = np.linspace(0, 1, num_points)
    x = (1-t)**2 * p0[0] + 2*(1-t)*t * p1[0] + t**2 * p2[0]
    y = (1-t)**2 * p0[1] + 2*(1-t)*t * p1[1] + t**2 * p2[1]
    z = (1-t)**2 * p0[2] + 2*(1-t)*t * p1[2] + t**2 * p2[2]
    return x, y, z
def draw_nx_graph(graph: ParameterGraph, title: str):
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
    plt.title(title)
    plt.show()


def draw_3d_graph(graph: nx.Graph, title: str, save_path: str=None, display_inline=False) -> None:
    num_links = graph.number_of_edges()
    num_nodes = graph.number_of_nodes()
    print("Number of links: ", num_links)
    print("Number of nodes: ", num_nodes)

    G = ig.Graph(directed=True)

    labels = []
    node_colors = []
    Xn, Yn, Zn = [], [], []

    node_color_map = {
    'INPUT': '#FFB3BA',
    'LINEAR': '#BAFFC9',
    'CONV': '#BAE1FF',
    'NORM': '#C9BAFF',
    'NON_PARAMETRIC': '#FFDFBA'
    }
    labels = []
    node_colors = []
    Xn, Yn, Zn = [], [], []

    # Group nodes by subset
    nodes_by_subset = {}
    for node_id, node_data in graph.nodes(data=True):
        subset = node_data['subset']
        if subset not in nodes_by_subset:
            nodes_by_subset[subset] = []
        nodes_by_subset[subset].append((node_id, node_data))

    # Sort subsets
    sorted_subsets = sorted(nodes_by_subset.keys())

    for layer, subset in enumerate(sorted_subsets):
        nodes = nodes_by_subset[subset]
        num_nodes = len(nodes)
        
        # Calculate circle radius based on the number of nodes
        radius = max(1, num_nodes / (2 * math.pi))
        
        for i, (node_id, node_data) in enumerate(nodes):
            labels.append(str(node_data['node_obj'].features.node_type))
            node_type = str(node_data['node_obj'].features.node_type)
            node_colors.append(node_color_map[node_type])
            
            # Calculate position on the circle
            angle = 2 * math.pi * i / num_nodes
            x = radius * math.cos(angle)
            y = radius * math.sin(angle)
            z = layer * 1  # Increase vertical separation between layers
            
            Xn.append(x)
            Yn.append(y)
            Zn.append(z)

            G.add_vertex(node_id)

    # Create a mapping from node_id to its index in the position lists
    node_to_index = {node_id: i for i, (node_id, _) in enumerate(graph.nodes(data=True))}

    # Create a color map for edge types
    unique_edge_types = set(str(data['edge_obj'].features.edge_type) for _, _, data in graph.edges(data=True))
    #color_map = px.colors.qualitative.D3_r
    color_map=[
    # "#1F77B4",
    # "#D62728",
    # "#316395",
    # "#8C564B",
    "#b3b6b7",
    "#7F7F7F",
    "#5dade2",
    "#000000",
    "#1c70c8",
    "#35327e"
    ]

    edge_type_to_color = {edge_type: color_map[i % len(color_map)] for i, edge_type in enumerate(unique_edge_types)}
    edges = {} 
    edges = {}
    for u, v, data in graph.edges(data=True):
        u_index = node_to_index[u]
        v_index = node_to_index[v]
        G.add_edge(u_index, v_index)
        
        edge = data['edge_obj'].features.edge_type
        edgename = f"{u_index}-{v_index}"
        if edges.get(edgename, None) is None:
            edges[edgename] = {'count': 0, 'type': [], 'colors':[]}
        # if edge.value == 3:
        #     color = "#7F7F7F"
        # else:
        color = edge_type_to_color[str(edge)]
        edges[edgename]['count'] += 1
        edges[edgename]['type'].append(str(edge))
        edges[edgename]['colors'].append(color)
    for i in range(num_nodes):
        print(i, labels[i], node_colors[i])
    Xe = []
    Ye = []
    Ze = []
    edge_colors_expanded = [] # rep


    for edgename, vals in edges.items():
        u, v = map(int, edgename.split('-'))
        count = vals['count']
        if count == 1:
            Xe += [Xn[u], Xn[v], None]
            Ye += [Yn[u], Yn[v], None]
            Ze += [Zn[u], Zn[v], None]
            edge_color = vals['colors'][0]
            edge_colors_expanded += [edge_color] * 3
        else:  # multi edges
            for i in range(count):
                # Calculate midpoint
                mid_x = (Xn[u] + Xn[v]) / 2
                mid_y = (Yn[u] + Yn[v]) / 2
                mid_z = (Zn[u] + Zn[v]) / 2

                # Calculate control point (slightly off the midpoint)
                offset = 0.1 * (i + 1)  # Increase offset for each edge
                ctrl_x = mid_x + offset * (Yn[v] - Yn[u])
                ctrl_y = mid_y - offset * (Xn[v] - Xn[u])
                ctrl_z = mid_z + offset

                # Generate Bézier curve
                x, y, z = bezier_curve([Xn[u], Yn[u], Zn[u]], 
                                    [ctrl_x, ctrl_y, ctrl_z], 
                                    [Xn[v], Yn[v], Zn[v]])

                Xe.extend(x.tolist() + [None])
                Ye.extend(y.tolist() + [None])
                Ze.extend(z.tolist() + [None])
                
                edge_color = vals['colors'][i]
                edge_colors_expanded.extend([edge_color] * (len(x) + 1))  # +1 for the None

    print("adding traces")
    
    edge_trace = go.Scatter3d(x=Xe,
                              y=Ye,
                              z=Ze,
                              mode='lines',
                              line=dict(color=edge_colors_expanded, width=2),
                              hoverinfo='none'
                              )

    node_trace = go.Scatter3d(x=Xn,
                            y=Yn,
                            z=Zn,
                            mode='markers',
                            name='actors',
                            marker=dict(symbol='circle',
                                        size=6,
                                        color=node_colors,
                                        line=dict(color='rgb(50,50,50)', width=0.5)
                                        ),
                            text=labels,
                            hoverinfo='text'
                            )
    # Create legend traces
    legend_traces = []
    for edge_type, color in edge_type_to_color.items():
        legend_trace = go.Scatter3d(x=[None], y=[None], z=[None], mode='lines',
                                    name=f'Edge Type: {edge_type}',
                                    line=dict(color=color, width=2),
                                    showlegend=True)
        legend_traces.append(legend_trace)
        # Create legend traces for node types
    for node_type, color in node_color_map.items():
        legend_trace = go.Scatter3d(x=[None], y=[None], z=[None], mode='markers',
                                    name=f'Node Type: {node_type}',
                                    marker=dict(size=6, color=color),
                                    showlegend=True)
        legend_traces.append(legend_trace)

    axis = dict(showbackground=False,
                showline=False,
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                title='')

    x_range = [min(Xn) - 0.5, max(Xn) + 0.5]
    y_range = [min(Yn) - 0.5, max(Yn) + 0.5]
    z_range = [min(Zn) - 0.5, max(Zn) + 0.5]
    scene=dict(
            xaxis=dict(axis, range=x_range),
            yaxis=dict(axis, range=y_range),
            zaxis=dict(axis, range=z_range),
            aspectmode='manual',
            aspectratio=dict(x=1, y=1, z=2),
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=0.5)
            ),
        )
    scale = 1.8
    scene['xaxis']['range'] = [scale * x for x in scene['xaxis']['range']]
    scene['yaxis']['range'] = [scale * y for y in scene['yaxis']['range']]
    scene['zaxis']['range'] = [scale * z for z in scene['zaxis']['range']]
    layout = go.Layout(
        title=title,
        width=800,
        height=600,
        showlegend=True,
        scene=scene,
        margin=dict(t=100),
        hovermode='closest',
    )

    fig = go.Figure(data=[edge_trace, node_trace] + legend_traces, layout=layout)
    if not display_inline:
        fig.write_html(save_path)
        fig.show()
    else:
        # draw with pio
        pio.show(fig)
    print("Done")


def draw_graph(graph: ParameterGraph, save_path: str=None, dim: str = '2d', title= 'Network Visualization', display_inline=False):
    '''
    Draw the global graph

    Args:
    - graph (MultiDiGraph): Global graph
    '''
    if dim == '2d':
        draw_nx_graph(graph, title)
    elif dim == '3d':
        draw_3d_graph(graph, title, save_path, display_inline)