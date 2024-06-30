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
    G.add_vertices(graph.number_of_nodes())

    # edge_counts = {}
    # edge_types = []
    # edge_colors = []
    
    
    # Create a color map for edge types
    unique_edge_types = set(data['edge_obj'].features.edge_type for _, _, data in graph.edges(data=True))
    #color_map = px.colors.qualitative.D3_r
    color_map=[
    "#1F77B4",
    "#D62728",
    "#316395",
    "#8C564B",
    
]
    edge_type_to_color = {edge_type: color_map[i % len(color_map)] for i, edge_type in enumerate(unique_edge_types)}
    edges = {}
    for u, v, data in graph.edges(data=True):
        G.add_edge(u, v)
        edge_type = data['edge_obj'].features.edge_type
        edgename = str(u) + '-' + str(v)
        if edges.get(edgename, None) is None:
            edges[edgename] = {'count': 0, 'type': [], 'colors':[]}
        if edge_type.value==3:
            color = "#7F7F7F"
        else:
            color = edge_type_to_color[edge_type]
        edges[edgename]['count'] += 1
        edges[edgename]['type'].append(edge_type)
        edges[edgename]['colors'].append(color)

        # edge_types.append(edge_type)
        # edge_colors.append(edge_type_to_color[edge_type])
    # print(set(edge_types))
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
    edge_colors_expanded = []

    # for (u, v), count in edge_counts.items():
    for edgename, vals in edges.items():
        u,v = edgename.split('-')
        u = int(u)
        v = int(v)
        count = vals['count']
        if count == 1:
            Xe += [layt[u][0], layt[v][0], None]
            Ye += [layt[u][1], layt[v][1], None]
            Ze += [layt[u][2], layt[v][2], None]
            edge_color = vals['colors'][0]
            edge_colors_expanded += [edge_color] * 3
        else:
            for i in range(len(vals['colors'])):
                mid_x = (layt[u][0] + layt[v][0]) / 2
                mid_y = (layt[u][1] + layt[v][1]) / 2
                mid_z = (layt[u][2] + layt[v][2]) / 2
                
                offset_scale = .05
                offset = offset_scale * (i + 1)
                ctrl_x = mid_x + offset * (layt[v][1] - layt[u][1])
                ctrl_y = mid_y - offset * (layt[v][0] - layt[u][0])
                ctrl_z = mid_z + offset
                
                t = np.linspace(0, 1, 20)
                x = (1-t)**2 * layt[u][0] + 2*(1-t)*t * ctrl_x + t**2 * layt[v][0]
                y = (1-t)**2 * layt[u][1] + 2*(1-t)*t * ctrl_y + t**2 * layt[v][1]
                z = (1-t)**2 * layt[u][2] + 2*(1-t)*t * ctrl_z + t**2 * layt[v][2]

                Xe.extend(x)
                Ye.extend(y)
                Ze.extend(z)
                edge_color = vals['colors'][i]
                #hard code light gray for now
                # edge_color = 'rgb(211,211,211)' 
                edge_colors_expanded.extend([edge_color] * len(x))

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
                                          color=group,
                                          colorscale='Viridis',
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

    axis = dict(showbackground=False,
                showline=False,
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                title='')

    layout = go.Layout(
        title=title,
        width=800,
        height=600,
        showlegend=True,
        scene=dict(
            xaxis=dict(axis),
            yaxis=dict(axis),
            zaxis=dict(axis),
        ),
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