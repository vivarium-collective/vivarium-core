"""Plot topologies using networkx and matplotlib."""

import os
from typing import Any, cast, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import networkx as nx

from vivarium.core.process import Process, Factory


def construct_storage_path() -> Path:
    """Construct a Path to draw the standard "storage" flowchart shape."""
    # NOTE: After a MOVETO, we need to put the pen down for CLOSEPOLY to
    # complete a filled shape.
    _path_data = [
        (Path.MOVETO, [-1.000, -0.800]),
        (Path.CURVE4, [-0.900, -1.000]),  # bottom curve
        (Path.CURVE4, [+0.900, -1.000]),
        (Path.CURVE4, [+1.000, -0.800]),
        (Path.LINETO, [+1.000, +0.800]),  # right side
        (Path.CURVE4, [+0.900, +1.000]),  # top back curve
        (Path.CURVE4, [-0.900, +1.000]),
        (Path.CURVE4, [-1.000, +0.800]),
        (Path.CURVE4, [-0.900, +0.600]),  # top front main curve
        (Path.CURVE4, [+0.900, +0.600]),
        (Path.CURVE4, [+1.000, +0.800]),
        (Path.LINETO, [+1.000, +0.700]),  # upper left edge
        (Path.CURVE4, [+0.900, +0.500]),  # top front second curve
        (Path.CURVE4, [-0.900, +0.500]),
        (Path.CURVE4, [-1.000, +0.700]),
        (Path.LINETO, [-1.000, +0.800]),  # left edge
        (Path.CLOSEPOLY, [0.00, 0.00])]   # close a filled poly-line shape
    _path_codes, _path_vertices = zip(*_path_data)
    return Path(_path_vertices, _path_codes)


STORAGE_PATH = construct_storage_path()


def get_bipartite_graph(topology):
    ''' Get a graph with Processes, Stores, and edges from a Vivarium topology '''
    if 'topology' in topology:
        topology = topology['topology']

    process_nodes = []
    store_nodes = []
    edges = {}
    for process_id, connections in topology.items():
        process_id = process_id.replace("_", "_\n")  # line breaks at underscores
        process_nodes.append(process_id)

        for port, store_id in connections.items():
            store_id = '\n'.join(store_id)  # TODO: a fancier graph for a dict
            store_id = store_id.replace('..\n', '⬆︎')

            if store_id not in store_nodes:
                store_nodes.append(store_id)

            edge = (process_id, store_id)
            edges[edge] = port

    # are there overlapping names?
    overlap = [name for name in process_nodes if name in store_nodes]
    if overlap:
        print('{} shared by processes and stores'.format(overlap))

    return process_nodes, store_nodes, edges


def get_networkx_graph(topology):
    ''' Make a networkX graph from a Vivarium topology '''
    process_nodes, store_nodes, edges = get_bipartite_graph(topology)

    # make networkX graph
    g = nx.Graph()
    for node_id in process_nodes:
        g.add_node(node_id, type='Process')
    for node_id in store_nodes:
        g.add_node(node_id, type='Store')
    for (process_id, store_id), port in edges.items():
        g.add_edge(process_id, store_id, port=port)

    return g


def graph_figure(
        graph: nx.Graph,
        *,
        graph_format: str = 'bipartite',
        show_ports: bool = True,
        store_color: Any = 'tab:blue',
        process_color: Any = 'tab:orange',
        color_edges: bool = True,
        fill_color: Any = 'w',
        node_size: float = 8000,
        font_size: int = 14,
        node_distance: float = 2.5,
        buffer: float = 1.0,
        border_width: float = 3,
        label_pos: float = 0.65,
) -> plt.Figure:
    """ Make a figure from a networkx graph.

    :param graph: the networkx.Graph to plot
    :param graph_format: 'bipartite' or not
    :param show_ports: whether to show the Port labels
    :param store_color: color for the Store nodes; any matplotlib color value
    :param process_color: color for the Process nodes; any matplotlib color value
    :param color_edges: color each edge between Store and Process a different color
    :param fill_color: fill color for the Store and Process nodes; any
        matplotlib color value
    :param node_size: size to draw the Store and Process nodes
    :param font_size: size for the Store, Process, and Port labels
    :param node_distance: distance to spread out the nodes
    :param buffer: buffer space around the graph
    :param border_width: width of the border line around Store and Process nodes
    :param label_pos: position of the Port labels along their connection lines,
        (0=head, 0.5=center, 1=tail)
    """

    node_attributes = dict(graph.nodes.data())
    process_nodes = [
        node_id for node_id, attributes in node_attributes.items()
        if attributes['type'] == 'Process']
    store_nodes = [
        node_id for node_id, attributes in node_attributes.items()
        if attributes['type'] == 'Store']

    edge_list = list(graph.edges)
    edges = {
        edge: graph.edges[edge]['port']
        for edge in edge_list}

    # get positions
    pos = {}
    if graph_format == 'bipartite':
        for idx, node_id in enumerate(process_nodes, 1):
            pos[node_id] = np.array([-1, -idx])
        for idx, node_id in enumerate(store_nodes, 1):
            pos[node_id] = np.array([1, -idx])

    # plot
    n_rows = max(len(process_nodes), len(store_nodes))
    fig = plt.figure(1, figsize=(12, n_rows * node_distance))

    # nx.draw(graph, pos=pos, node_size=node_size)
    nx.draw_networkx_nodes(graph, pos,
                           nodelist=process_nodes,
                           node_color=fill_color,
                           edgecolors=process_color,
                           node_size=node_size,
                           linewidths=border_width,
                           node_shape='s'
                           )
    nx.draw_networkx_nodes(graph, pos,
                           nodelist=store_nodes,
                           node_color=fill_color,
                           edgecolors=store_color,
                           node_size=node_size,
                           linewidths=border_width,
                           node_shape=cast(str, STORAGE_PATH)
                           )
    # edges
    if color_edges:
        colors = list(range(1, len(edges) + 1))
    else:
        colors = 'k'
    nx.draw_networkx_edges(graph, pos,
                           edge_color=colors,
                           width=1.5)
    # labels
    nx.draw_networkx_labels(graph, pos,
                            font_size=font_size)
    if show_ports:
        nx.draw_networkx_edge_labels(graph, pos,
                                     edge_labels=edges,
                                     font_size=font_size,
                                     label_pos=label_pos)

    # add buffer
    xmin, xmax, ymin, ymax = plt.axis()
    plt.xlim(xmin - buffer, xmax + buffer)
    plt.ylim(ymin - buffer, ymax + buffer)
    plt.axis('off')

    return fig


def save_network(out_dir='out', filename='network'):
    os.makedirs(out_dir, exist_ok=True)
    fig_path = os.path.join(out_dir, filename)
    print(f"Writing {fig_path}")
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()


def plot_compartment_topology(
        compartment,
        settings: Optional[Dict[str, Any]] = None,
        out_dir=None,
        filename=None,
):
    """ 
    an old function, reproduced by plot_topology """
    settings = settings or {}
    return plot_topology(
        compartment,
        settings,
        out_dir,
        filename)


def plot_topology(
        composite,
        settings: Optional[Dict[str, Any]] = None,
        out_dir=None,
        filename=None,
):
    """ Plot a composite's topology """

    settings = settings or {}
    network = composite.generate()

    # make networkx graph
    g = get_networkx_graph(network)

    # make graph figure
    fig = graph_figure(g, **settings)

    if out_dir is not None:
        # save fig
        save_network(
            out_dir=out_dir,
            filename=filename)
    else:
        return fig


# tests
class MultiPort(Process):
    name = 'multi_port'

    def ports_schema(self):
        return {
            'a': {
                'molecule': {
                    '_default': 0,
                    '_emit': True}},
            'b': {
                'molecule': {
                    '_default': 0,
                    '_emit': True}},
            'c': {
                'molecule': {
                    '_default': 0,
                    '_emit': True}}}

    def next_update(self, timestep, states):
        return {
            'a': {'molecule': 1},
            'b': {'molecule': 1},
            'c': {'molecule': 1}}


class MergePort(Factory):
    """combines both of MultiPort's ports into one store"""
    name = 'multi_port_generator'
    defaults = {
        'topology': {
            'multiport1': {
                'a': ('A',),
                'b': ('B',),
                'c': ('C',),
            },
            'multiport2': {
                'a': ('A',),
                'b': ('B',),
                'c': ('C',),
            },
        }
    }

    def __init__(self, config=None):
        super().__init__(config)
        self.topology = self.config['topology']

    def generate_processes(self, config):
        return {
            'multiport1': MultiPort({}),
            'multiport2': MultiPort({}),
        }

    def generate_topology(self, config):
        return self.topology


def test_graph(
        save_fig=False,
        topology: Optional[Dict[str, Any]] = None
):
    if topology is None:
        topology = {
            'multiport1': {},
            'multiport2': {}}
    composite = MergePort({'topology': topology})
    network = composite.generate()

    # make networkx graph
    g = get_networkx_graph(network)

    # make graph figure
    fig = graph_figure(g)

    if save_fig:
        save_network(out_dir='out/topology', filename='topology')


if __name__ == '__main__':
    # topology = {
    #         'multiport1': {
    #             'a': ('D',),
    #             'b': ('D',),
    #             'c': ('D',),
    #         },
    #         'multiport2': {}}

    topology = {
            'multiport1': {
                'a': ('A', 'AA',),
                'b': ('A', 'BB',),
                'c': ('A', 'CC',),
            },
            'multiport2': {}}

    test_graph(
        save_fig=True,
        topology=topology,
    )
