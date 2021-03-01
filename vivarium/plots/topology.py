"""Plot topologies using networkx and matplotlib."""

import os
import argparse
from typing import Any, cast, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.figure import Figure
import networkx as nx

from vivarium.core.process import Process, Composer


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
    """ Get a graph with Processes, Stores, and edges from a Vivarium topology """
    if 'topology' in topology:
        topology = topology['topology']

    process_nodes = []
    store_nodes = []
    edges = {}
    compartment_nodes = []
    place_edges = []
    for process_id, connections in topology.items():
        process_nodes.append(process_id)

        for port, path in connections.items():
            # store_id = '\n'.join(path)  # TODO: a fancier graph for a dict
            # store_id = store_id.replace('..\n', '⬆︎')
            store_id = path[-1]
            if store_id not in store_nodes:
                store_nodes.append(store_id)

            if len(path) > 1:
                # hierarchy place edges between inner/outer stores
                for store_1, store_2 in zip(path, path[1:]):

                    # save the place edge
                    place_edge = (store_1, store_2)
                    place_edges.append(place_edge)

                    # add all intermediate stores to store list
                    if store_1 not in compartment_nodes:
                        compartment_nodes.append(store_1)

            edge = (process_id, store_id)
            edges[edge] = port

    # are there overlapping names?
    overlap = [name for name in process_nodes if name in store_nodes]
    if overlap:
        print('{} shared by processes and stores'.format(overlap))

    return process_nodes, store_nodes, edges, place_edges


def get_networkx_graph(topology):
    """ Make a networkX graph from a Vivarium topology """
    process_nodes, store_nodes, edges, place_edges = get_bipartite_graph(topology)

    # make networkX graph
    g = nx.Graph()
    for node_id in process_nodes:
        g.add_node(node_id, type='Process')
    for node_id in store_nodes:
        g.add_node(node_id, type='Store')

    # add topology edges
    for (process_id, store_id), port in edges.items():
        g.add_edge(process_id, store_id, port=port)

    return g, place_edges


def graph_figure(
        graph: nx.Graph,
        *,
        graph_format: str = 'horizontal',
        place_edges: Optional[list] = None,
        show_ports: bool = True,
        store_color: Any = 'tab:blue',
        process_color: Any = 'tab:orange',
        store_colors: Optional[Dict] = None,
        process_colors: Optional[Dict] = None,
        color_edges: bool = True,
        edge_width: float = 2.0,
        fill_color: Any = 'w',
        node_size: float = 8000,
        font_size: int = 14,
        node_distance: float = 2.5,
        buffer: float = 1.0,
        border_width: float = 3,
        label_pos: float = 0.65,
) -> Figure:
    """ Make a figure from a networkx graph.

    :param graph: the networkx.Graph to plot
    :param graph_format: 'horizontal', 'vertical', or 'hierarchy'
    :param show_ports: whether to show the Port labels
    :param store_color: default color for the Store nodes; any matplotlib color value
    :param process_color: default color for the Process nodes; any matplotlib color value
    :param store_colors: (dict) specific colors for the Store nodes, mapping from store name to matplotlib color
    :param process_colors: (dict) specific colors for the Process nodes, mapping from store name to matplotlib color
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
    process_colors = process_colors or {}
    store_colors = store_colors or {}
    place_edges = place_edges or []

    node_attributes = dict(graph.nodes.data())
    process_nodes = [
        node_id for node_id, attributes in node_attributes.items()
        if attributes['type'] == 'Process']
    store_nodes = [
        node_id for node_id, attributes in node_attributes.items()
        if attributes['type'] == 'Store']

    edge_list = list(graph.edges)
    edges = {}
    for edge in edge_list:
        if 'port' in graph.edges[edge]:
            edges[edge] = graph.edges[edge]['port']

    # plot
    n_stores = len(store_nodes)
    n_processes = len(process_nodes)
    n_max = max(n_stores, n_processes)

    # get positions
    pos = {}
    if graph_format == 'hierarchy':
        # add new place edges by iterating over all place_edges
        outers = set()
        inners = set()
        for (store_1, store_2) in place_edges:
            outers.add(store_1)
            inners.add(store_2)
            graph.add_edge(store_1, store_2, place_edge=True)

        # add non-embedded nodes to outers
        all_stores = outers.union(inners)
        non_embedded = set(store_nodes).difference(all_stores)
        outers.update(non_embedded)

        # add intermediate nodes to store_nodes
        intermediate_nodes = all_stores.difference(store_nodes)
        store_nodes.extend(list(intermediate_nodes))

        # determine the hierarchy levels
        levels = []
        accounted = set()
        unaccounted = outers.union(inners)
        top_level = outers - inners
        levels.append(list(top_level))
        accounted.update(top_level)
        unaccounted = unaccounted.difference(accounted)
        while len(unaccounted) > 0:
            next_level = set()
            for (store_1, store_2) in place_edges:
                if store_1 in accounted and store_2 in unaccounted:
                    next_level.add(store_2)
            levels.append(list(next_level))
            accounted.update(next_level)
            unaccounted = unaccounted.difference(accounted)

        # buffer makes things centered
        n_max = max([len(level) for level in levels])
        buffer_processes = (n_max - n_processes) / 2

        # place the nodes according to levels
        for idx, node_id in enumerate(process_nodes, 1):
            pos[node_id] = np.array([buffer_processes + idx, 1])
        for level_idx, level in enumerate(levels, 1):
            level_buffer = (n_max - len(level)) / 2
            for idx, node_id in enumerate(level, 1):
                pos[node_id] = np.array([level_buffer + idx, -1*level_idx])

        fig = plt.figure(1, figsize=(n_max * node_distance, 6 + 3 * len(levels)))

    elif graph_format == 'vertical':
        # processes in a column, and stores in a column
        for idx, node_id in enumerate(process_nodes, 1):
            pos[node_id] = np.array([-1, -idx])
        for idx, node_id in enumerate(store_nodes, 1):
            pos[node_id] = np.array([1, -idx])

        fig = plt.figure(1, figsize=(12, n_max * node_distance))

    elif graph_format == 'horizontal':
        # processes in a row, and stores in a row
        # buffer makes things centered
        buffer_processes = (n_max - n_processes) / 2
        buffer_stores = (n_max - n_stores) / 2

        for idx, node_id in enumerate(process_nodes, 1):
            pos[node_id] = np.array([buffer_processes + idx, 1])
        for idx, node_id in enumerate(store_nodes, 1):
            pos[node_id] = np.array([buffer_stores + idx, -1])

        fig = plt.figure(1, figsize=(n_max * node_distance, 12))

    # get node colors
    process_color_list = [
        process_colors.get(process_name, process_color)
        for process_name in process_nodes]
    store_color_list = [
        store_colors.get(store_name, store_color)
        for store_name in store_nodes]

    # draw the process nodes
    nx.draw_networkx_nodes(graph, pos,
                           nodelist=process_nodes,
                           node_color=fill_color,
                           edgecolors=process_color_list,
                           node_size=node_size,
                           linewidths=border_width,
                           node_shape='s')
    # draw the store nodes
    nx.draw_networkx_nodes(graph, pos,
                           nodelist=store_nodes,
                           node_color=fill_color,
                           edgecolors=store_color_list,
                           node_size=node_size,
                           linewidths=border_width,
                           node_shape=cast(str, STORAGE_PATH))

    # edges
    edge_args = {}

    # edge colors
    if color_edges:
        edge_args['edge_cmap'] = plt.get_cmap('nipy_spectral')
        edge_args['edge_color'] = list(range(1, len(edges) + 1))
        if graph_format == 'hierarchy':
            edge_args['edge_color'].extend([0 for _ in place_edges])

    # edge width
    edge_args['width'] = [edge_width for _ in edges.keys()]
    if graph_format == 'hierarchy':
        # thicker edges for hierarchy connections
        edge_args['width'].extend([edge_width * 2 for _ in place_edges])

    nx.draw_networkx_edges(graph, pos,
                           # width=1.5,
                           **edge_args)

    # edge labels
    nx.draw_networkx_labels(graph, pos,
                            font_size=font_size)
    if show_ports:
        nx.draw_networkx_edge_labels(graph, pos,
                                     edge_labels=edges,
                                     font_size=font_size,
                                     label_pos=label_pos)

    # add white buffer around final figure
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
    # plt.close()


def plot_topology(
        composite,
        settings: Optional[Dict] = None,
        out_dir: Optional[str] = None,
        filename: Optional[str] = None,
):
    """ Plot a composite's topology """

    settings = settings or {}
    if isinstance(composite, Composer):
        composite = composite.generate()

    # make networkx graph
    g, place_edges = get_networkx_graph(composite)
    settings['place_edges'] = place_edges

    # make graph figure
    fig = graph_figure(g, **settings)

    if out_dir is not None:
        # save fig
        save_network(
            out_dir=out_dir,
            filename=filename)
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


class MergePort(Composer):
    """combines both of MultiPort's ports into one store"""
    name = 'multi_port_composer'
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
        fig_name=None,
        topology=None,
        settings=None,
):

    if topology is None:
        topology = {
            'multiport1': {},
            'multiport2': {}}
    composer = MergePort({'topology': topology})

    config = {'settings': settings}
    if fig_name:
        config.update({
            'out_dir': 'out/topology',
            'filename': fig_name})

    plot_topology(composer, **config)


def main():

    parser = argparse.ArgumentParser(description='topology')
    parser.add_argument(
        '--topology', '-t',
        type=str,
        choices=['1', '2', '3'],
        help='the topology id'
    )
    args = parser.parse_args()

    settings = {}
    topology_id = str(args.topology)
    if topology_id == '1':
        topology = {
                'multiport1': {
                    'a': ('D',),
                    'b': ('D',),
                    'c': ('D',),
                },
                'multiport2': {}}
        fig_name = 'topology_1'
        settings = {
            'graph_format': 'vertical',
            'process_colors': {'multiport1': 'r'},
            'store_colors': {'C': 'k'},
        }
    elif topology_id == '2':
        topology = {
                'multiport1': {
                    'a': ('A', 'AA',),
                    'b': ('A', 'BB',),
                    'c': ('A', 'CC',),
                },
                'multiport2': {}}
        fig_name = 'topology_2'
    elif topology_id == '3':
        topology = {
                'multiport1': {
                    'a': ('A', 'AA', 'AAA',),
                    'b': ('A', 'BB',),
                    'c': ('A', 'CC',),
                },
                'multiport2': {}}
        fig_name = 'topology_3'
        settings = {
            'graph_format': 'hierarchy',
            'store_color': 'navy'
        }
    else:
        pass
        # more complex topology, with ..?

    test_graph(
        fig_name=fig_name,
        topology=topology,
        settings=settings,
    )


if __name__ == '__main__':
    main()
