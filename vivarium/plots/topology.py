"""Plot topologies using networkx and matplotlib."""

import os
import copy
import argparse
from typing import Any, cast, Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.figure import Figure
import networkx as nx

from vivarium.core.process import Process
from vivarium.core.composer import Composer, Composite
from vivarium.core.store import generate_state
from vivarium.library.topology import normalize_path
from vivarium.library.dict_utils import deep_merge
from vivarium.processes.engulf import ToyAgent
from vivarium.processes.timeline import TimelineProcess


def construct_storage_path() -> Path:
    """Construct a Path to draw the standard "storage" flowchart shape."""
    # NOTE: After a MOVETO, we need to put the pen down for CLOSEPOLY to
    # complete a filled shape.
    _path_data = [
        # main shape
        (Path.MOVETO, [-1.000, -0.800]),  # start at bottom left corner
        (Path.CURVE4, [-0.900, -1.000]),  # bottom curve
        (Path.CURVE4, [+0.900, -1.000]),
        (Path.CURVE4, [+1.000, -0.800]),
        (Path.LINETO, [+1.000, +0.800]),  # right side
        (Path.CURVE4, [+0.900, +1.000]),  # top back curve
        (Path.CURVE4, [-0.900, +1.000]),
        (Path.CURVE4, [-1.000, +0.800]),
        (Path.LINETO, [-1.000, -0.800]),  # back to the bottom
        (Path.CLOSEPOLY, [0.00, 0.00]),  # close a filled poly-line shape

        # front edge 1
        (Path.MOVETO, [-1.000, +0.800]),  # to the top left corner
        (Path.CURVE4, [-0.900, +0.600]),  # top front main curve
        (Path.CURVE4, [+0.900, +0.600]),
        (Path.CURVE4, [+1.000, +0.800]),

        # front edge 2
        (Path.MOVETO, [-1.000, +0.700]),  # to the top left corner
        (Path.CURVE4, [-0.900, +0.500]),
        (Path.CURVE4, [+0.900, +0.500]),  # top front second curve
        (Path.CURVE4, [+1.000, +0.700]),
    ]
    _path_codes, _path_vertices = zip(*_path_data)
    return Path(_path_vertices, _path_codes)


STORAGE_PATH = construct_storage_path()


def get_bigraph(composite):
    """ Get a graph with Processes, Stores, and edges from a Vivarium topology """
    topology = composite['topology']
    processes = composite['processes']
    hierarchy_object = generate_state(
        processes=processes, topology=topology, initial_state={})

    # get path to processes and stores, name them by their paths
    # leaf_paths = hierarchy_object.depth(filter_function=lambda x: x.inner == {})
    process_paths = hierarchy_object.depth(filter_function=lambda x: isinstance(x.value, Process))
    process_nodes = ['\n'.join(process_path[0]) for process_path in process_paths]
    store_paths = hierarchy_object.depth(filter_function=lambda x: x.inner != {})
    store_nodes = ['\n'.join(store_path[0]) for store_path in store_paths if store_path[0] != tuple()]

    # get the edges between processes and stores
    edges = {}
    for (process_path, process) in process_paths:
        process_id = '\n'.join(process_path)
        assert process_id in process_nodes, (
            f"{process_id} process id is not in process_nodes list: {process_nodes}")

        process_topology = process.topology
        for port, store_path in process_topology.items():
            if isinstance(store_path, dict):
                if '_path' in store_path:
                    store_path = store_path['_path']
                else:
                    store_paths = tuple()

            store_path = normalize_path(process_path[:-1] + store_path)
            store_id = '\n'.join(store_path)
            if store_id not in store_nodes:
                # print(f"Adding {store_id} to store_nodes list {store_nodes}")
                store_paths.append((store_path, hierarchy_object.get_path(path=store_path)))
                store_nodes.append(store_id)

            # save the edge
            edge = (process_id, store_id)
            edges[edge] = port

    # get the place edges between hierarchy stores
    place_edges = []
    for (store_path, _) in store_paths:
        if len(store_path) > 1:
            # hierarchy place edges between inner/outer stores
            # for store_1, store_2 in zip(store_path, store_path[1:]):
            for level, _ in enumerate(store_path[1:], 1):
                store_1 = '\n'.join(store_path[:level])
                store_2 = '\n'.join(store_path[:level + 1])
                place_edge = (store_1, store_2)
                place_edges.append(place_edge)

    # are there overlapping names?
    overlap = [name for name in process_nodes if name in store_nodes]
    if overlap:
        print('{} shared by processes and stores'.format(overlap))

    return process_nodes, store_nodes, edges, place_edges


# def replace_node_labels(
#         node_labels, node_list=None, node_dict=None, edge_list=None, edge_dict=None):
#     if node_list:
#         return [
#             node_labels.get(node_id, node_id)
#             for node_id in node_list]
#     elif node_dict:
#         return {
#             node_labels.get(node_id, node_id): value
#             for node_id, value in node_dict.items()}
#     elif edge_list:
#         return [(
#             node_labels.get(node_1, node_1),
#             node_labels.get(node_2, node_2)
#         ) for (node_1, node_2) in edge_list]
#     elif edge_dict:
#         return {(
#              node_labels.get(node_1, node_1),
#              node_labels.get(node_2, node_2)
#          ): edge_name for (node_1, node_2), edge_name in edge_dict.items()}
#     return None


def remove(
        remove_nodes, node_list=None, node_dict=None, edge_list=None, edge_dict=None):
    """remove specified nodes"""
    if node_list:
        return [
            node_id for node_id in node_list
            if node_id not in remove_nodes]
    elif node_dict:
        return {
            node_id: value
            for node_id, value in node_dict.items()
            if node_id not in remove_nodes}
    elif edge_list:
        return [
            (node_1, node_2)
            for (node_1, node_2) in edge_list
            if (node_1 not in remove_nodes and node_2 not in remove_nodes)]
    elif edge_dict:
        return {
            (node_1, node_2): port
            for (node_1, node_2), port in edge_dict.items()
            if (node_1 not in remove_nodes and node_2 not in remove_nodes)}
    return None


def get_networkx_graph(composite, remove_nodes=None):
    """ Make a networkX graph from a Vivarium topology """
    remove_nodes = remove_nodes or []

    # get the nodes and edges from the composite
    process_nodes, store_nodes, edges, place_edges = get_bigraph(composite)

    # remove specified nodes
    process_nodes = remove(remove_nodes, node_list=process_nodes)
    store_nodes = remove(remove_nodes, node_list=store_nodes)
    edges = remove(remove_nodes, edge_dict=edges)
    place_edges = remove(remove_nodes, edge_list=place_edges)

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
        coordinates: Optional[Dict] = None,
        node_labels: Optional[Dict] = None,
        graph_format: str = 'horizontal',
        place_edges: Optional[list] = None,
        show_ports: bool = True,
        store_color: Any = 'tab:blue',
        process_color: Any = 'tab:orange',
        store_colors: Optional[Dict] = None,
        process_colors: Optional[Dict] = None,
        color_edges: bool = False,
        dashed_edges: bool = False,
        edge_width: float = 2.0,
        fill_color: Any = 'w',
        node_size: float = 8000,
        font_size: int = 14,
        node_distance: float = 3.0,
        buffer: float = 0.5,
        border_width: float = 3,
        custom_widths: Optional[Dict] = None,
        label_pos: float = 0.65,
) -> Figure:
    """ Make a figure from a networkx graph.

    :param graph: the networkx. Graph to plot
    :param coordinates: (dict) a dictionary of locations for all nodes in the graph, with {'node_id': (x, y)}
    :param node_labels: (dict) a dictionary of labels for all nodes in the graph, with {'node_id': 'node_label'}
    :param graph_format: 'horizontal', 'vertical', or 'hierarchy'
    :param store_color: default color for the Store nodes; any matplotlib color value
    :param process_color: default color for the Process nodes; any matplotlib color value
    :param store_colors: (dict) specific colors for the Store nodes, mapping from store name to matplotlib color
    :param process_colors: (dict) specific colors for the Process nodes, mapping from store name to matplotlib color
    :param color_edges: color each edge between Store and Process a different color
    :param dashed_edges: edges between Store and Process are dashed lines
    :param show_ports: whether to show the Port labels
    :param fill_color: fill color for the Store and Process nodes; any
        matplotlib color value
    :param node_size: size to draw the Store and Process nodes
    :param font_size: size for the Store, Process, and Port labels
    :param node_distance: distance to spread out the nodes
    :param buffer: buffer space around the graph
    :param border_width: width of the border line around Store and Process nodes
    :param custom_widths: (dict) changes the widths of specific Store and Process nodes (defaults to board_width)
    :param label_pos: position of the Port labels along their connection lines,
        (0=head, 0.5=center, 1=tail)
    """
    node_labels = node_labels or {}
    process_colors = process_colors or {}
    store_colors = store_colors or {}
    place_edges = place_edges or []
    custom_widths = custom_widths or {}

    node_attributes = dict(graph.nodes.data())
    process_nodes = [
        node_id for node_id, attributes in node_attributes.items()
        if attributes['type'] == 'Process']
    store_nodes = [
        node_id for node_id, attributes in node_attributes.items()
        if attributes['type'] == 'Store']

    # fill in all node labels
    node_labels = {
        node_id: node_labels.get(node_id, node_id)
        for node_id in node_attributes.keys()}

    edge_list = list(graph.edges)
    edges = {}
    for edge in edge_list:
        if 'port' in graph.edges[edge]:
            edges[edge] = graph.edges[edge]['port']

    # get position
    pos: Dict = {}
    if graph_format:
        pos_format = graph_format_location(
            graph,
            process_nodes,
            store_nodes,
            place_edges,
            graph_format)
        pos = deep_merge(pos, pos_format)
    if coordinates:
        pos_format = {
            node: np.array(coord)
            for node, coord in coordinates.items()
            if node in process_nodes + store_nodes}
        pos = deep_merge(pos, pos_format)

    # initialize figure based on positions, nodes, and buffer
    pos_values = list(pos.values())
    xs = [p[0] for p in pos_values]
    ys = [p[1] for p in pos_values]
    xr = max(xs) - min(xs)
    yr = max(ys) - min(ys)
    fig = plt.figure(1, figsize=(
        xr * node_distance + 2 * buffer,
        yr * node_distance + 2 * buffer))

    # get node colors
    process_color_list = [
        process_colors.get(process_name, process_color)
        for process_name in process_nodes]
    store_color_list = [
        store_colors.get(store_name, store_color)
        for store_name in store_nodes]

    #get node widths
    process_width_list = [
        custom_widths.get(process_name, border_width)
        for process_name in process_nodes]

    store_width_list = [
        custom_widths.get(store_name, border_width)
        for store_name in store_nodes]

    # draw the process nodes
    nx.draw_networkx_nodes(graph, pos,
                           nodelist=process_nodes,
                           node_color=fill_color,
                           edgecolors=process_color_list,
                           node_size=node_size,
                           linewidths=process_width_list,
                           node_shape='s')
    # draw the store nodes
    nx.draw_networkx_nodes(graph, pos,
                           nodelist=store_nodes,
                           node_color=fill_color,
                           edgecolors=store_color_list,
                           node_size=node_size,
                           linewidths=store_width_list,
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
    # thicker edges for hierarchy connections
    edge_args['width'].extend([edge_width * 2 for _ in place_edges])
    if dashed_edges:
        edge_args['style'] = ['dashed' for _ in edges.keys()]
        edge_args['style'].extend(['solid' for _ in place_edges])

    nx.draw_networkx_edges(graph, pos, **edge_args)

    # node labels
    nx.draw_networkx_labels(graph, pos,
                            labels=node_labels,
                            font_size=font_size)
    if show_ports:
        # edge labels
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


def get_hierarchy_levels(graph, store_nodes, place_edges):
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

    return levels


def graph_format_location(
    graph,
    process_nodes,
    store_nodes,
    place_edges,
    graph_format,
):
    n_stores = len(store_nodes)
    n_processes = len(process_nodes)

    # get positions
    pos = {}
    if graph_format == 'hierarchy':
        levels = get_hierarchy_levels(
            graph, store_nodes, place_edges)

        # buffer makes things centered
        n_max = max([len(level) for level in levels])

        # place the process nodes according on the left
        # buffer_processes = (n_max - n_processes) / 2
        for idx, node_id in enumerate(process_nodes, 0):
            pos[node_id] = np.array([-1, -idx])

        # place the store nodes according to levels
        for level_idx, level in enumerate(levels, 0):
            level_buffer = (n_max - len(level)) / 2
            for idx, node_id in enumerate(level, 1):
                pos[node_id] = np.array([level_buffer + 0.9*idx, -1.2*level_idx])

    elif graph_format == 'vertical':
        # processes in a column, and stores in a column
        for idx, node_id in enumerate(process_nodes, 1):
            pos[node_id] = np.array([-1, -idx])
        for idx, node_id in enumerate(store_nodes, 1):
            pos[node_id] = np.array([1, -idx])

    elif graph_format == 'horizontal':
        # processes in a row, and stores in a row
        # buffer makes things centered
        n_max = max(n_stores, n_processes)
        buffer_processes = (n_max - n_processes) / 2
        buffer_stores = (n_max - n_stores) / 2

        for idx, node_id in enumerate(process_nodes, 1):
            pos[node_id] = np.array([buffer_processes + idx, 1])
        for idx, node_id in enumerate(store_nodes, 1):
            pos[node_id] = np.array([buffer_stores + idx, -1])

    return pos


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

    settings = copy.deepcopy(settings) or {}
    if isinstance(composite, Composer):
        composite = composite.generate()
    elif isinstance(composite, Process):
        composite = composite.generate()
    assert 'processes' in composite and 'topology' in composite

    # make networkx graph
    remove_nodes = settings.pop('remove_nodes', [])
    g, place_edges = get_networkx_graph(composite, remove_nodes)
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


def test_merge_port_topology(
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


def merge_port_configs(topology_id):
    topology = {}
    settings = {}
    if topology_id == '0':
        pass
    elif topology_id == '1':
        # vertical graph format
        topology = {
                'multiport1': {
                    'a': ('D',),
                    'b': ('D',),
                    'c': ('D',),
                },
                'multiport2': {}}
        settings = {
            'graph_format': 'vertical',
            'process_colors': {'multiport1': 'r'},
            'store_colors': {'C': 'k'},
        }
    elif topology_id == '2':
        # default horizontal graph format
        topology = {
                'multiport1': {
                    'a': ('A', 'AA',),
                    'b': ('A', 'BB',),
                    'c': ('A', 'CC',),
                },
                'multiport2': {}}
        settings = {'dashed_edges': True}
    elif topology_id == '3':
        # hierarchy graph format
        topology = {
                'multiport1': {
                    'a': ('A', 'AA', 'AAA',),
                    'b': ('A', 'BB',),
                    'c': ('A', 'CC',),
                },
                'multiport2': {}}
        settings = {
            'graph_format': 'hierarchy',
            'store_color': 'navy'
        }
    elif topology_id == '4':
        # manual locations
        topology = {
                'multiport1': {
                    'a': ('A', 'AA',),
                    'b': ('A', 'BB', 'BBB'),
                    'c': ('A', 'CC',),
                },
                'multiport2': {}}
        settings = {
            'coordinates': {
                'multiport1': (-1, 0), 'multiport2': (-1, -2),
                'A': (1, 0), 'B': (2, 0), 'C': (3, 0),
                'A\nAA': (1, -2), 'A\nBB': (2, -2), 'A\nCC': (3, -2),
                'A\nBB\nBBB': (2, -4),
            },
            'store_color': 'navy',
            'dashed_edges': True,
            'remove_nodes': ['A\nBB', 'multiport1']
        }
    else:
        raise ValueError(f'topology_id "{topology_id}" is invalid')
    return topology, settings


def main():

    parser = argparse.ArgumentParser(description='topology')
    parser.add_argument('-t', default=None, type=str, help='topology id for test_merge_port_topology')
    parser.add_argument('-x', action='store_true', help='hierarchy id')
    args = parser.parse_args()

    if args.t:
        topology_id = str(args.t)
        topology, settings = merge_port_configs(topology_id)
        fig_name = f"topology_{topology_id}"
        test_merge_port_topology(
            fig_name=fig_name,
            topology=topology,
            settings=settings)

    elif args.x:
        agent_ids = ['1', '2']

        config = {
            'exchange': {
                'internal_path': ('concentrations',),
                'external_path': ('..', '..', 'concentrations')},
            'engulf': {
                'inner_path': ('agents',),
                'outer_path': ('..', '..', 'agents')}}
        toy_agent_composer = ToyAgent(config)
        toy_agent_1 = toy_agent_composer.generate(path=('agents', agent_ids[0]))
        toy_agent_2 = toy_agent_composer.generate(path=('agents', agent_ids[1]))

        timeline = [(3, {('agents', agent_ids[1], 'engulf-trigger'): [agent_ids[0]]})]
        timeline_composer = TimelineProcess({'timeline': timeline})
        full_composite = timeline_composer.generate()

        full_composite.merge(composite=toy_agent_1)
        full_composite.merge(composite=toy_agent_2)

        # plot topology
        agent_1_x = 0
        agent_2_x = 6
        level_1 = -1
        level_2 = -2
        level_3 = -3
        settings = {
            'coordinates': {
                # timeline
                'timeline': (3, 0),
                # agent 1
                'agents\n1\nexchange': (agent_1_x, -1), 'agents\n1\nengulf': (agent_1_x, -1.5), 'agents\n1\nexpel': (agent_1_x, -2),
                # agent 2
                'agents\n2\nexchange': (agent_2_x, -1), 'agents\n2\nengulf': (agent_2_x, -1.5), 'agents\n2\nexpel': (agent_2_x, -2),
                # 1st level stores
                'concentrations': (2, level_1), 'agents': (3, level_1), 'global': (4, level_1/2),
                # 2nd level stores
                'agents\n1': (1.75, level_2), 'agents\n2': (4.25, level_2),
                # 3rd level stores
                # agent 1
                'agents\n1\nconcentrations': (1, level_3), 'agents\n1\nengulf-trigger': (1.5, level_3),
                'agents\n1\nexpel-trigger': (2, level_3), 'agents\n1\nagents': (2.5, level_3),
                # agent 2
                'agents\n2\nconcentrations': (3.5, level_3), 'agents\n2\nengulf-trigger': (4, level_3),
                'agents\n2\nexpel-trigger': (4.5, level_3), 'agents\n2\nagents': (5, level_3),
            },
            'graph_format': 'hierarchy',
            'store_color': 'navy',
            'dashed_edges': True,
            'node_distance': 5,
            'node_labels': {
                'agents\n1': 'agent1',
                'agents\n2': 'agent2',
                'agents\n1\nexpel-trigger': 'expel',
                'agents\n1\nengulf-trigger': 'engulf',
                'agents\n1\nconcentrations': 'conc',
                'agents\n1\nagents': 'agents',
                'agents\n2\nexpel-trigger': 'expel',
                'agents\n2\nengulf-trigger': 'engulf',
                'agents\n2\nconcentrations': 'conc',
                'agents\n2\nagents': 'agents',
            }
        }
        config = {
            'settings': settings,
            'out_dir': 'out/topology',
            'filename': 'embedded'}
        plot_topology(full_composite, **config)


if __name__ == '__main__':
    main()
