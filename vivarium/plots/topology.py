import os

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from vivarium.core.process import Process, Factory
from vivarium.core.experiment import Experiment



def get_bipartite_graph(topology):
    ''' Get a graph with Processes, Stores, and edges from a Vivarium topology '''
    if 'topology' in topology:
        topology = topology['topology']

    process_nodes = []
    store_nodes = []
    edges = {}
    for process_id, connections in topology.items():
        process_id = str(process_id)
        process_id = process_id.replace("'", "").replace("_", "_\n")
        process_nodes.append(process_id)

        for port, store_id in connections.items():
            store_id = str(store_id)
            store_id = store_id.replace("'", "").replace(" ", "").replace("(", "").replace(")", "").replace(",", "\n")
            if store_id[-1:] == '\n':
                store_id = store_id[:-1]

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
    G = nx.Graph()
    for node_id in process_nodes:
        G.add_node(node_id, type='Process')
    for node_id in store_nodes:
        G.add_node(node_id, type='Store')
    for (process_id, store_id), port in edges.items():
        G.add_edge(process_id, store_id, port=port)

    return G


def graph_figure(
        graph,
        format='bipartite',
        show_ports=True,
        store_rgb='tab:blue',
        process_rgb='tab:orange',
        node_size=8000,
        font_size=10,
        node_distance=2.5,
        buffer=1.0,
        label_pos=0.75,
):
    """ Make a figure from a networkx graph """

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
    if format == 'bipartite':
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
                           node_color=process_rgb,
                           node_size=node_size,
                           node_shape='s'
                           )
    nx.draw_networkx_nodes(graph, pos,
                           nodelist=store_nodes,
                           node_color=store_rgb,
                           node_size=node_size,
                           node_shape='o'
                           )
    # edges
    colors = list(range(1, len(edges) + 1))
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
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()


def plot_compartment_topology(
        compartment,
        settings={},
        out_dir=None,
        filename=None,
):
    """ 
    an old function, reproduced by plot_topology """
    return plot_topology(
        compartment,
        settings,
        out_dir,
        filename)


def plot_topology(
        composite,
        settings={},
        out_dir=None,
        filename=None,
):
    """ Plot a composite's topology """

    network = composite.generate()

    # make networkx graph
    G = get_networkx_graph(network)

    # make graph figure
    fig = graph_figure(G, **settings)

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
        topology={
            'multiport1': {},
            'multiport2': {}}
    ):

    composite = MergePort({'topology': topology})
    network = composite.generate()

    # make networkx graph
    G = get_networkx_graph(network)

    # make graph figure
    fig = graph_figure(G)

    if save_fig:
        save_network(out_dir='out/topology', filename=str(topology))



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
