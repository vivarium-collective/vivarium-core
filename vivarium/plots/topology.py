import os

import numpy as np

import matplotlib.pyplot as plt
import networkx as nx


def plot_compartment_topology(
        compartment,
        settings={},
        out_dir=None,
        filename=None,
):
    """
    Make a plot of the topology
     - compartment: a compartment
    """
    store_rgb = [x/255 for x in [239, 131, 148]]
    process_rgb = [x / 255 for x in [249, 204, 86]]
    node_size = 8000
    font_size = 10
    node_distance = 2.5
    buffer = 1.0
    label_pos = 0.75

    network = compartment.generate({})
    topology = network['topology']
    processes = network['processes']

    # get figure settings
    show_ports = settings.get('show_ports', True)

    # make graph from topology
    G = nx.Graph()
    process_nodes = []
    store_nodes = []
    edges = {}
    for process_id, connections in topology.items():
        process_id = str(process_id)
        process_id = process_id.replace("'", "").replace("_", "_\n")
        process_nodes.append(process_id)
        G.add_node(process_id)

        for port, store_id in connections.items():
            store_id = str(store_id)
            store_id = store_id.replace("'", "").replace(" ", "").replace("(", "").replace(")", "").replace(",", "\n")

            if store_id not in store_nodes:
                store_nodes.append(store_id)
            if store_id not in list(G.nodes):
                G.add_node(store_id)

            edge = (process_id, store_id)
            edges[edge] = port
            G.add_edge(process_id, store_id)

    # are there overlapping names?
    overlap = [name for name in process_nodes if name in store_nodes]
    if overlap:
        print('{} shared by processes and stores'.format(overlap))

    # get positions
    pos = {}
    n_rows = max(len(process_nodes), len(store_nodes))
    fig = plt.figure(1, figsize=(12, n_rows * node_distance))

    for idx, node_id in enumerate(process_nodes, 1):
        pos[node_id] = np.array([-1, -idx])
    for idx, node_id in enumerate(store_nodes, 1):
        pos[node_id] = np.array([1, -idx])

    # plot
    nx.draw_networkx_nodes(G, pos,
                           nodelist=process_nodes,
                           node_color=process_rgb,
                           node_size=node_size,
                           node_shape='s')
    nx.draw_networkx_nodes(G, pos,
                           nodelist=store_nodes,
                           node_color=store_rgb,
                           node_size=node_size,
                           node_shape='o')
    # edges
    colors = list(range(1, len(edges)+1))
    nx.draw_networkx_edges(G, pos,
                           edge_color=colors,
                           width=1.5)
    # labels
    nx.draw_networkx_labels(G, pos,
                            font_size=font_size)
    if show_ports:
        nx.draw_networkx_edge_labels(G, pos,
                                 edge_labels=edges,
                                 font_size=font_size,
                                 label_pos=label_pos)

    # add buffer
    xmin, xmax, ymin, ymax = plt.axis()
    plt.xlim(xmin - buffer, xmax + buffer)
    plt.ylim(ymin - buffer, ymax + buffer)
    plt.axis('off')

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        if filename is None:
            filename = 'topology'
        # save figure
        fig_path = os.path.join(out_dir, filename)
        plt.savefig(fig_path, bbox_inches='tight')
        plt.close()
    else:
        return fig
