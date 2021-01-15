"""
Make formatted files for Gephi Network Visualization

@organization: Covert Lab, Department of Bioengineering, Stanford University
"""

import csv
import os
from typing import Any, Dict, Optional


def get_loose_nodes(stoichiometry):
    # get all needed exchanges
    nodes, edges = make_network(stoichiometry)

    sources = set()
    targets = set()
    for edge in edges:
        source = edge[0]
        target = edge[1]
        sources.add(source)
        targets.add(target)

    loose_nodes = sources.symmetric_difference(targets)

    return loose_nodes


def get_reactions(stoichiometry, molecules):
    '''
    for each entry in molecules (list), return all the reactions with the molecules coefficient
    '''
    reactions = {}
    for reaction_id, stoich in stoichiometry.items():
        mol_coeffs = {mol_id: coeff
                      for mol_id, coeff in stoich.items()
                      if mol_id in molecules}
        reactions[reaction_id] = mol_coeffs

    return reactions


def make_network(stoichiometry, info: Optional[Dict[str, Any]] = None):
    '''
    Makes a gephi network
    info can contain node_sizes, node_types

    .. code-block:: python

        info = {
            'node_sizes': node_sizes (dict),
            'node_types': node_types (dict)
        }
    '''

    info = info or {}
    node_types = info.get('node_types', {})
    node_sizes = info.get('node_sizes', {})
    reaction_fluxes = info.get('reaction_fluxes', {})

    nodes = {}
    edges = []
    for reaction_id, stoich in stoichiometry.items():
        flux = reaction_fluxes.get(reaction_id, 1)
        # add reaction to node list
        n_type = node_types.get(reaction_id, 'reaction')
        n_size = node_sizes.get(reaction_id, 1)
        nodes[reaction_id] = {
            'label': reaction_id,
            'type': n_type,
            'size': n_size}

        # add molecules to node list, and connections to edge list
        for molecule_id, coeff in stoich.items():
            n_type = node_types.get(molecule_id, 'molecule')
            n_size = node_sizes.get(molecule_id, 1)
            nodes[molecule_id] = {
                'label': molecule_id,
                'type': n_type,
                'size': n_size}

            # add edge between reaction and molecule
            # a reactant
            if coeff < 0:
                edge = [molecule_id, reaction_id, flux]
            # a product
            elif coeff > 0:
                edge = [reaction_id, molecule_id, flux]
            else:
                print(reaction_id + ', ' + molecule_id + ': coeff = 0')
                break
            edges.append(edge)

    return nodes, edges


def collapse_network(nodes, edges, remove_nodes_list):
    ''' remove_nodes (list) -- nodes to be removed '''
    new_nodes = nodes.copy()

    remove_edges = []
    add_edges = []
    for node_id in remove_nodes_list:
        # remove node from from new_nodes
        new_nodes.pop(node_id, None)

        # remove all edges from and to this node_id
        from_edges = [[from_node, to_node] for [from_node, to_node] in edges if to_node is node_id]
        to_edges = [[from_node, to_node] for [from_node, to_node] in edges if from_node is node_id]
        remove_edges.extend(from_edges)
        remove_edges.extend(to_edges)

        # connect the severed nodes
        for [from_node, c_node1] in from_edges:
            make_edges = [[from_node, to_node] for [c_node2, to_node] in to_edges]
            add_edges.extend(make_edges)

    # make new edges
    new_edges = [edge for edge in edges if edge not in remove_edges]
    new_edges.extend(add_edges)

    # remove redundant new edges
    new_edges2 = []
    for edge in new_edges:
        if edge not in new_edges2:
            new_edges2.append(edge)

    return new_nodes, new_edges2


# TODO(jerry): Can we rename plotOutDir per PEP8 and update callers?
# noinspection PyPep8Naming
def save_network(nodes, edges, plotOutDir='out/network'):
    '''
    Save nodes and edges

    Requires:
        nodes (dict) with {node_id: {'label' (str), 'type' (str), size (float)}}
        edges (list) with [[node_id1, node_id2] ...]
    '''

    out_dir = os.path.join(plotOutDir)
    os.makedirs(out_dir, exist_ok=True)
    nodes_out = os.path.join(out_dir, 'nodes.csv')
    edges_out = os.path.join(out_dir, 'edges.csv')

    # Save network to csv
    # nodes list
    with open(nodes_out, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')

        # write header
        writer.writerow(['Id', 'Label', 'Type', 'Size'])

        for node, specs in nodes.items():
            label = specs['label']
            type_ = specs['type']
            size = specs['size']

            row = [node, label, type_, size]
            writer.writerow(row)

    # edges list
    with open(edges_out, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')

        # write header
        writer.writerow(['Source', 'Target', 'Weight'])

        for edge in edges:
            source = edge[0]
            target = edge[1]
            weight = edge[2]

            row = [source, target, weight]
            writer.writerow(row)
