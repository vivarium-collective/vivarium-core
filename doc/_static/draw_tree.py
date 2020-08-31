import networkx as nx
from matplotlib import pyplot as plt


def main():
    G = nx.Graph()
    edges = [
        ('root', 'cell'),
        ('root', 'global'),
        ('root', 'injector'),
        ('root', 'glucose_phosphorylation'),
        ('root', 'my_deriver'),
        ('cell', 'ATP'),
        ('cell', 'ADP'),
        ('cell', 'G6P'),
        ('cell', 'GLC'),
        ('cell', 'HK'),
        ('global', 'initial_mass'),
        ('global', 'mass'),
    ]
    fig, ax = plt.subplots(figsize=(15, 5))
    G.add_edges_from(edges)
    nx.draw_networkx(
        G,
        pos=nx.nx_pydot.graphviz_layout(G, prog="dot"),
        with_labels=True,
        node_size=2000,
        font_size=14,
        ax=ax
    )
    plt.savefig('tree.png')


if __name__ == '__main__':
    main()
