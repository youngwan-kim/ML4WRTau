import torch
import random
import os
import argparse
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx

def guess_node_type(x, node_type=None):
    type_map = {
        0: "muon",
        1: "electron",
        2: "tau",
        3: "jet",
        4: "fatjet"
    }

    if node_type is not None:
        return type_map.get(int(node_type), "unknown")

    if len(x) == 0:
        return "unknown"
    if x[0] > 100:   
        return "fatjet"
    return "jet"

def visualize_graph(graph: Data, title="Graph"):
    G = to_networkx(graph, to_undirected=True)
    pos = nx.spring_layout(G, seed=42)

    colors = []
    sizes = []
    for i, x in enumerate(graph.x):
        pt = x[0].item()
        node_type = graph.node_type[i] if hasattr(graph, "node_type") else None
        obj_type = guess_node_type(x, node_type)
        if obj_type == "fatjet":
            colors.append("orange")
        elif obj_type == "jet":
            colors.append("blue")
        elif obj_type == "muon":
            colors.append("green")
        elif obj_type == "electron":
            colors.append("red")
        elif obj_type == "tau":
            colors.append("purple")
        else:
            colors.append("gray")
        sizes.append(100 + pt * 2) 

    edge_labels = {}
    if hasattr(graph, "edge_attr") and graph.edge_attr is not None:
        for i, (u, v) in enumerate(G.edges()):
            dr = graph.edge_attr[i].item()
            edge_labels[(u, v)] = f"{dr:.2f}" 

    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color=colors, node_size=sizes, font_size=10)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color="gray")
    legend_labels = {'jet': 'blue', 'muon': 'green', 'electron': 'red', 'fatjet':'orange','tau':'purple'}
    for label, color in legend_labels.items():
        plt.plot([], [], marker='o', label=label, color=color, linestyle='')
    plt.title(title)
    plt.axis("off")
    plt.legend()
    plt.show()
    plt.savefig("debug.png")

def main(pt_file):
    print(f"Loading: {pt_file}")
    data_list = torch.load(pt_file, weights_only=False)

    if not data_list:
        print("Empty file.")
        return

    print(f"{len(data_list)} graphs found.")
    graph = random.choice(data_list)
    visualize_graph(graph, title=os.path.basename(pt_file))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pt_file", help="Path to .pt file containing graphs")
    args = parser.parse_args()
    main(args.pt_file)
