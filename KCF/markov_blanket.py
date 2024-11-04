import glob
import os

import networkx as nx
import pandas as pd
import plotly.graph_objs as go


def markov_blanket(graph, node):
    """
    Calculate the Markov blanket of a specified node in a directed graph.

    The Markov blanket of a node includes its parents, children, and the parents of its children.

    Args:
        graph (networkx.DiGraph): A directed graph representing the relationships between nodes.
        node (str): The node for which the Markov blanket is to be calculated.

    Raises:
        ValueError: If the specified node does not exist in the graph.

    Returns:
        set: A set containing the nodes in the Markov blanket of the specified node.
    """
    if node not in graph:
        raise ValueError(f"Węzeł {node} nie istnieje w grafie.")

    parents = set(graph.predecessors(node))
    children = set(graph.successors(node))

    parents_of_children = set()
    for child in children:
        parents_of_children.update(graph.predecessors(child))

    markov_blanket_set = parents.union(children).union(parents_of_children)
    markov_blanket_set.discard(node)

    return markov_blanket_set


def bfs_levels(graph, start):
    """
    Perform a breadth-first search (BFS) on the graph to determine the levels of nodes.

    This function assigns a level to each node, where the starting node has level 0,
    its direct successors have level 1, and so on.

    Args:
        graph (networkx.DiGraph): A directed graph to traverse.
        start (str): The node from which to start the BFS.

    Returns:
        dict: A dictionary mapping each node to its level in the graph.
    """
    levels = {}
    queue = [(start, 0)]
    visited = set()
    while queue:
        node, level = queue.pop(0)
        if node not in visited:
            visited.add(node)
            levels[node] = level
            for neighbor in graph.successors(node):
                queue.append((neighbor, level + 1))
    return levels


def draw_graph(graph, save_path, directed_graph, node_markov, mbm):
    """
    Draw the graph and save it as an image file.

    The graph will be visualized with nodes colored differently based on whether they are
    the specified Markov node or part of its Markov blanket.

    Args:
        graph (networkx.DiGraph): The graph to be drawn.
        save_path (str): The file path where the image will be saved.
        directed_graph (bool): Indicates whether the graph is directed.
        node_markov (str): The node for which the Markov blanket is being visualized.
        mbm (set): The Markov blanket of the specified node.

    Returns:
        None
    """
    if directed_graph:
        levels = bfs_levels(graph, node_markov)
        max_level = max(levels.values())
        level_count = {i: 0 for i in range(max_level + 1)}

        for level in levels.values():
            level_count[level] += 1

        pos = {}
        x_gap = 1
        y_gap = 1

        for level in range(max_level + 1):
            num_nodes = level_count[level]
            x_positions = [x_gap * (i - num_nodes / 2) for i in range(num_nodes)]
            nodes_at_level = [node for node in levels if levels[node] == level]

            for i, node in enumerate(nodes_at_level):
                pos[node] = (x_positions[i], -y_gap * level)

        for node in graph.nodes:
            if node in pos:
                graph.nodes[node]["pos"] = pos[node]
            else:
                graph.nodes[node]["pos"] = (5, 0)

    edge_x = []
    edge_y = []
    edge_weight_text = []
    edge_weight_text_pos_x = []
    edge_weight_text_pos_y = []

    for edge in graph.edges(data="weight"):
        x0, y0 = graph.nodes[edge[0]]["pos"]
        x1, y1 = graph.nodes[edge[1]]["pos"]
        edge_weight_text_pos_x.append((x0 + x1) / 2)
        edge_weight_text_pos_y.append((y0 + y1) / 2)
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
        if edge[2] is None:
            edge_weight_text.append(None)
        else:
            edge_weight_text.append(round(edge[2], 4))

    node_x = []
    node_y = []
    node_text = []
    color = []
    for node in graph.nodes():
        x, y = graph.nodes[node]["pos"]
        node_x.append(x)
        node_y.append(y)
        node_text.append(str(node))
        if node == node_markov:
            color.append("red")
        elif node in mbm:
            color.append("yellow")
        else:
            color.append("lightblue")

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        textposition="bottom center",
        hoverinfo="text",
        text=node_text,
        marker=dict(showscale=False, size=20, line_width=2, color=color),
    )

    data = []
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="text",
        mode="lines",
        text=edge_weight_text,
    )

    edge_labels_trace = go.Scatter(
        x=edge_weight_text_pos_x,
        y=edge_weight_text_pos_y,
        mode="text",
        text=edge_weight_text,
        hoverinfo="none",
    )

    data = [edge_trace, node_trace, edge_labels_trace]

    fig = go.Figure(
        data=data,
        layout=go.Layout(
            width=1200,
            height=1000,
            showlegend=False,
            hovermode="closest",
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
        ),
    )

    fig.write_image(save_path)


def load_graph_from_adjlist(file_path):
    """
    Load a directed graph from an adjacency list file.

    The graph is created using the networkx library.

    Args:
        file_path (str): The file path to the adjacency list.

    Returns:
        networkx.DiGraph: The directed graph loaded from the adjacency list.
    """
    G = nx.read_adjlist(file_path, nodetype=str, create_using=nx.DiGraph())
    return G


def main():
    """
    Main function to execute the graph processing tasks.

    This function loads a graph from an adjacency list file, calculates the Markov blanket
    for a specified node, and draws the graph with the Markov blanket highlighted.

    Returns:
        None
    """
    file_path = "C:/Users/olkab/Desktop/Magisterka/Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/KCF/no_lack_of_diagnosis_TBG_10_fold_vis/graphs/fold 9/root_X1_node_X30_k_2.adjlist"

    os.makedirs(
        "C:/Users/olkab/Desktop/Magisterka/Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/KCF/Markov Blanket",
        exist_ok=True,
    )

    G = load_graph_from_adjlist(file_path)
    os.makedirs(
        "C:/Users/olkab/Desktop/Magisterka/Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/KCF/Markov Blanket/",
        exist_ok=True,
    )

    node = f"X1"
    mb = markov_blanket(G, node)
    with open(
        "C:/Users/olkab/Desktop/Magisterka/Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/KCF/Markov Blanket/Markov_Blankey.txt",
        "a",
        encoding="utf-8",
    ) as f:
        f.write(f"Otulina Markowa dla węzła {node}: {mb}\n")
        print(f"Otulina Markowa dla węzła {node}: {mb}")

    draw_graph(
        G,
        "C:/Users/olkab/Desktop/Magisterka/Atificial-intelligence-algorithms-for-the-prediction-and-classification-of-thyroid-diseases/KCF/Markov Blanket/X1.png",
        True,
        node,
        mb,
    )
