import pandas as pd
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt

filepath = r'C:\Users\parim\OneDrive\Documents\Artificial Intelligence LAB\ASSIGNMENT_1\edges.csv'
edges_df = pd.read_csv(filepath)

graph = defaultdict(list)
for _, row in edges_df.iterrows():
    graph[row['Source']].append(row['Target'])
    graph[row['Target']].append(row['Source'])

def dfs_recursive(graph, node, visited, parent=None, edges=[]):
    if node not in visited:
        visited.add(node)
        if parent is not None:
            edges.append((parent, node))
        for neighbor in graph[node]:
            dfs_recursive(graph, neighbor, visited, node, edges)
    return edges

def dfs_non_recursive(graph, start):
    visited = set()
    stack = [start]
    edges = []

    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    edges.append((node, neighbor))
            stack.extend(graph[node][::-1])
    return edges

def visualize_dfs_tree(edges, title):
    G = nx.DiGraph()
    G.add_edges_from(edges)
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightgreen', node_size=500, font_size=10, font_weight='bold', edge_color='blue')
    plt.title(title)
    plt.show()

def visualize_graph(edges_df):
    G = nx.Graph()
    for _, row in edges_df.iterrows():
        G.add_edge(row['Source'], row['Target'])
    plt.figure(figsize=(10, 8))
    nx.draw(G, with_labels=True, node_color='lightblue', node_size=500, font_size=10, font_weight='bold', edge_color='gray')
    plt.title("Graph Visualization")
    plt.show()

if __name__ == "__main__":
    visualize_graph(edges_df)
    visited = set()
    dfs_tree_recursive = dfs_recursive(graph, list(graph.keys())[5], visited)
    print("\nDFS Recursive Tree Edges:", dfs_tree_recursive)
    visualize_dfs_tree(dfs_tree_recursive, "DFS Recursive Tree")
    dfs_tree_non_recursive = dfs_non_recursive(graph, list(graph.keys())[5])
    print("\nDFS Non-Recursive Tree Edges:", dfs_tree_non_recursive)
