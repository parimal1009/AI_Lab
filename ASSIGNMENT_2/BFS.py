import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque

# Load the edge list from CSV
filepath = r"C:\Users\parim\OneDrive\Documents\Artificial Intelligence LAB\ASSIGNMENT_2\edges.csv"
edges_df = pd.read_csv(filepath)

# Construct an undirected weighted graph
graph = nx.Graph()
for _, row in edges_df.iterrows():
    graph.add_edge(row['Source'], row['Target'], weight=row['Weight'])

# BFS traversal to construct BFS tree
def bfs_tree(graph, start):
    visited = set()
    queue = deque([start])
    bfs_edges = []
    
    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            for neighbor in graph.neighbors(node):
                if neighbor not in visited and neighbor not in queue:
                    weight = graph[node][neighbor]['weight']
                    bfs_edges.append((node, neighbor, weight))
                    queue.append(neighbor)
    return bfs_edges

# Visualize the input graph
def visualize_graph(graph, title="Weighted Graph"):
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(graph)  
    nx.draw(graph, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10, font_weight='bold', edge_color='gray')
    
    # Draw edge labels (weights)
    edge_labels = {(u, v): d["weight"] for u, v, d in graph.edges(data=True)}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=10, font_color="red")
    
    plt.title(title)
    plt.show()

# Visualize the BFS tree
def visualize_bfs_tree(bfs_edges, title="BFS Tree"):
    bfs_tree = nx.Graph()
    bfs_tree.add_weighted_edges_from(bfs_edges)
    
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(bfs_tree)  
    nx.draw(bfs_tree, pos, with_labels=True, node_color='lightgreen', node_size=500, font_size=10, font_weight='bold', edge_color='blue')
    
    # Draw edge labels (weights)
    edge_labels = {(u, v): w for u, v, w in bfs_edges}
    nx.draw_networkx_edge_labels(bfs_tree, pos, edge_labels=edge_labels, font_size=10, font_color="red")
    
    plt.title(title)
    plt.show()

# Run the program
if __name__ == "__main__":
    visualize_graph(graph, "Input Weighted Graph")
    start_node = list(graph.nodes())[0]   
      
    bfs_edges = bfs_tree(graph, start_node)
    
    print("\nBFS Tree Edges:", bfs_edges)
    visualize_bfs_tree(bfs_edges, "BFS Tree")
