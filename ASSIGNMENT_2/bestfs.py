import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import heapq

# Load the edge list from CSV
filepath = r"C:\Users\parim\OneDrive\Documents\Artificial Intelligence LAB\ASSIGNMENT_2\edges.csv"
edges_df = pd.read_csv(filepath)

# Construct a directed weighted graph
graph = nx.DiGraph()
for _, row in edges_df.iterrows():
    graph.add_edge(row['Source'], row['Target'], weight=row['Weight'])

# Best First Search (BFS with priority queue)
def best_first_search(graph, start):
    visited = set()
    pq = []  # Min-heap priority queue
    best_edges = []
    
    heapq.heappush(pq, (0, start, None))  # (weight, node, parent)
    
    while pq:
        weight, node, parent = heapq.heappop(pq)
        
        if node not in visited:
            visited.add(node)
            if parent is not None:
                best_edges.append((parent, node, weight))
                
            for neighbor in graph.successors(node):
                if neighbor not in visited:
                    heapq.heappush(pq, (graph[node][neighbor]['weight'], neighbor, node))
    
    return best_edges

# Visualize the input graph
def visualize_graph(graph, title="Weighted Directed Graph"):
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10, font_weight='bold', edge_color='gray', arrows=True)
    
    edge_labels = {(u, v): d["weight"] for u, v, d in graph.edges(data=True)}
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_size=10, font_color="red")
    
    plt.title(title)
    plt.show()

# Visualize the Best First Search Tree
def visualize_best_first_tree(best_edges, title="Best First Search Tree"):
    best_tree = nx.DiGraph()
    best_tree.add_weighted_edges_from(best_edges)
    
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(best_tree)
    nx.draw(best_tree, pos, with_labels=True, node_color='lightgreen', node_size=500, font_size=10, font_weight='bold', edge_color='blue', arrows=True)
    
    edge_labels = {(u, v): w for u, v, w in best_edges}
    nx.draw_networkx_edge_labels(best_tree, pos, edge_labels=edge_labels, font_size=10, font_color="red")
    
    plt.title(title)
    plt.show()

# Run the program
if __name__ == "__main__":
    visualize_graph(graph, "Input Weighted Directed Graph")
    start_node = list(graph.nodes())[0]  # Selecting an arbitrary start node
    
    best_edges = best_first_search(graph, start_node)
    
    print("\nBest First Search Tree Edges:", best_edges)
    visualize_best_first_tree(best_edges, "Best First Search Tree")
