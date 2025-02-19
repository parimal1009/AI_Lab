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

# A* Search Algorithm
def a_star_search(graph, start, goal):
    def heuristic(node):
        return 0  # Default heuristic (Dijkstra-like behavior). Can be customized.

    visited = set()
    pq = []  # Min-heap priority queue
    best_edges = []

    # (f-score, g-score, node, parent)
    heapq.heappush(pq, (0, 0, start, None))
    g_score = {node: float('inf') for node in graph.nodes}
    g_score[start] = 0

    while pq:
        _, g, node, parent = heapq.heappop(pq)

        if node in visited:
            continue

        visited.add(node)
        if parent is not None:
            best_edges.append((parent, node, g))

        if node == goal:
            break  # Stop when goal is reached

        for neighbor in graph.successors(node):
            edge_weight = graph[node][neighbor]['weight']
            tentative_g_score = g + edge_weight

            if tentative_g_score < g_score[neighbor]:
                g_score[neighbor] = tentative_g_score
                f_score = tentative_g_score + heuristic(neighbor)
                heapq.heappush(pq, (f_score, tentative_g_score, neighbor, node))

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

# Visualize the A* Search Tree
def visualize_astar_tree(best_edges, title="A* Search Tree"):
    astar_tree = nx.DiGraph()
    astar_tree.add_weighted_edges_from(best_edges)
    
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(astar_tree)
    nx.draw(astar_tree, pos, with_labels=True, node_color='lightgreen', node_size=500, font_size=10, font_weight='bold', edge_color='blue', arrows=True)
    
    edge_labels = {(u, v): w for u, v, w in best_edges}
    nx.draw_networkx_edge_labels(astar_tree, pos, edge_labels=edge_labels, font_size=10, font_color="red")
    
    plt.title(title)
    plt.show()

# Run the program
if __name__ == "__main__":
    visualize_graph(graph, "Input Weighted Directed Graph")

    start_node = list(graph.nodes())[0]  # Selecting an arbitrary start node
    goal_node = list(graph.nodes())[-1]  # Selecting an arbitrary goal node

    best_edges = a_star_search(graph, start_node, goal_node)

    print("\nA* Search Tree Edges:", best_edges)
    visualize_astar_tree(best_edges, "A* Search Tree")
