#import the required libraries
import random
import networkx as nx
import matplotlib.pyplot as plt
from heapq import heappop, heappush
from collections import defaultdict

# A function to create a graph with 26 nodes (A-Z) and adjustable connectivity
def create_graph(connectivity):
    G = nx.Graph() # Creates an empty graph
    nodes = [chr(i) for i in range(65, 91)] # Creates nodes using ASCII values for A to Z
    G.add_nodes_from(nodes) # adds nodes to the graph

    # Adds costs and edges to nodes
    for i in range(26):
        for j in range(i + 1, 26):
            if random.random() < connectivity:
                distance = random.randint(1, 100)
                traffic = random.randint(1, 5)
                urgency = random.randint(1, 5)
                # Adds edges to the graph and assigns costs to the nodes
                G.add_edge(nodes[i], nodes[j], distance=distance, traffic=traffic, urgency=urgency)
    
    return G # returns the graph

# A function that estimates the cost from the start to the goal nodes.
def heuristic(start, goal, G):
    try:
        # Calculate the shortest path considering only the distance
        path_length = nx.shortest_path_length(G, source=start, target=goal, weight='distance')
        
        # Adjust the path length based on average traffic and urgency along the path
        path = nx.shortest_path(G, source=start, target=goal, weight='distance')
        traffic_factor = sum(G[u][v]['traffic'] for u, v in zip(path[:-1], path[1:])) / len(path)
        urgency_factor = sum(G[u][v]['urgency'] for u, v in zip(path[:-1], path[1:])) / len(path)
        
        # Simple model to integrate traffic and urgency
        adjusted_length = path_length * (1 + 0.1 * traffic_factor - 0.1 * (6 - urgency_factor))
        return adjusted_length # returns the adjusted length
    except nx.NetworkXNoPath:
        return float('inf') # If no path exists, return infinity


def a_star_search(graph, start, goal):
    # Perform A* search to find the optimal path from start to goal.
    # Combines the actual path cost and heuristic estimate to guide the search.
    frontier = []
    heappush(frontier, (0, start))
    came_from = {}
    cost_so_far = defaultdict(lambda: float('inf'))
    cost_so_far[start] = 0

    while frontier:
        _, current = heappop(frontier)

        if current == goal:
            break

        for neighbor in graph.neighbors(current):
            new_cost = cost_so_far[current] + graph[current][neighbor]['distance']
            if new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic(neighbor, goal, graph)
                heappush(frontier, (priority, neighbor))
                came_from[neighbor] = current

    # Reconstruct the path
    if current == goal:
        path = []
        total_distance = 0
        while current != start:
            path.append(current)
            next_node = came_from[current]
            edge_data = graph[next_node][current]
            total_distance += edge_data['distance']
            current = next_node
        path.append(start)
        path.reverse()
        return path, total_distance
    else:
        return None  # No path found

def plot_graph(G, path=None, total_distance=None):
    # Plot the graph with nodes and edges. Highlight the path if provided.
    pos = nx.spring_layout(G)  # positions for all nodes

    # Nodes
    nx.draw_networkx_nodes(G, pos, node_size=700)

    # Edges
    nx.draw_networkx_edges(G, pos, width=1)
    edge_labels = {(u, v): d['distance'] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    if path:
        path_edges = list(zip(path[:-1], path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=3, edge_color='r')

    # Labels
    nx.draw_networkx_labels(G, pos, font_size=20, font_family='sans-serif')

    plt.title(f"Total Distance Traveled: {total_distance}" if total_distance else "Graph without path")
    plt.axis('off')  # Turn off the axis
    plt.show()  # Display the graph

def print_graph_details(G):
    # Print details of each node and its edges.
    for node in G.nodes():
        print(f"Node {node} connections:")
        for neighbor in G.neighbors(node):
            edge_data = G[node][neighbor]
            print(f"  -> To Node {neighbor} | Distance: {edge_data['distance']} | Traffic: {edge_data['traffic']} | Urgency: {edge_data['urgency']}")

# Create the graph with 26 alphabet nodes and 30% connectivity
G = create_graph(0.3)

# Print node and edge details
print_graph_details(G)

# Example usage: find path from 'A' to 'Z'
start_node = input("Enter the start node (A-Z): ").upper()
goal_node = input("Enter the goal node (A-Z): ").upper()
result = a_star_search(G, start_node, goal_node)

# Plot the graph with the path highlighted
if result:
    path, total_distance = result
    print("Path from start to goal:", path)
    print(f"Total distance: {total_distance}")
    plot_graph(G, path, total_distance)
else:
    print("No path found between start and goal.")
