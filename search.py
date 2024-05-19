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
        traffic_factor = sum(G[s][e]['traffic'] for s, e in zip(path[:-1], path[1:])) / len(path)
        urgency_factor = sum(G[s][e]['urgency'] for s, e in zip(path[:-1], path[1:])) / len(path)
        
        # Simple model to integrate traffic and urgency
        adjusted_length = path_length * (1 + 0.1 * traffic_factor - 0.1 * (6 - urgency_factor))
        return adjusted_length # returns the adjusted length
    except nx.NetworkXNoPath:
        return float('inf') # If no path exists, return infinity

# A function to perform an A* search to find the optimal path from start to goal.
def a_star_search(graph, start, goal):
    frontier = [] # Priority queue to store nodes to explore
    heappush(frontier, (0, start)) # Adds the start node to the frontier
    came_from = {} # Dictionary to store the path
    cost_so_far = defaultdict(lambda: float('inf')) # Dictionary to store the cost from start to each node
    cost_so_far[start] = 0 # Cost from start to start is set to 0

    #  Iterates through the frontier to find the optimal path
    while frontier:
        _, current = heappop(frontier) # Removes the node with the lowest cost and stores it as current

        # If the goal is reached, break the loop
        if current == goal:
            break
        
        # Iterates through the neighbors of the current node
        for neighbor in graph.neighbors(current):
            new_cost = cost_so_far[current] + graph[current][neighbor]['distance'] # Calculates the new cost
            # If the new cost is lower than the previous cost, update the cost and add the neighbor to the frontier
            if new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost # Updates the cost
                priority = new_cost + heuristic(neighbor, goal, graph) # Calculates the priority using the heuristic
                heappush(frontier, (priority, neighbor)) # Adds the neighbor to the frontier
                came_from[neighbor] = current # Updates the path

    # If the goal is reached, reconstruct the path and return it
    if current == goal:
        path = [] # List to store the path
        total_distance = 0 # Variable to store the total distance
        # Reconstructs the path and calculates the total distance while iterating from the goal to the start
        while current != start:
            path.append(current) # Adds the current node to the path
            next_node = came_from[current] # Updates the current node
            edge_data = graph[next_node][current] # Gets the edge data
            total_distance += edge_data['distance'] # Updates the total distance
            current = next_node # Updates the current node
        path.append(start) # Adds the start node to the path
        path.reverse() # Reverses the path to start from the start node
        return path, total_distance # Returns the path and the total distance
    else:
        return None  # No path found

# A function to plot the graph with the path highlighted
def plot_graph(G, path=None, total_distance=None):
    pos = nx.spring_layout(G)  # positions for all nodes

    # Nodes
    nx.draw_networkx_nodes(G, pos, node_size=700)

    # Edges
    nx.draw_networkx_edges(G, pos, width=1) # Draw all edges
    edge_labels = {(s, e): d['distance'] for s, e, d in G.edges(data=True)} # Distance labels
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels) # Draw distance labels

    # Highlight the path
    if path:
        path_edges = list(zip(path[:-1], path[1:])) # Edges in the optimal path
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, width=3, edge_color='r') # Draw the path in red

    # Labels
    nx.draw_networkx_labels(G, pos, font_size=20)

    # Title & Display
    plt.title(f"Total Distance Traveled: {total_distance}" if total_distance else "Graph without path")
    plt.axis('off')  # Turn off the axis
    plt.show()  # Display the graph

# A function to print the details of the graph
def print_graph_details(G):

    # Print node details and connections
    for node in G.nodes():
        print(f"Node {node} connections:")
        for neighbor in G.neighbors(node):
            edge_data = G[node][neighbor]
            print(f"  -> To Node {neighbor} | Distance: {edge_data['distance']} | Traffic: {edge_data['traffic']} | Urgency: {edge_data['urgency']}")

# Create the graph with 30% connectivity
G = create_graph(0.3)

# Print node and edge details for extra, optimal detail
print_graph_details(G)

# Get the start and goal nodes from the user
start_node = input("Enter the start node (A-Z): ").upper()
goal_node = input("Enter the goal node (A-Z): ").upper()

# Perform A* search to find the optimal path
result = a_star_search(G, start_node, goal_node)

# If a path is found, print the path and total distance and plot the graph
if result:
    path, total_distance = result
    print("Path from start to goal:", path)
    print(f"Total distance: {total_distance}")
    plot_graph(G, path, total_distance)
else:
    print("No path found between start and goal.")
