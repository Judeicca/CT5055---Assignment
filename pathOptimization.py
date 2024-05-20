import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import string
import random

# Constants
ALPHA = 1.0 # Influence of pheromone
BETA = 5.0 # Influence of heuristic (inverse of cost)
RHO = 0.5 # Pheromone evaporation rate
Q = 100 # Pheromone deposit factor
NUM_ANTS = 10 # Number of ants
NUM_ITERATIONS = 100 # Number of iterations
INIT_PHEROMONE = 1.0 # Initial pheromone level
CONVERGENCE_THRESHOLD = 1e-5 # Improvement threshold
CONVERGENCE_ITERATIONS = 10  # Number of iterations without improvement to consider convergence

# A Function to Generate a Graph with Random Weights and Traffic Conditions
def generate_graph(num_nodes):
    alphabet = string.ascii_uppercase 
    graph = {letter: {} for letter in alphabet[:num_nodes]}

    # Initialize edges with random distance and traffic conditions
    for node in graph:
        for neighbour in graph:
            if node != neighbour:
                distance = random.randint(1, 10) # Random distance between 1 and 10 for distance
                traffic = random.randint(1, 5) # Random number from 1 to 5 for traffic conditions
                urgency = random.randint(1, 5) # Random number from 1 to 5 for delivery urgency
                cost = distance + (traffic - 1) + (urgency - 1) # Calculate cost based on distance, traffic, and urgency
                graph[node][neighbour] = cost
    return graph

# A Function to Choose the Next Node Based on Pheromones and Heuristic Information
def choose_next_node(current_node, allowable_nodes, pheromones, graph):
    # Calculate probabilities based on pheromones and heuristic information
    pheromone_list = np.array([pheromones[current_node][j] 
                               ** ALPHA * (1.0 / graph[current_node][j]) ** BETA for j in allowable_nodes])
    prob_list = pheromone_list / pheromone_list.sum() # Normalize probabilities
    return np.random.choice(allowable_nodes, 1, p=prob_list)[0] # Choose the next node based on probabilities

# A Function to Update Pheromones Based on Ants' Paths
def update_pheromones(pheromones, ants, graph):
    # For each edge, evaporate pheromones and deposit new pheromones
    for i in pheromones:
        for j in pheromones[i]:
            pheromones[i][j] *= (1 - RHO)  # Evaporation of pheromones
    for path, cost in ants:
        for i, j in zip(path[:-1], path[1:]):
            pheromones[i][j] += Q / cost  # Deposit pheromones

# A Function to Plot the Graph with the Best Path Found by ACO
def plot_graph(graph, best_path):
    G = nx.DiGraph() # Create a directed graph
    G.add_nodes_from(graph.keys()) # Add nodes to the graph

    # For each node, add edges to neighbours with weights
    for node, edges in graph.items():
        for neighbour, weight in edges.items():
            G.add_edge(node, neighbour, weight=weight)
    pos = nx.kamada_kawai_layout(G) # Position nodes using Kamada-Kawai layout
    labels = {} # Create a dictionary to store edge labels

    # For each node, add the weight of the edge to the label dictionary
    for node in graph:
        for neighbour, weight in graph[node].items():
            labels[(node, neighbour)] = weight

    path_edges = list(zip(best_path[:-1], best_path[1:])) # Extract edges for the best path
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=15, font_weight='bold') # Draw the graph
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2) # Highlight the best path in red
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels) # Draw the edge labels
    plt.title('Best Path Found') 
    plt.show()

# A Function to Implement the Ant Colony Optimisation Algorithm
def aco_algorithm(graph, pheromones):
    best_cost = float('inf') # Initialize best cost to infinity
    best_path = None # Initialize best path to None
    prev_best_cost = float('inf') # Initialize previous best cost to infinity
    no_improvement_count = 0 # Counter for no improvement
    convergence_counter = 0  # Counter for convergence criteria

    for _ in range(NUM_ITERATIONS):
        ants = []
        # For each ant, construct a path based on pheromones and heuristics
        for _ in range(NUM_ANTS):
            path = ['A'] # Start at node 'A'
            current_node = 'A' # Initialize current node to 'A'
            visited = set(path) # Add 'A' to visited nodes

            # While there are unvisited nodes, choose the next node
            while len(visited) < len(graph):
                next_node = choose_next_node(current_node, [n for n in graph[current_node] if n not in visited], pheromones, graph)
                path.append(next_node) 
                visited.add(next_node)
                current_node = next_node
            cost = sum(graph[path[i]][path[i + 1]] for i in range(len(path) - 1)) # Calculate the cost of the path
            ants.append((path, cost)) # Add the path and cost to the list of ants
            # Update the best path if a shorter path is found
            if cost < best_cost:
                best_cost = cost
                best_path = path
                convergence_counter = 0  # Reset convergence counter if a better solution is found
        update_pheromones(pheromones, ants, graph) # Update pheromones based on ant paths
        
        # Check for convergence based on improvement threshold
        if abs(prev_best_cost - best_cost) < CONVERGENCE_THRESHOLD:
            no_improvement_count += 1
        else:
            no_improvement_count = 0
        if no_improvement_count >= CONVERGENCE_ITERATIONS:
            break
        
        prev_best_cost = best_cost
        # Check for convergence criteria based on number of iterations without improvement
        if convergence_counter > 10:  # If no improvement for 10 iterations, break
            break  # Terminate if convergence criteria met
        convergence_counter += 1
    return best_path

# Generate graph with desired number of nodes
num_nodes = 10  # Number of nodes in the graph
graph = generate_graph(num_nodes)

# Initialize pheromones
pheromones = {i: {j: INIT_PHEROMONE for j in graph[i]} for i in graph} 

# Run the Ant Colony Optimization algorithm
best_path_found = aco_algorithm(graph, pheromones)
print("Best path:", best_path_found)
plot_graph(graph, best_path_found)
