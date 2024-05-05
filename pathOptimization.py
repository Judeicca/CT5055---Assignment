import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import string
import random

# Constants
ALPHA = 1.0  # Influence of pheromone
BETA = 5.0   # Influence of heuristic (inverse of cost)
RHO = 0.5    # Pheromone evaporation rate
Q = 100      # Pheromone deposit factor
NUM_ANTS = 10
NUM_ITERATIONS = 100
INIT_PHEROMONE = 1.0

alphabet = string.ascii_uppercase
graph = {letter: {} for letter in alphabet}

# Initialize edges with random weights
for node in graph:
    for neighbour in graph:
        if node != neighbour:
            graph[node][neighbour] = random.randint(1, 10)  # Random weight between 1 and 10

# Initialize pheromones
pheromones = {i: {j: INIT_PHEROMONE for j in graph[i]} for i in graph}

def choose_next_node(current_node, allowable_nodes, pheromones, graph):
    pheromone_list = np.array([pheromones[current_node][j] ** ALPHA * (1.0 / graph[current_node][j]) ** BETA for j in allowable_nodes])
    prob_list = pheromone_list / pheromone_list.sum()
    return np.random.choice(allowable_nodes, 1, p=prob_list)[0]

def update_pheromones(pheromones, ants, graph):
    for i in pheromones:
        for j in pheromones[i]:
            pheromones[i][j] *= (1 - RHO)  # Evaporation
    for path, cost in ants:
        for i, j in zip(path[:-1], path[1:]):
            pheromones[i][j] += Q / cost  # Deposit

def plot_graph(graph, best_path):
    G = nx.DiGraph()
    G.add_nodes_from(graph.keys())
    for node, edges in graph.items():
        for neighbour, weight in edges.items():
            G.add_edge(node, neighbour, weight=weight)
    pos = nx.spring_layout(G)  # Automatically position nodes
    labels = {}
    for node in graph:
        for neighbour, weight in graph[node].items():
            labels[(node, neighbour)] = weight

    path_edges = list(zip(best_path[:-1], best_path[1:]))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=15, font_weight='bold')
    nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.title('Best Path Found by ACO')
    plt.show()

def aco_algorithm(graph, pheromones):
    best_cost = float('inf')
    best_path = None
    for _ in range(NUM_ITERATIONS):
        ants = []
        for _ in range(NUM_ANTS):
            path = ['A']
            current_node = 'A'
            visited = set(path)
            while len(visited) < len(graph):
                next_node = choose_next_node(current_node, [n for n in graph[current_node] if n not in visited], pheromones, graph)
                path.append(next_node)
                visited.add(next_node)
                current_node = next_node
            cost = sum(graph[path[i]][path[i + 1]] for i in range(len(path) - 1))
            ants.append((path, cost))
            if cost < best_cost:
                best_cost = cost
                best_path = path
        update_pheromones(pheromones, ants, graph)
    return best_path

# Running the algorithm
best_path_found = aco_algorithm(graph, pheromones)
print("Best path:", best_path_found)
plot_graph(graph, best_path_found)
