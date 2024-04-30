import random
import networkx as nx
import matplotlib.pyplot as plt
import heapq
import math

class GraphEntry:
    def __init__(self, source, destination, distance, traffic_conditions, delivery_urgency):
        self.source = source
        self.destination = destination
        self.distance = distance
        self.traffic_conditions = traffic_conditions
        self.delivery_urgency = delivery_urgency

    def __str__(self):
        return f"""Source Node: {self.source}\nDestination Node: {self.destination}\nDistance: {self.distance} km\nTraffic Conditions: {self.traffic_conditions} 
        ({"No" if self.traffic_conditions == 1 else "Very high" if self.traffic_conditions == 5 else "High" if self.traffic_conditions == 4 else "Moderate" if self.traffic_conditions == 3 else "Low"} traffic)\nDelivery Urgency: {self.delivery_urgency} 
        ({"Not urgent" if self.delivery_urgency == 1 else "Less urgent" if self.delivery_urgency == 2 else "Moderate urgency" if self.delivery_urgency == 3 else "Urgent" if self.delivery_urgency == 4 else "Very urgent"})\n"""

class Graph:
    def __init__(self):
        self.edges = {}

    def add_edge(self, node1, node2, cost):
        if node1 not in self.edges:
            self.edges[node1] = []
        if node2 not in self.edges:
            self.edges[node2] = []
        self.edges[node1].append((node2, cost))
        self.edges[node2].append((node1, cost))  # Add the reverse edge as well

def generate_synthetic_graph(num_entries):
    entries = []
    nodes = ['A', 'B', 'C', 'D', 'E', 'F']
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            source = nodes[i]
            destination = nodes[j]
            distance = random.randint(5, 25)
            traffic_conditions = random.randint(1, 5)
            delivery_urgency = random.randint(1, 5)
            entries.append(GraphEntry(source, destination, distance, traffic_conditions, delivery_urgency))
    return entries

def heuristic(current_node, goal):
    return math.sqrt((ord(current_node) - ord(goal))**2)

def main():
    num_entries = 5
    graph_data = generate_synthetic_graph(num_entries)
    
    print("Graph data:")
    for i, entry in enumerate(graph_data, 1):
        print(f"Entry {i}:\n{entry}")

    # Create a graph
    graph = Graph()

    # Add edges to the graph
    for entry in graph_data:
        graph.add_edge(entry.source, entry.destination, entry.distance)

    # Define state space
    state_space = list(graph.edges.keys())

    # Set initial state
    while True:
        initial_state = input("Enter the initial state (A, B, C, D, E, F): ").upper()
        if initial_state in state_space:
            break
        else:
            print("Invalid initial state. Please enter a valid node.")

    # Set goal state
    while True:
        goal_state = input("Enter the goal state (A, B, C, D, E, F): ").upper()
        if goal_state in state_space:
            break
        else:
            print("Invalid goal state. Please enter a valid node.")

    # Find the optimal path using A* algorithm
    open_set = [(0, initial_state)]
    came_from = {}
    g_score = {node: float('inf') for node in state_space}
    g_score[initial_state] = 0
    f_score = {node: float('inf') for node in state_space}
    f_score[initial_state] = heuristic(initial_state, goal_state)

    while open_set:
        current_cost, current_node = heapq.heappop(open_set)
        if current_node == goal_state:
            path = []
            while current_node in came_from:
                path.append(current_node)
                current_node = came_from[current_node]
            path.append(initial_state)
            print("Optimal path:", path[::-1])
            break

        for neighbor, cost in graph.edges[current_node]:
            tentative_g_score = g_score[current_node] + cost
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current_node
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal_state)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    else:
        print("No path found.")

    # Draw the graph
    G = nx.DiGraph()
    for node, edges in graph.edges.items():
        for edge in edges:
            G.add_edge(node, edge[0], weight=edge[1])

    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_size=1500, node_color="skyblue", font_size=10, font_weight="bold", arrowsize=20)

    # Add edge labels
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    # Show the graph
    plt.title("Synthetic Graph")
    plt.show()

    # Print graph entries
    for i, entry in enumerate(graph_data, 1):
        print(f"Entry {i}:\n{entry}")
    
if __name__ == "__main__":
    main()
