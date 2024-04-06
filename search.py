import matplotlib.pyplot as plt
import networkx as nx
import random

def randomDistance():
    number = random.randrange(0,20)
    return number

def randomTraffic():
    traffic = random.randrange(0,5)
    return traffic

def randomUrgency():
    urgency = random.randrange(0,5)
    return urgency

class CityMap:
    def __init__(self):
        self.graph = nx.Graph()

    def add_connection(self, node1, node2, distance, traffic, urgency):
        cost = distance + traffic - urgency
        self.graph.add_edge(node1, node2, weight=cost)

    def find_shortest_path(self, start):
        return nx.single_source_dijkstra_path_length(self.graph, start)

    def shortest_path_to(self, start, end):
        shortest_path = nx.shortest_path(self.graph, start, end)
        shortest_distance = nx.shortest_path_length(self.graph, start, end)
        return shortest_path, shortest_distance


if __name__ == "__main__":
    city_map = CityMap()

    print("Connections")

    # Add connections and their associated costs
    city_map.add_connection("A", "B", randomDistance(), randomTraffic(), randomUrgency())
    city_map.add_connection("A", "C", randomDistance(), randomTraffic(), randomUrgency())
    city_map.add_connection("B", "C", randomDistance(), randomTraffic(), randomUrgency())
    city_map.add_connection("C", "D", randomDistance(), randomTraffic(), randomUrgency())
    city_map.add_connection("C", "E", randomDistance(), randomTraffic(), randomUrgency())
    city_map.add_connection("D", "E", randomDistance(), randomTraffic(), randomUrgency())
    city_map.add_connection("B", "E", randomDistance(), randomTraffic(), randomUrgency())
    city_map.add_connection("B", "D", randomDistance(), randomTraffic(), randomUrgency())
    city_map.add_connection("D", "A", 20, 5, 5)

    start_node = "A"
    end_node = "D"

    shortest_path, shortest_distance = city_map.shortest_path_to(start_node, end_node)
    print("Shortest path from", start_node, "to", end_node, ":", shortest_path)
    print("Shortest distance:", shortest_distance)

    # Visualize the graph
    pos = nx.spring_layout(city_map.graph)
    nx.draw(city_map.graph, pos, with_labels=True, node_color="lightblue", node_size=1500, font_size=12, font_weight="bold")
    labels = nx.get_edge_attributes(city_map.graph, 'weight')
    nx.draw_networkx_edge_labels(city_map.graph, pos, edge_labels=labels)
    plt.title("City Map")
    plt.show()