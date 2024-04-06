import matplotlib.pyplot as plt
import networkx as nx

class CityMap:
    def __init__(self):
        self.graph = nx.Graph()

    def add_connection(self, node1, node2, cost):
        self.graph.add_edge(node1, node2, weight=cost)

    def find_shortest_path(self, start):
        return nx.single_source_dijkstra_path_length(self.graph, start)

    def shortest_path_to(self, start, end):
        shortest_path = nx.shortest_path(self.graph, start, end)
        shortest_distance = nx.shortest_path_length(self.graph, start, end)
        return shortest_path, shortest_distance


if __name__ == "__main__":
    city_map = CityMap()

    # Add connections and their associated costs
    city_map.add_connection("A", "B", 5)
    city_map.add_connection("A", "C", 10)
    city_map.add_connection("B", "C", 3)
    city_map.add_connection("B", "D", 9)
    city_map.add_connection("C", "D", 2)

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