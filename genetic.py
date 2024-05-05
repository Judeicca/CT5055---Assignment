import random
import matplotlib.pyplot as plt

class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate, crossover_rate, max_generations, coordinates):
        # Initialize the Genetic Algorithm with basic config/parameters
        self.population_size = population_size  # Number of solutions in each generation
        self.mutation_rate = mutation_rate  # Probability of random alterations in a route
        self.crossover_rate = crossover_rate  # Probability that two routes will mix to create new routes
        self.max_generations = max_generations  # Max generations
        self.coordinates = coordinates  # Dictionary of node coordinates
        self.best_solution = None  # Store the best solution
        self.best_fitness = float('-inf')  # Highest fitness value found initialized to negative infinity

    def initialize_population(self, start_node, end_node):
        # Create the initial population as a list of random routes
        population = []
        for _ in range(self.population_size):
            chromosome = self.generate_random_chromosome(start_node, end_node)
            population.append(chromosome)
        return population

    def generate_random_chromosome(self, start_node, end_node):
        # Generate a random route that starts at `start_node`, ends at `end_node`, and visits all other nodes randomly
        nodes = list(labels)  # Create a list of all node labels
        nodes.remove(start_node)  # Exclude the start node
        nodes.remove(end_node)  # Exclude the end node
        random.shuffle(nodes)  # Shuffle the remaining nodes randomly
        return [start_node] + nodes + [end_node]  # Return a complete route

    def crossover(self, parent1, parent2):
        # Combine two routes (parents) to create new routes (children) via crossover
        start_node = parent1[0]  # Start node from the first parent
        end_node = parent1[-1]  # End node from the first parent
        crossover_point = random.randint(1, len(parent1) - 2)  # Choose a random point for crossover
        
        # Initialize children with None to ensure correct length
        child1 = [None] * len(parent1)
        child2 = [None] * len(parent1)
        # Set start and end nodes for children
        child1[0], child1[-1] = start_node, end_node
        child2[0], child2[-1] = start_node, end_node
        
        # Copy segments from parents to children up to the crossover point
        child1[1:crossover_point] = parent1[1:crossover_point]
        child2[1:crossover_point] = parent2[1:crossover_point]
        
        # Complete the routes by ensuring all nodes are included without repetition
        self.fill_remaining(child1, parent2, set(child1))
        self.fill_remaining(child2, parent1, set(child2))
        
        return child1, child2

    def mutate(self, chromosome):
        # Randomly alter the route to introduce variability
        for i in range(len(chromosome) - 1):
            if random.random() < self.mutation_rate:  # Mutation occurs based on the mutation rate
                j = random.randint(1, len(chromosome) - 2)  # Select another position randomly
                chromosome[i], chromosome[j] = chromosome[j], chromosome[i]  # Swap the nodes
        return chromosome

    def evaluate_fitness(self, chromosome):
        # Calculate the fitness of a route based on its total urgency, distance, and traffic
        total_distance = self.calculate_total_distance(chromosome)
        total_traffic = sum(traffic[node] for node in chromosome)
        total_urgency = sum(urgency[node] for node in chromosome)
        # Fitness formula: higher urgency and lower distance and traffic lead to higher fitness
        fitness = total_urgency / (total_distance + 0.1 * total_traffic)
        return fitness

    def evolve(self, population):
        # Generate a new generation from the current population
        new_population = []
        while len(new_population) < len(population):
            parent1, parent2 = random.sample(population, 2)  # Select two parents randomly
            child1, child2 = self.crossover(parent1, parent2)  # Create two children via crossover
            child1 = self.mutate(child1)  # Mutate the first child
            child2 = self.mutate(child2)  # Mutate the second child
            new_population.extend([child1, child2])  # Add the new children to the new population
        # Return the top individuals from the combined old and new populations
        return sorted(new_population + population, key=self.evaluate_fitness, reverse=True)[:self.population_size]

    def run_genetic_algorithm(self, start_node, end_node):
        population = self.initialize_population(start_node, end_node)
        plt.ion()
        fig, ax = plt.subplots()

        for generation in range(self.max_generations):
            population = self.evolve(population)
            best_solution = max(population, key=lambda x: self.evaluate_fitness(x))
            best_fitness = self.evaluate_fitness(best_solution)
            if best_fitness > self.best_fitness:
                self.best_solution = best_solution
                self.best_fitness = best_fitness

            if generation % 1 == 0:  # Changes how often the plot is updated
                ax.clear()
                self.plot_best_solution(ax, best_solution, generation + 1)
                plt.draw()
                plt.pause(0.1)

            print(f"Generation {generation + 1}: Best Path = {best_solution} with Fitness = {best_fitness}")

        plt.ioff()
        plt.show()


    def plot_best_solution(self, ax, solution, generation):
        # Plot the best solution on the graph
        for label, coord in self.coordinates.items():
            ax.scatter(*coord, color='red')  # Plot nodes as red dots
            ax.text(coord[0], coord[1] + 2, label, fontsize=12, ha='center')  # Label nodes

        path_coords = [self.coordinates[node] for node in solution]  # Extract coordinates for the best solution
        ax.plot([p[0] for p in path_coords], [p[1] for p in path_coords], 'b-o')  # Draw the path

        total_distance = self.calculate_total_distance(solution)  # Calculate total distance for the path
        ax.text(0.5, -0.1, f'Generation: {generation}, Total Distance: {total_distance}',
                transform=ax.transAxes, fontsize=12, ha='center')  # Display generation and distance

    def fill_remaining(self, child, parent, current_set):
        # Fill remaining nodes in a child route to ensure it's complete without repetition
        position = next((i for i, x in enumerate(child) if x is None), None)
        while position is not None:
            for node in parent:
                if node not in current_set:
                    child[position] = node
                    current_set.add(node)
                    break
            position = next((i for i, x in enumerate(child) if x is None), None)

    def calculate_total_distance(self, chromosome):
        # Calculate the total distance of a route
        return sum(distance_matrix[chromosome[i]][chromosome[i + 1]] for i in range(len(chromosome) - 1))

# Example usage of the Genetic Algorithm with specific parameters
num_nodes = 26
labels = [chr(ord('A') + i) for i in range(num_nodes)]
distance_matrix = {labels[i]: {labels[j]: random.randint(1, 10) for j in range(num_nodes)} for i in range(num_nodes)}
traffic = {label: random.randint(1, 5) for label in labels}
urgency = {label: random.randint(1, 5) for label in labels}
coordinates = {label: (random.randint(0, 100), random.randint(0, 100)) for label in labels}

# Configuration for the genetic algorithm
population_size = 20
mutation_rate = 0.025
crossover_rate = 0.7
max_generations = 100

# Initialize and run the genetic algorithm
ga = GeneticAlgorithm(population_size, mutation_rate, crossover_rate, max_generations, coordinates)
start_node, end_node = 'A', 'Z'  # Define start and end nodes
ga.run_genetic_algorithm(start_node, end_node)