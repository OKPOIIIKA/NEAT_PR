import gymnasium as gym
import numpy as np
import random
from visualize import plot_metrics, visualize_network_with_weight_categories  

class NEATConfig:
    def __init__(self):
        self.population_size = 50  
        self.species_threshold = 3.0  
        self.species_elitism = 2  
        
        self.mutate_weights_prob = 0.8
        self.weight_perturb_prob = 0.9
        self.weight_random_prob = 0.1
        self.weight_mutation_power = 0.5
        self.add_node_prob = 0.03
        self.add_conn_prob = 0.05
        self.disable_conn_prob = 0.1
        self.enable_conn_prob = 0.2
        
        self.crossover_prob = 0.75
        self.interspecies_mating_rate = 0.001

        self.max_steps = 2000
        self.fitness_threshold = 300

class Gene:
    def __init__(self, innovation_number, input_node, output_node, weight, is_enabled=True):
        self.innovation_number = innovation_number  
        self.input_node = input_node 
        self.output_node = output_node 
        self.weight = weight 
        self.is_enabled = is_enabled  

    def mutate(self):
        self.weight += random.uniform(-0.1, 0.1)  

    def __str__(self):
        return f"Gene({self.input_node}->{self.output_node}): weight={self.weight}, innovation={self.innovation_number}"

class NeuralNetwork:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.nodes = list(range(input_size + output_size))  
        self.genes = []  

    def add_gene(self, gene):
        self.genes.append(gene)

    def think(self, inputs):
        outputs = np.zeros(self.output_size)
        for gene in self.genes:
            if gene.is_enabled:
                outputs[gene.output_node - self.input_size] += inputs[gene.input_node] * gene.weight
        return outputs

    def mutate(self, config):
        mutation_type = random.choice(['weight', 'structure'])
        if mutation_type == 'weight':
            gene = random.choice(self.genes)
            gene.mutate()
        elif mutation_type == 'structure':
            self.add_new_connection(config)

    def add_new_connection(self, config):
        new_gene = Gene(
            innovation_number=random.randint(1000, 9999),
            input_node=random.randint(0, self.input_size - 1),
            output_node=random.randint(self.input_size, self.input_size + self.output_size - 1),
            weight=random.uniform(-1, 1)
        )
        self.add_gene(new_gene)

    def __str__(self):
        return f"NeuralNetwork({len(self.genes)} genes)"

def crossover(parent1, parent2):
    child = NeuralNetwork(parent1.input_size, parent1.output_size)
    for gene1 in parent1.genes:
        for gene2 in parent2.genes:
            if gene1.innovation_number == gene2.innovation_number:
                if random.random() > 0.5:  
                    child.add_gene(gene1)
                else:
                    child.add_gene(gene2)
    return child

class Speciation:
    def __init__(self, tolerance=3.0):
        self.tolerance = tolerance
        self.species = []

    def assign_species(self, population):
        for agent in population:
            assigned = False
            for species in self.species:
                if self.is_similar(agent, species.representative):
                    species.members.append(agent)
                    assigned = True
                    break
            if not assigned:
                new_species = Species()
                new_species.members.append(agent)
                self.species.append(new_species)

    def is_similar(self, agent1, agent2):

        distance = sum([abs(g1.weight - g2.weight) for g1, g2 in zip(agent1.network.genes, agent2.network.genes)])
        return distance < self.tolerance

class Species:
    def __init__(self):
        self.members = []  
        self.representative = None  

    def update_representative(self):
        self.representative = self.members[random.randint(0, len(self.members) - 1)]


def coevolution(population1, population2):

    for agent1 in population1:
        for agent2 in population2:
            if random.random() > 0.5:
                agent1.fitness += 1  
            else:
                agent2.fitness += 1  

    population1.sort(key=lambda x: x.fitness, reverse=True)
    population2.sort(key=lambda x: x.fitness, reverse=True)

    new_population1 = population1[:len(population1)//2]  
    new_population2 = population2[:len(population2)//2]  
    return new_population1, new_population2

class Agent:
    def __init__(self, input_size, output_size):
        self.network = NeuralNetwork(input_size, output_size)  
        self.fitness = 0  

    def evaluate(self, env):

        observation = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = self.network.think(observation)  
            observation, reward, done, info = env.step(action)
            total_reward += reward  
        return total_reward

class NEAT:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.population_size = config.population_size
        self.agents = [Agent(self.env.observation_space.shape[0], self.env.action_space.shape[0]) for _ in range(self.population_size)]
        self.generation = 0
        self.best_fitness = -float('inf')  
        self.best_genome = None
        self.speciation = Speciation()

        self.fitness_history = []

    def evolve(self):
        
        for agent in self.agents:
            agent.fitness = agent.evaluate(self.env)  

        self.speciation.assign_species(self.agents)

        new_agents = []
        for species in self.speciation.species:
            species.update_representative()
            for member in species.members:
                parent = species.representative
                child = crossover(parent, member)
                new_agents.append(child)

        for agent in new_agents:
            agent.network.mutate(self.config)

        self.agents = new_agents
        self.generation += 1
    
        self.best_fitness = max(agent.fitness for agent in self.agents) 
        print(f"Epoch {self.generation}: Best fitness = {self.best_fitness:.2f}")

        self.fitness_history.append(self.best_fitness)

        if self.generation in [1, 50, 150]:
            visualize_network_with_weight_categories(agent.network, self.generation, filename=f"neural_network_structure_epoch_{self.generation}.png")
        
    def plot_final_metrics(self):
        plot_metrics(self.fitness_history)

if __name__ == "__main__":
    env = gym.make('BipedalWalker-v3')
    config = NEATConfig()  
    neat = NEAT(env, config)

    for generation in range(200):  
        neat.evolve()

    neat.plot_final_metrics()
