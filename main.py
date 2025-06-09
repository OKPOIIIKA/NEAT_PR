import gymnasium as gym
import numpy as np
import random
from visualize import plot_metrics
# Конфигурация NEAT
class NEATConfig:
    def __init__(self):
        self.population_size = 50  # Размер популяции
        self.species_threshold = 3.0  # Порог для разделения на виды
        self.species_elitism = 2  # Количество элитных особей в виде
        
        # Параметры мутации
        self.mutate_weights_prob = 0.8
        self.weight_perturb_prob = 0.9
        self.weight_random_prob = 0.1
        self.weight_mutation_power = 0.5
        self.add_node_prob = 0.03
        self.add_conn_prob = 0.05
        self.disable_conn_prob = 0.1
        self.enable_conn_prob = 0.2
        
        # Параметры рекомбинации
        self.crossover_prob = 0.75
        self.interspecies_mating_rate = 0.001
        
        # Параметры оценки приспособленности
        self.max_steps = 2000
        self.fitness_threshold = 300

# 1. Кодирование информации (Гены и нейронные сети)
class Gene:
    def __init__(self, innovation_number, input_node, output_node, weight, is_enabled=True):
        self.innovation_number = innovation_number  # Уникальный номер иновации
        self.input_node = input_node  # Номер входного нейрона
        self.output_node = output_node  # Номер выходного нейрона
        self.weight = weight  # Вес связи
        self.is_enabled = is_enabled  # Статус связи (включена/выключена)

    def mutate(self):
        """Мутация веса связи"""
        self.weight += random.uniform(-0.1, 0.1)  # Случайное изменение веса

    def __str__(self):
        return f"Gene({self.input_node}->{self.output_node}): weight={self.weight}, innovation={self.innovation_number}"

class NeuralNetwork:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.nodes = list(range(input_size + output_size))  # входные и выходные нейроны
        self.genes = []  # Гены для связей между нейронами

    def add_gene(self, gene):
        #Добавление нового гена в нейронную сеть
        self.genes.append(gene)

    def think(self, inputs):
        #Простой метод, который рассчитывает выходные данные сети
        outputs = np.zeros(self.output_size)
        for gene in self.genes:
            if gene.is_enabled:
                outputs[gene.output_node - self.input_size] += inputs[gene.input_node] * gene.weight
        return outputs

    def mutate(self, config):
        #Мутация сети (изменение весов или добавление новых связей)
        mutation_type = random.choice(['weight', 'structure'])
        if mutation_type == 'weight':
            # Мутирование весов существующих связей
            gene = random.choice(self.genes)
            gene.mutate()
        elif mutation_type == 'structure':
            # Добавление новой связи
            self.add_new_connection(config)

    def add_new_connection(self, config):
        #Добавление новой связи в сеть
        new_gene = Gene(
            innovation_number=random.randint(1000, 9999),
            input_node=random.randint(0, self.input_size - 1),
            output_node=random.randint(self.input_size, self.input_size + self.output_size - 1),
            weight=random.uniform(-1, 1)
        )
        self.add_gene(new_gene)

    def __str__(self):
        return f"NeuralNetwork({len(self.genes)} genes)"

# 2. Скрещивание (с учетом инноваций)
def crossover(parent1, parent2):
    child = NeuralNetwork(parent1.input_size, parent1.output_size)
    for gene1 in parent1.genes:
        for gene2 in parent2.genes:
            if gene1.innovation_number == gene2.innovation_number:
                # В случае совпадения иновации (если такие есть)
                if random.random() > 0.5:  
                    child.add_gene(gene1)
                else:
                    child.add_gene(gene2)
    return child

# 3. Специализация (создание видов)
class Speciation:
    def __init__(self, tolerance=3.0):
        self.tolerance = tolerance
        self.species = []

    def assign_species(self, population):
        """Назначение особей в разные виды на основе расстояния между генами"""
        for agent in population:
            assigned = False
            for species in self.species:
                if self.is_similar(agent, species.representative):
                    species.members.append(agent)
                    assigned = True
                    break
            if not assigned:
                # Создаем новый вид
                new_species = Species()
                new_species.members.append(agent)
                self.species.append(new_species)

    def is_similar(self, agent1, agent2):
        """Проверка на схожесть между агентами на основе расстояния генов"""
        distance = sum([abs(g1.weight - g2.weight) for g1, g2 in zip(agent1.network.genes, agent2.network.genes)])
        return distance < self.tolerance

class Species:
    def __init__(self):
        self.members = []  # Список членов вида
        self.representative = None  # Представитель вида

    def update_representative(self):
        """Обновление представителя вида на основе среднего представления"""
        self.representative = self.members[random.randint(0, len(self.members) - 1)]

# 4. Коэволюционное усложнение
def coevolution(population1, population2):
    """Коэволюция двух популяций"""
    for agent1 in population1:
        for agent2 in population2:
            if random.random() > 0.5:
                agent1.fitness += 1  # Победа агента 1
            else:
                agent2.fitness += 1  # Победа агента 2

    # Эволюция популяций на основе результатов сражений
    population1.sort(key=lambda x: x.fitness, reverse=True)
    population2.sort(key=lambda x: x.fitness, reverse=True)

    # Отбор и создание нового поколения
    new_population1 = population1[:len(population1)//2]  # Топ половина
    new_population2 = population2[:len(population2)//2]  # Топ половина
    return new_population1, new_population2

# Основная структура агента
class Agent:
    def __init__(self, input_size, output_size):
        self.network = NeuralNetwork(input_size, output_size)  # Нейронная сеть
        self.fitness = 0  # Приспособленность агента

    def evaluate(self, env):
        """Оценка приспособленности агента"""
        observation = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = self.network.think(observation)  # Прогнозирование действия с помощью нейронной сети
            observation, reward, done, info = env.step(action)
            total_reward += reward  # Сумма вознаграждения для оценки приспособленности
        return total_reward

# Основной класс для NEAT
class NEAT:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.population_size = config.population_size
        self.agents = [Agent(self.env.observation_space.shape[0], self.env.action_space.shape[0]) for _ in range(self.population_size)]
        self.generation = 0
        self.best_fitness = -float('inf')  # Изначально лучший результат равен минус бесконечности
        self.best_genome = None
        self.speciation = Speciation()

    def evolve(self):
        """Эволюция популяции"""
        for agent in self.agents:
            agent.fitness = agent.evaluate(self.env)  # Оценка приспособленности

        # Специализация популяции
        self.speciation.assign_species(self.agents)

        # Скрещивание
        new_agents = []
        for species in self.speciation.species:
            species.update_representative()
            for member in species.members:
                parent = species.representative
                child = crossover(parent, member)
                new_agents.append(child)

        # Мутация
        for agent in new_agents:
            agent.network.mutate(self.config)

        # Обновляем популяцию
        self.agents = new_agents
        self.generation += 1

        # Получаем лучший результат (fitness)
        self.best_fitness = max(agent.fitness for agent in self.agents) 
        print(f"Epoch {self.generation}: Best fitness = {self.best_fitness:.2f}")

if __name__ == "__main__":
    env = gym.make('BipedalWalker-v3')
    config = NEATConfig()  
    neat = NEAT(env, config)

    for generation in range(500):  # Эволюция в течение 500 поколений
        neat.evolve()
