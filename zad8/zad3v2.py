import random
import numpy as np


class Item:
    def __init__(self, value, weight):
        self.value = value
        self.weight = weight


class Backpack:
    def __init__(self, max_weight, items):
        self.max_weight = max_weight
        self.items = items


def get_fitness(backpack):
    total_value = 0
    total_weight = 0
    for item in backpack.items:
        total_value += item.value
        total_weight += item.weight
    if total_weight > backpack.max_weight:
        return 0
    return total_value


def create_initial_population(pop_size, backpack):
    population = []
    for i in range(pop_size):
        population.append([random.randint(0, 1) for _ in range(len(backpack.items))])
    return population


def get_elite(population, elite_size, backpack):
    fitness_values = []
    for individual in population:
        backpack_copy = Backpack(backpack.max_weight,[backpack.items[i] for i in range(len(individual)) if individual[i] == 1])
        fitness_values.append(get_fitness(backpack_copy))
    elite_indexes = sorted(range(len(fitness_values)), key=lambda i: fitness_values[i], reverse=True)[:elite_size]
    return [population[i] for i in elite_indexes]


def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    child = parent1[:crossover_point] + parent2[crossover_point:]
    return child


def mutate(individual, mutation_rate):
    for i in range(len(individual)):
        if random.uniform(0, 1) < mutation_rate:
            if individual[i] == 1:
                individual[i] = 0
            else:
                individual[i] = 1
    return individual


def create_new_population(previous_population, elite, pop_size):
    new_population = []
    population_elite_size = int(pop_size * 0.25)
    population_children_size = pop_size - population_elite_size
    new_population.extend(elite)
    population_children = []
    while len(population_children) < population_children_size:
        parent1_index = random.randint(0, len(previous_population) - 1)
        parent2_index = random.randint(0, len(previous_population) - 1)
        if parent1_index != parent2_index:
            child = crossover(previous_population[parent1_index], previous_population[parent2_index])
            population_children.append(child)
    for individual in population_children:
        new_population.append(mutate(individual, 0.05))
    return new_population


def next_generation(current_generation, elite_size, pop_size, backpack):
    elite = get_elite(current_generation, elite_size, backpack)
    new_population = create_new_population(current_generation, elite, pop_size)
    return new_population


# def run_genetic_algorithm(backpack, elite_size, pop_size, num_generations):
#     population = create_initial_population(pop_size, backpack)
#     for i in range(num_generations):
#         population = next_generation(population, elite_size, pop_size, backpack)
#     best_backpack = Backpack(backpack.max_weight, [backpack.items[i] for i in range(len(population[0])) if population[0][i] == 1])
#     return best_backpack


if __name__ == '__main__':
    weight = np.array([3, 13, 10, 9, 7, 1, 8, 8, 2, 9])
    cost = np.array([266, 442, 671, 526, 388, 245, 210, 145, 126, 322])
    items = []
    for i in range(len(weight)):
        items.append(Item(cost[i], weight[i]))
    backpack = Backpack(35, items)
    elite_size = int(8 * 0.25)
    pop_size = 8
    num_generations = 100
    population = create_initial_population(pop_size, backpack)
    for i in range(num_generations):
        population = next_generation(population, elite_size, pop_size, backpack)
    best_backpack = Backpack(backpack.max_weight, [backpack.items[i] for i in range(len(population[0])) if population[0][i] == 1])
    print("Wartość:", get_fitness(best_backpack))
    print("Przedmioty:")
    for _, item in enumerate(best_backpack.items):
        print("wartość ", item.value, "waga", item.weight)
