import numpy as np

def get_two_best(population):
    sums = sum(population.T)
    which = np.argmax(sums)
    sums[which] = 0
    which2 = np.argmax(sums)
    return [which, which2]

def crossover(population):
    child1 = np.zeros((1, 10))
    child2 = np.zeros((1, 10))
    rands = np.random.randint(0, 10)
    best1, best2 = get_two_best(population)
    child1[0, 0:rands] = population[best1, 0:rands]
    child1[0, rands:10] = population[best2, rands:10]
    child2[0, 0:rands] = population[best2, 0:rands]
    child2[0, rands:10] = population[best1, rands:10]
    return [child1, child2]

def get_two_worst(population):
    sums = sum(population.T)
    which = np.argmin(sums)
    sums[which] = 100
    which2 = np.argmin(sums)
    return [which, which2]

def exchange_two_worst(population):
    worst1, worst2 = get_two_worst(population)
    child1, child2 = crossover(population)
    population[worst1] = child1
    population[worst2] = child2
    return population

def mutate_two_best(population):
    best1, best2 = get_two_best(population)
    does = np.random.randint(0, 10)
    if does < 6:
        which2 = np.random.randint(0, 10)
        if population[best1, which2] == 1:
            population[best1, which2] = 0
        else:
            population[best1, which2] = 1
    does = np.random.randint(0, 10)
    if does < 6:
        which2 = np.random.randint(0, 10)
        if population[best2, which2] == 1:
            population[best2, which2] = 0
        else:
            population[best2, which2] = 1
    return population

def is_goal(population):
    sums = sum(population.T)
    sumsofsums = sum(sums)
    if sumsofsums == 100:
        return 1
    return 0

if __name__ == '__main__':
    population = np.random.randint(0, 2, size=(10, 10))
    while is_goal(population) == 0:
        population = exchange_two_worst(population)
        population = mutate_two_best(population)
    print(population)

