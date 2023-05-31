import numpy as np

def to_number(chrom):
    length = 3
    number = 0
    number2 = 0
    index = 0
    for i in chrom:
        if length == -1:
            index = 1
            length = 3
        if index == 0:
            number += i * (np.power(2, length))
        else:
            number2 += i * (np.power(2, length))
        length -= 1
    return number, number2

def get_fitness(chrom):
    num1, num2 = to_number(chrom)
    result = 2 * np.power(num1, 2) + num2
    if result == 33:
        return 100
    elif result > 50:
        return 0
    elif 0 < result < 33:
        return result
    elif 33 < result <= 50:
        return abs(33 - result)

def sum_of_fitness(population):
    sums = 0
    for i in population:
        if type(get_fitness(i))==type(None):
            return 0
        sums += get_fitness(i)
    return sums

def is_goal(population):
    for i in population:
        num1, num2 = to_number(i)
        if 2 * np.power(num1, 2) + num2 == 33:
            print(num1, num2)
            return 1
    return 0

def count_chances(population):
    p = np.zeros(10)
    all = sum_of_fitness(population)
    k = 0
    for i in population:
        p[k] = get_fitness(i) / all
        k += 1
    return p

def crossover(population):
    child = np.zeros((1, 8))
    isin = np.arange(len(population))
    np.random.shuffle(isin)
    for i in range(3):
        rands = np.random.randint(0, 8)
        child[0, 0:rands] = population[isin[i], 0:rands]
        child[0, rands:8] = population[isin[i+1], rands:8]
        choice = np.random.randint(0, 2)
        if choice == 1:
            population[isin[i]] = child
        else:
            population[isin[i+1]] = child
    return population

def mutate(population):
    for i in range(len(population)):
        does = np.random.randint(0, 10)
        if does == 0:
            which = np.random.randint(0, 8)
            if population[i, which] == 1:
                population[i, which] = 0
            else:
                population[i, which] = 1
    return population

if __name__ == '__main__':
    population = np.random.randint(0, 2, size=(10, 8))
    while is_goal(population) == 0:
        if sum_of_fitness(population) != 0:
            chance = count_chances(population)
            population = population[np.random.choice(population.shape[0], size=10, p=chance), :]
        else:
            population = population[np.random.choice(population.shape[0], size=10), :]
        population = crossover(population)
        population = mutate(population)



