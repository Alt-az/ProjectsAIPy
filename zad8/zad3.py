import numpy as np
class Genetic:
    weight = np.array([3, 13, 10, 9, 7, 1, 8, 8, 2, 9])
    cost = np.array([266, 442, 671, 526, 388, 245, 210, 145, 126, 322])
    best = np.zeros(10)
    def funPun(self,chromosome):
        sum_weight = 0
        sum_cost = 0
        for x in range(len(chromosome)):
            sum_weight += chromosome[x] * self.weight[x]
            sum_cost += chromosome[x] * self.cost[x]
        if sum_weight <= 35:
            if sum_cost > self.funPun(best):
                self.best = chromosome
            return sum_cost
        else:
            return 0

if __name__ == '__main__':
    population = np.random.randint(0, 2, size=(8, 10))
    print(population)
    best = np.zeros(10)
