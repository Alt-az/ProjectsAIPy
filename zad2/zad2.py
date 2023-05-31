import numpy as np


def neural_network(input, weights, bias):
    return (np.matmul(weights, input)) + bias


def neuron(weight, goal, input, alpha, rep):
    pred = np.zeros_like(goal)
    error = np.zeros(len(input[0]))
    for x in range(0, rep):
        for h in range(0, len(input[0])):
            pred[:, h] = neural_network(input[:, h], weight, 0)
            delta = np.outer(2 / len(weight) * (pred[:, h] - goal[:, h]), input[:, h])
            error[h] = 1/len(weight) * np.sum((pred[:, h] - goal[:, h]) ** 2)
            weight = weight - (delta * alpha)
        sumerror=np.sum(error)
    return pred, sumerror


if __name__ == '__main__':
    weights = np.array([[0.1, 0.1, -0.3], [0.1, 0.2, 0.0], [0.0, 0.7, 0.1], [0.2, 0.4, 0.0], [-0.3, 0.5, 0.1]])
    input = np.array([[0.5, 0.1, 0.2, 0.8], [0.75, 0.3, 0.1, 0.9], [0.1, 0.7, 0.6, 0.2]])
    goal = np.array([[0.1, 0.5, 0.1, 0.7], [1.0, 0.2, 0.3, 0.6], [0.1, -0.5, 0.2, 0.2], [0.0, 0.3, 0.9, -0.1],
                     [-0.1, 0.7, 0.1, 0.8]])
    alpha = 0.01
    output, error = neuron(weights, goal, input, alpha, 1000)
    print(output, '\n', error)
