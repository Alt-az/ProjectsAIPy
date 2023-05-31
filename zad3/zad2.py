import numpy as np

def neural_network(input,weights,bias):
    return (np.matmul(weights,input))+bias

def relu(x):
    x[x < 0] = 0
    return x

def relu_deriv(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x


def deep_neural_network(input,weights,goals,epoki):
    alpha=0.01
    for x in range(0,epoki):
        for i in range(0, 4):
            hidden_layer = neural_network(input[:, i], weights[0], 0)
            hidden_layer = relu(hidden_layer)
            layer_output = neural_network(hidden_layer, weights[1], 0)
            layer_output_delta = 2 * 1 / 3 * (layer_output - goals.T[i])
            layer_output_weight_delta = np.outer(layer_output_delta, hidden_layer)
            layer_hidden_1_delta = np.matmul(weights[1].T ,layer_output_delta)
            layer_hidden_1_delta = layer_hidden_1_delta * relu_deriv(hidden_layer.T)
            layer_hidden_1_weight_delta = np.outer(layer_hidden_1_delta, input[:, i])
            weights[0] = weights[0] - alpha * layer_hidden_1_weight_delta
            weights[1] = weights[1] - alpha * layer_output_weight_delta
            print(layer_output)
        print("--------")


if __name__ == '__main__':
    # input=np.random.uniform(-0.1,0.1,3)
    # weights=np.random.uniform(-0.1,0.1,(5,3))
    input =np.array([[0.5, 0.1, 0.2, 0.8],[0.75, 0.3, 0.1, 0.9],[0.1, 0.7, 0.6, 0.2]])
    weightsh = np.array([[0.1, 0.1, -0.3], [0.1, 0.2, 0.0], [0.0, 0.7, 0.1], [0.2, 0.4, 0.0], [-0.3, 0.5, 0.1]])
    weightsy = np.array([[0.7, 0.9, -0.4, 0.8, 0.1], [0.8, 0.5, 0.3, 0.1, 0.0], [-0.3, 0.9, 0.3, 0.1, -0.2]])
    goals=np.array([[0.1, 0.5, 0.1, 0.7],[1.0, 0.2, 0.3, 0.6],[0.1, -0.5, 0.2, 0.2]])
    weights=[weightsh,weightsy]
    deep_neural_network(input,weights,goals,50)
