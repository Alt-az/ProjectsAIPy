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


def deep_neural_network(input,weights):
    alpha=0.01
    for i in range(0, 4):
        hidden_layer = neural_network(input[:, i], weights[0], 0)
        hidden_layer = relu(hidden_layer)
        layer_output = neural_network(hidden_layer, weights[1], 0)
        print(layer_output)






if __name__ == '__main__':
    # input=np.random.uniform(-0.1,0.1,3)
    # weights=np.random.uniform(-0.1,0.1,(5,3))
    input =np.array([[0.5, 0.1, 0.2, 0.8],[0.75, 0.3, 0.1, 0.9],[0.1, 0.7, 0.6, 0.2]])
    weightsh = np.array([[0.1, 0.1, -0.3], [0.1, 0.2, 0.0], [0.0, 0.7, 0.1], [0.2, 0.4, 0.0], [-0.3, 0.5, 0.1]])
    weightsy = np.array([[0.7, 0.9, -0.4, 0.8, 0.1], [0.8, 0.5, 0.3, 0.1, 0.0], [-0.3, 0.9, 0.3, 0.1, -0.2]])
    goals=np.array([[0.376,0.3765,0.305],[0.082,0.133,0.123],[0.053,0.067,0.073],[0.49,0.465,0.402]])
    weights=[weightsh,weightsy]
    deep_neural_network(input,weights)

# print(layer_output,goals[i])
#