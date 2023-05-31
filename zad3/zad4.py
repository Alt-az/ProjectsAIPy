import numpy as np

def neural_network(input, weights, bias):
    return (np.matmul(weights, input)) + bias

def load_weights(file_name):
    return np.loadtxt(file_name,dtype=float)

def relu(x):
    x[x < 0] = 0
    return x

def relu_deriv(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x

def neuron(traingoal, traininput,testinput,testgoal, alpha, rep):
    weight= []
    weight.append(np.random.uniform(-0.1, 0.1, (5, 3)))
    weight.append(np.random.uniform(-0.1, 0.1, (4, 5)))
    for x in range(0, rep):
        for h in range(0, len(traininput[0])):
            goal=np.zeros((4,1))
            goal[int(traingoal[h])-1]=1
            input=traininput[:, h]
            input=np.reshape(input,(3,1))
            hidden_layer = np.matmul(weight[0], input)
            hidden_layer = relu(hidden_layer)
            layer_output = np.matmul(weight[1], hidden_layer)
            layer_output_delta = (2 / 4) * (layer_output - goal)
            layer_output_weight_delta = np.matmul(layer_output_delta, hidden_layer.T)
            layer_hidden_1_delta = np.matmul(weight[1].T, layer_output_delta)
            layer_hidden_1_delta = layer_hidden_1_delta * relu_deriv(hidden_layer)
            layer_hidden_1_weight_delta = np.matmul(layer_hidden_1_delta, input.T)
            weight[0] = weight[0] - alpha * layer_hidden_1_weight_delta
            weight[1] = weight[1] - alpha * layer_output_weight_delta
    correct=0
    for h in range(0, len(traininput[0])):
        goal = np.zeros((4, 1))
        goal[int(testgoal[h]) - 1] = 1
        input = testinput[:, h]
        input = np.reshape(input, (3, 1))
        pred = np.matmul(weight[0], input)
        pred = relu(pred)
        pred = np.matmul(weight[1], pred)
        if (np.argmax(pred)+1 == testgoal[h]):
            correct=correct+1
        print(np.argmax(pred)+1,'|',int(testgoal[h]))
    result=correct/len(traininput[0])*100
    print('skutecznosc to', result,'%')


if __name__ == '__main__':
    train=load_weights('training_colors.txt')
    traininput=train[:,0:3].T
    traingoal=train[:,3]
    test = load_weights('test_colors.txt')
    testinput = test[:, 0:3].T
    testgoal = test[:, 3]
    neuron(traingoal,traininput,testinput,testgoal,0.01,100)
