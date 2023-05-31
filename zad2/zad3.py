import numpy as np

def neural_network(input, weights, bias):
    return (np.matmul(weights, input)) + bias

def load_weights(file_name):
    return np.loadtxt(file_name,dtype=float)

def neuron(traingoal, traininput,testinput,testgoal, alpha, rep):
    weight = np.random.uniform(-0.1, 0.1, (4,3))
    for x in range(0, rep):
        for h in range(0, len(traininput[0])):
            goal=np.zeros((4,1))
            goal[int(traingoal[h])-1]=1
            pred = neural_network(traininput[:, h], weight, 0)
            delta = np.outer(2 / len(weight) * (pred - goal.T).T, traininput[:, h])
            weight = weight - (delta * alpha)
    correct=0
    for h in range(0, len(traininput[0])):
        goal = np.zeros((4, 1))
        goal[int(testgoal[h]) - 1] = 1
        pred= neural_network(testinput[:, h], weight, 0)
        if (np.argmax(pred)+1==testgoal[h]):
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
    neuron(traingoal,traininput,testinput,testgoal,0.05,1000)


