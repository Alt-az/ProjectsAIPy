import numpy as np

class fully_connected_layers:
    layers = 0
    weights = []
    neurons_last_layer = 0

    def __init__(self, num_input):
        self.neurons_last_layer = num_input

    def add_layer(self, n, weight_min_max):
        self.layers += 1
        self.weights.append(np.random.uniform(weight_min_max[0],weight_min_max[1],(n,self.neurons_last_layer)))
        return

    def predict(self,input):
        layers=self.layers
        for x in range(0,layers):
            input=np.matmul(self.weights[x], input)
        return input

    def load_weights(self,file_name):
        f=np.loadtxt(file_name,dtype=float)
        self.layers+=1
        self.weights.append(f)
        return


if __name__ == '__main__':
    neurals=fully_connected_layers(3)
    # neurals.add_layer(5,[-0.1,0.1])
    neurals.load_weights('file1.txt')
    neurals.load_weights('file2.txt')
    output=neurals.predict([0.5,0.75,0.1])
    print(output)