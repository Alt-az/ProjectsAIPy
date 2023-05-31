import numpy as np


def training_images(path):
    with open(path, 'rb') as f:
        magic_number = int.from_bytes(f.read(4), 'big')
        image_count = int.from_bytes(f.read(4), 'big')
        row_count = int.from_bytes(f.read(4), 'big')
        column_count = int.from_bytes(f.read(4), 'big')
        image_data = f.read()
        images = np.frombuffer(image_data, dtype=np.uint8).reshape((image_count, row_count, column_count))
        return images


def training_labels(path):
    with open(path, 'rb') as f:
        magic_number = int.from_bytes(f.read(4), 'big')
        label_count = int.from_bytes(f.read(4), 'big')
        label_data = f.read()
        labels = np.frombuffer(label_data, dtype=np.uint8)
        return labels


class fully_connected_layers:
    layers = 0
    weights = []
    neurons_last_layer = 0
    alpha = 0.02

    def __init__(self, num_input):
        self.neurons_last_layer = num_input

    def sigmoid(self, x):
        one = np.ones_like(x)
        return (one / (one + np.exp(-x)))

    def sigmoid_deriv(self, x):
        one = np.ones_like(x)
        return (x(one - x))

    def tanh(self, x):
        x = np.tanh(x)
        return x

    def tanh_deriv(self, x):
        one = np.ones_like(x)
        x = (one - (x ** 2))
        return x

    def relu(self, x):
        x[x < 0] = 0
        return x

    def relu_deriv(self, x):
        x[x <= 0] = 0
        x[x > 0] = 1
        return x

    def softmax(self, x):
        size, size1 = x.shape
        for i in range(0, size1):
            x[:, i] = np.exp(x[:, i]) / np.sum(np.exp(x[:, i]))
        return x

    def add_layer(self, n, weight_min_max):
        self.layers += 1
        self.weights.append(np.random.uniform(weight_min_max[0], weight_min_max[1], (n, self.neurons_last_layer)))
        self.neurons_last_layer = n
        return

    def predict(self, input):
        inputflat = np.reshape(input, [-1, 784]).T
        inputflat = inputflat.astype('float32') / 255
        hidden_layer = np.matmul(self.weights[0], inputflat)
        hidden_layer = self.tanh(hidden_layer)
        layer_output = np.matmul(self.weights[1], hidden_layer)
        layer_output = self.softmax(layer_output)
        return layer_output

    def fit(self, input, expected_output):
        size, = input[:, 0, 0].shape
        for i in range(0, size, 100):
            goal = np.zeros((10, 100))
            inputflat = np.reshape(input[i:i + 100], [100, 784]).T
            for j in range(0, 100):
                goal[expected_output[i + j], j] = 1
            inputflat = inputflat.astype('float32') / 255
            hidden_layer = np.matmul(self.weights[0], inputflat)
            hidden_layer = self.tanh(hidden_layer)
            hidden_layer_size, = hidden_layer[:, 0].shape
            binar = np.ones((hidden_layer_size, 100))
            binar[:, :int(100 / 2)] = 0
            np.random.shuffle(binar)
            hidden_layer = hidden_layer * binar * 2
            layer_output = np.matmul(self.weights[1], hidden_layer)
            layer_output = self.softmax(layer_output)
            layer_output_delta = (2 / 10) * (layer_output - goal) / 100
            layer_output_weight_delta = np.matmul(layer_output_delta, hidden_layer.T)
            layer_hidden_1_delta = np.matmul(self.weights[1].T, layer_output_delta)
            layer_hidden_1_delta = layer_hidden_1_delta * self.tanh_deriv(hidden_layer)
            layer_hidden_1_delta = layer_hidden_1_delta * binar
            layer_hidden_1_weight_delta = np.matmul(layer_hidden_1_delta, inputflat.T)
            self.weights[0] = self.weights[0] - self.alpha * layer_hidden_1_weight_delta
            self.weights[1] = self.weights[1] - self.alpha * layer_output_weight_delta
        # print(sum((layer_output - goal) ** 2))
        return

    def save_weights(self, file_name):
        array = np.asarray(self.weights, dtype="object")
        np.save(file_name, array, allow_pickle=True)
        return

    def load_weights(self, file_name):
        content = np.load(file_name, allow_pickle=True)
        self.layers += 1
        self.weights = content.copy()
        return


if __name__ == '__main__':
    trainpaths = ['data/train-images.idx3-ubyte', 'data/train-labels.idx1-ubyte']
    trainimages = training_images(trainpaths[0])
    trainlabels = training_labels(trainpaths[1])
    testimages = training_images("data/t10k-images.idx3-ubyte")
    testlabels = training_labels("data/t10k-labels.idx1-ubyte")
    clas = fully_connected_layers(784)
    clas.add_layer(100, [-0.01, 0.01])
    clas.add_layer(10, [-0.1, 0.1])
    for i in range(0, 100):
        clas.fit(trainimages[0:1000], trainlabels[0:1000])
    correct = 0
    all = 0
    size, = testimages[:, 0, 0].shape
    for i in range(0, size):
        t = clas.predict(testimages[i])
        if np.argmax(t) == testlabels[i]:
            correct = correct + 1
        all = all + 1
    result = (correct / all) * 100
    print(result, '%')
    clas.save_weights('weights.npy')
    print(clas.weights[0].shape, clas.weights[1].shape)
    clas.load_weights('weights.npy')
    print(clas.weights[0].shape, clas.weights[1].shape)
    for i in range(0, size):
        t = clas.predict(testimages[i])
        if np.argmax(t) == testlabels[i]:
            correct = correct + 1
        all = all + 1
    result = (correct / all) * 100
    print(result, '%')
