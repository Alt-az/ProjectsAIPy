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

def relu(x):
    x[x < 0] = 0
    return x

def relu_deriv(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x

def predict(input,kernel_weights,output_weight):
    input_sections = []
    for filter_y in range(0, 26, 1):
        for filter_x in range(0, 26, 1):
            input_sections.append(input[filter_y:(filter_y + 3), filter_x:(filter_x + 3)].T.flatten())
    input_sections = np.array(input_sections)
    input_sections = input_sections.astype('float32') / 255
    kernel_layer = np.matmul(input_sections, kernel_weights.T)
    kernel_layer = relu(kernel_layer)
    layer_output = np.matmul(output_weight, kernel_layer.flatten())
    return layer_output

def convolutional_neural_networks(input, expected_output, alpha):
    kernel_weights = np.random.uniform(-0.01, 0.01, (16, 9))
    output_weight = np.random.uniform(-0.1, 0.1, (10, 10816))
    size, = input[:, 0, 0].shape
    input_all_sections = []
    for i in range(0, size, 1):
        input_sections = []
        for filter_y in range(0, 26, 1):
            for filter_x in range(0, 26, 1):
                input_sections.append(input[i, filter_y:(filter_y + 3), filter_x:(filter_x + 3)].T.flatten())
        input_sections = np.array(input_sections)
        input_sections = input_sections.astype('float32') / 255
        input_all_sections.append(input_sections)
    input_all_sections = np.array(input_all_sections)
    input_all_sections = input_all_sections.astype('float32') / 255
    print("segmentation ended")
    for r in range(0, 50):
        print("Step", r)
        for i in range(0, size, 1):
            goal = np.zeros((10, 1))
            goal[expected_output[i]] = 1
            kernel_layer = np.matmul(input_all_sections[i], kernel_weights.T)
            kernel_layer = relu(kernel_layer)
            layer_output = np.matmul(output_weight, kernel_layer.flatten())
            layer_output = np.reshape(layer_output, goal.shape)
            layer_output_delta = 2/10*(np.subtract(layer_output, goal))
            kernel_layer_1_delta = np.matmul(output_weight.T, layer_output_delta)
            kernel_layer_flatten = np.reshape(kernel_layer.flatten(), (10816, 1))
            kernel_layer_1_delta = kernel_layer_1_delta * relu_deriv(kernel_layer_flatten)
            kernel_layer_1_delta_reshaped = np.reshape(kernel_layer_1_delta, kernel_layer.shape)
            layer_output_weight_delta = np.matmul(layer_output_delta, kernel_layer_flatten.T)
            kernel_layer_1_weight_delta = np.matmul(kernel_layer_1_delta_reshaped.T, input_all_sections[i])
            output_weight = output_weight - alpha * layer_output_weight_delta
            kernel_weights = kernel_weights - alpha * kernel_layer_1_weight_delta
    return output_weight, kernel_weights

if __name__ == '__main__':
    trainpaths = ['data/train-images.idx3-ubyte', 'data/train-labels.idx1-ubyte']
    trainimages = training_images(trainpaths[0])
    trainlabels = training_labels(trainpaths[1])
    testimages = training_images("data/t10k-images.idx3-ubyte")
    testlabels = training_labels("data/t10k-labels.idx1-ubyte")
    output_weights,kernel_weights=convolutional_neural_networks(trainimages[:10000], trainlabels[:10000], 0.01)
    correct = 0
    all = 0
    size, = testimages[:, 0, 0].shape
    for i in range(0, size):
        t = predict(testimages[i], kernel_weights, output_weights)
        if np.argmax(t) == testlabels[i]:
            correct = correct + 1
        all = all + 1
    result = (correct / all) * 100
    print(result, '%')

