import numpy as np


def tangleImage(input_image, filters, step, padding):
    padded_image = np.pad(input_image, padding)
    if filters.shape[0] != filters.shape[1] or padded_image.shape[0] != padded_image.shape[1]:
        return 0
    output_xaxis = int(((padded_image.shape[1]-filters.shape[1])/step)+1)
    output_yaxis = int(((padded_image.shape[0]-filters.shape[0])/step)+1)
    output_image = np.zeros((output_yaxis, output_xaxis))
    filtered_y = 0
    for y in range(output_image.shape[0]):
        filtered_x = 0
        for x in range(output_image.shape[1]):
            output_image[y, x] = np.sum(padded_image[filtered_y:(filters.shape[0]+filtered_y), filtered_x:(filters.shape[1]+filtered_x)]*filters)
            filtered_x += step
        filtered_y += step
    return output_image


if __name__ == '__main__':
    input_image = np.array([[1, 1, 1, 0, 0], [0, 1, 1, 1, 0], [0, 0, 1, 1, 1], [0, 0, 1, 1, 0], [0, 1, 1, 0, 0]])
    filters = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    output_image = np.array([[4, 3, 4], [2, 4, 3], [2, 3, 4]])
    print(tangleImage(input_image, filters, 1, 0))
