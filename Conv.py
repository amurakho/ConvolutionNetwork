"""

Create Convolution Network with hardcode
It will be really simple, becose i make it only for training
So this network will categorize... numbers, yeap
again numbers(with ENG alphabet)

Make hyperparams and skeleton from
https://towardsdatascience.com/image-classification-in-10-minutes-with-mnist-dataset-54c35b77a38d
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.ndimage

def share_image(train_images, test_images):
    """
    reshape image - make in same size(28x28)
    simple normilize data
    """
    train_images = np.array(train_images)
    test_images = np.array(test_images)

    train_images = train_images.reshape(np.shape(train_images)[0], 28, 28)
    test_images = test_images.reshape(np.shape(test_images)[0], 28, 28)

    train_images = train_images / 255
    test_images = test_images / 255

    return train_images, test_images


class Conv():
    """
    Simple convolution model
    predict:
        make a prediction of image
    Layer2D:
        create convolution layer (2D)
    """

    def __init__(self):
        self.conv_layers = {}
        self.dense_layers = {}
        self.pool_size = None

    def predict(self, file_name=None):
        """
        make a prediction of image
        :param file_name:
            name of file with weight
        :return:
            label
        """
        pass

    def createLayer2D(self, data, kernel, kernel_size, layers_num=1):
        """
        Create convolution layer
        :param data:
            image 28x28
        :param kernel:
            number of kernels
        :param kernel_size:
            size of filter
        :param layers_num:
            layers number

        layer structure:
            layer(list of tuples)[
                    {number: [bias, [filters], [new map]]},
                ]
        :return:
            new convolution map
        """

        # dimension of new conv map
        # conv_dim = np.shape(data)[1] - kernel_size[0] + 1
        # conv_dim = np.shape(data)[1]
        # create layer
        for layer_id in range(layers_num):
            self.conv_layers[layer_id] = np.array([
                # bias
                np.random.uniform(-0.5, 0.5),
                # new random kernel
                np.random.uniform(-0.5, 0.5, [kernel, kernel_size[0], kernel_size[1]]),
                # new empty map for each kernel
                # np.zeros([conv_dim, conv_dim])
            ])

    def covolution(self, image, layer):
        """
        Take a map and make convolution for it
        :param image:
            map
        :param layer:
            the layer from wich will take the filter
        :return:
            new convolution map
        """
        # for each filter in layer
        map_dim = np.shape(image)[0]
        map = np.zeros([map_dim, map_dim])
        for feature in layer[1]:
            map += scipy.ndimage.convolve(image, feature)
        return map

    def createDense(self, output_num, layer_id, input_num):
        """
        Init the weights and bias for each neuron in dense layer
        :param neuron_num:
            number of neurons for each layer
        :param layer_id:
            id of new layer
        """
        # self.dense_layers[layer_id] = []
        # for neuron in range(output_num):
        #     # weight for each neuron for each input
        #     weight = np.random.rand(input_num)
        #     # bias for each neuron
        #     bias = np.random.rand()
        #     self.dense_layers[layer_id].append([weight, bias])

        self.dense_layers[layer_id] = {}
        self.dense_layers[layer_id]['weights'] = np.random.uniform(-0.5, 0.5, [output_num, input_num])
        self.dense_layers[layer_id]['bias'] = np.random.uniform(-0.5, 0.5, output_num)

        # for neuron in range(output_num):
        #     # weight for each neuron for each input
        #     self.dense_layers[layer_id]['weight'] = np.random.rand(input_num)
        #     # bias for each neuron
        #     self.dense_layers[layer_id]['bias'] = np.random.rand()
        #     # self.dense_layers[layer_id].append([weight, bias])

    def fit(self, data, kernel, kernel_size):
        # create two conv layers
        # create few full network layers
        self.createLayer2D(data, kernel, kernel_size, layers_num=2)
        # create dense layer which have 49 neurons(for each input after conv)
        self.createDense(layer_id=0, output_num=10, input_num=49)
        # create dense layer which will work with softmax act. func
        # self.createDense(layer_id=1, output_num=10, input_num=49)
        # start trainig
        ress = []
        for image in data:
            x = self.covolution(image, self.conv_layers[0])
            x = self.relu(x)
            x = self.maxPooling2D([2, 2], x)
            x = self.covolution(x, self.conv_layers[1])
            x = self.relu(x)
            x = self.maxPooling2D([2, 2], x)
            x = self.flatten(x)
            x = self.dropout(x, 0.3)
            x = self.dense(x, self.dense_layers[0])
            x = self.relu(x)
            x = self.dropout(x, 0.2)
            res = self.softmax(x)
            break
        pass

    def maxPooling2D(self, pool_size, map):
        """
        Take the map and make a pooling with pool filter
        :param pool_size:
            Pool size
        :param map:
            map
        :return:
            new map after pooling
        """
        # find dimension for new map
        # map dimension / pool size
        new_shape = int(np.shape(map)[0] / pool_size[0])
        # create empty map
        pool_map = np.zeros([new_shape, new_shape])

        # each row in new map
        for rows_counter in range(new_shape):
            # take start and end from old map
            start_row = rows_counter * pool_size[0]
            end_row = start_row + pool_size[0]

            # each colums
            for cols_counter in range(new_shape):
                start_col = cols_counter * pool_size[1]
                end_col = start_col + pool_size[1]

                # take patch from old map
                patch = map[start_row:end_row, start_col:end_col]
                # take maximum
                pool_map[rows_counter][cols_counter] = np.max(patch)

        return pool_map

    def flatten(self, map):
        """
        Take the map matrix and create 2D array
        :return:
            2D array
        """
        flatten_map = np.ndarray.flatten(map)
        return flatten_map

    def dropout(self, map, propability):
        """
        Some random neurons(values in matrix) turn off(make it to zero)
        :return:
            new dropout map
        """
        dimension = np.shape(map)[0]
        prob = 1. - propability
        binomial_layer = np.random.binomial(1, prob, dimension)
        map = map * binomial_layer
        return map

    def dense(self, map, layer):
        """
        Make weighted sum
        :param map:
            flatten list
        :param layer:
            the layer with weights
        :return:
            list of weighted sum from each neuron
        """
        results = np.dot(layer['weights'], map) + layer['bias']
        return results

    def relu(self, map):
        """
        Relu function
        make all values which value is less then 0. to zero
        :param map:
            map after conv
        :return:
            new relu map
        """
        z = np.zeros_like(map)
        return np.where(map > z, map, z)

    def softmax(self, map):
        """
        Softmax activation function
        ei^z/sum(ei)
        :param map:
            map after dense
        :return:
            index of max
        """
        numerator = np.exp(map)
        denominator = np.sum(np.exp(map))
        res = numerator / denominator
        return np.argmax(res)


if __name__ == '__main__':

    train_images = pd.read_csv('data/train_images.csv')
    test_images = pd.read_csv('data/test_images.csv')

    train_labels = pd.read_csv('data/train_labels.csv')
    test_labels = pd.read_csv('data/test_labels.csv')

    train_images, test_images = share_image(train_images, test_images)

    model = Conv()

    model.fit(train_images, 5, [5, 5])
    # model.CreateLayer2D(train_images, 10, [5,5], 1)
    # plt.imshow(test, cmap='Greys')
    #
    # plt.show()

