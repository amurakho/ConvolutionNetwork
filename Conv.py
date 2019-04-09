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
        # coefitients for ADAM
        self.v = {}
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

    def createLayer2D(self, kernel, kernel_size, layer_id):
        """
        Create convolution layer
        :param kernel:
            number of kernels
        :param kernel_size:
            size of filter
        :param layer_id:
            layers id

        layer structure:
            layer(list of tuples)[
                    {number: [[filters], [new map]]},
                ]
        :return:
            new convolution map
        """

        # create layer
        self.conv_layers[layer_id] = np.array([
            # new random kernel
            np.random.uniform(-0.5, 0.5, [kernel, kernel_size[0], kernel_size[1]]),
            # for ADAM coefitients
            np.zeros([kernel, kernel_size[0], kernel_size[1]], dtype='float32')
        ])

    def createDense(self, output_num, layer_id, input_num):
        """
        Init the weights and bias for each neuron in dense layer
        :param neuron_num:
            number of neurons for each layer
        :param layer_id:
            id of new layer
        """
        self.dense_layers[layer_id] = {}
        # init weights
        self.dense_layers[layer_id]['weights'] = np.random.uniform(-0.5, 0.5, [output_num, input_num])
        # init bias
        self.dense_layers[layer_id]['bias'] = np.random.uniform(-0.5, 0.5, output_num)
        # init adam coefitients
        self.dense_layers[layer_id]['adam'] = np.zeros([output_num, input_num], dtype='float32')

    def fit(self,
            data,
            labels,
            kernel,
            kernel_size,
            epochs,
            learning_rate=0.1):
        # create two conv layers
        self.createLayer2D(kernel, kernel_size, layer_id=0)
        self.createLayer2D(kernel, kernel_size, layer_id=1)
        # create fully connected layer which have 49 neurons(for each input after conv)
        self.createDense(layer_id=0, output_num=10, input_num=49)
        ress = []
        # start trainig
        for epoch in range(epochs):
            x = self.covolution(data, self.conv_layers[0])
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
            loss = self.cross_entropy(res, labels)
            print(loss)
            # self.adam(loss, epoch, learning_rate)
            break
        pass

    def covolution(self, images, layer):
        """
        Take a map of images and make convolution for it
        :param images:
            map
        :param layer:
            the layer from wich will take the filter
        :return:
            new convolution map
        """
        nb_images = np.shape(images)[0]
        map_dim = np.shape(images)[1]

        map = np.zeros([nb_images, map_dim, map_dim])

        # for each image
        for image_idx in range(nb_images):
            # for each filter in layer
            for feature in layer[0]:
                map[image_idx] += scipy.ndimage.convolve(images[image_idx], feature)
        return map

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
        new_shape = int(np.shape(map)[1] / pool_size[0])
        image_nb = np.shape(map)[0]
        # create empty map
        pool_map = np.zeros([image_nb, new_shape, new_shape])

        # for each image
        for image_idx in range(image_nb):
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
                    patch = map[image_idx][start_row:end_row, start_col:end_col]
                    # take maximum
                    pool_map[image_idx][rows_counter][cols_counter] = np.max(patch)

        return pool_map

    def flatten(self, map):
        """
        Take the map matrix and create 2D array
        :return:
            2D array
        """
        image_nb = np.shape(map)[0]
        flatten_map = np.zeros((image_nb, np.prod(np.shape(map)[1:])))
        for image_idx in range(image_nb):
            flatten_map[image_idx, :] = np.ndarray.flatten(map[image_idx])
        return flatten_map

    def dropout(self, map, propability):
        """
        Some random neurons(values in matrix) turn off(make it to zero)
        :return:
            new dropout map
        """
        dimension = np.shape(map)
        prob = 1. - propability
        binomial_layer = np.random.binomial(1, prob, dimension)
        map = map * binomial_layer
        return map

    def dense(self, map, layer):
        """
        Make weighted sum for each image
        :param map:
            flatten list
        :param layer:
            the layer with weights
        :return:
            list of weighted sum from each neuron
        """
        image_nb = np.shape(map)[0]
        res_dim = np.shape(layer['weights'])[0]
        results = np.zeros([image_nb, res_dim])
        for image_idx in range(image_nb):
            results[image_idx] += np.dot(layer['weights'], map[image_idx]) + layer['bias']
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
            probability to the class
        """
        dim = np.shape(map)
        res = np.zeros(dim)
        for idx in range(dim[0]):
            numerator = np.exp(map[idx])
            denominator = np.sum(np.exp(map[idx]))
            res[idx] += numerator / denominator
        return res

    def classification(self, softmax_prob):
        return np.argmax(softmax_prob)

    def cross_entropy(self, softmax_prob, true_label):
        image_nb = np.shape(softmax_prob)[0]
        loss = np.zeros(image_nb)
        for idx in range(image_nb):
            # take a index of true label
            label_idx = true_label[idx][0]

            loss[idx] += np.log(softmax_prob[idx][label_idx])
        return -np.sum(loss) / image_nb

    def adam(self,
             loss,
             t,
             learning_rate,
             beta1=0.9,
             beta2=0.999,
             epsilon=1e-8):
        """
        Create the ADaM optimizer
        The ADaM is RMSprop + momentum
        :param loss:
            result of loss function
        :param betas:
            betas coefitients for RMSprop and momentum
        :return:
        """
        print(self.v['conv'])
        # for weights
        # self.vdw = beta1 * self.vdw + ((1 - beta1) * loss)
        # self.sdw = beta2 * self.sdw + ((1 - beta2) * loss**2)
        #
        # v_corr = self.vdw / (1 - np.power(beta1, t))
        # s_corr = self.sdw / (1 - np.power(beta2, t))
        #
        # self.dense_layers['weight'] -= (learning_rate * v_corr) / (np.sqrt(s_corr) + epsilon)

        # for bias


if __name__ == '__main__':

    train_images = pd.read_csv('data/train_images.csv')
    test_images = pd.read_csv('data/test_images.csv')

    train_labels = np.array(pd.read_csv('data/train_labels.csv'))
    test_labels = np.array(pd.read_csv('data/test_labels.csv'))

    train_images, test_images = share_image(train_images, test_images)

    model = Conv()

    # for test
    train_images = train_images[:5]
    train_labels = train_labels[:5]

    model.fit(train_images, train_labels, 5, [5, 5], 1)
    # model.CreateLayer2D(train_images, 10, [5,5], 1)
    # plt.imshow(test, cmap='Greys')
    #
    # plt.show()

