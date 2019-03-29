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

    def CreateLayer2D(self, data, kernel, kernel_size, layers_num=1):
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
        conv_dim = np.shape(data)[1] - kernel_size[0] + 1
        # create layer
        for layer_id in range(layers_num):
            self.conv_layers[layer_id] = np.array([
                # bias
                np.random.rand(),
                # new random kernel
                np.random.rand(kernel, kernel_size[0], kernel_size[1]),
                # new empty map for each kernel
                np.zeros([kernel, conv_dim, conv_dim])
            ])

    def covolution(self, data, layer):
        for image, i in enumerate(data):
            for feature in layer[1]:
                # преобразование Фурье?
                print(feature)
                break
            break

    def CreateDense(self):
        pass

    def fit(self, data):
        self.CreateLayer2D(data, 5, [5, 5])
        self.covolution(data, self.conv_layers[0])
        pass

    def MaxPooling2D(self):
        pass

    def Flatten(self):
        pass

    def Dropout(self):
        pass

    def Desnse(self):
        pass


if __name__ == '__main__':

    train_images = pd.read_csv('data/train_images.csv')
    test_images = pd.read_csv('data/test_images.csv')

    train_labels = pd.read_csv('data/train_labels.csv')
    test_labels = pd.read_csv('data/test_labels.csv')

    train_images, test_images = share_image(train_images, test_images)

    model = Conv()

    model.fit(train_images)
    # model.CreateLayer2D(train_images, 10, [5,5], 1)
    # plt.imshow(test, cmap='Greys')
    #
    # plt.show()