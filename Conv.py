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

    return(train_images, test_images)


class Conv():
    """
    Simple convolution model
    predict:
        make a prediction of image
    Layer2D:
        create convolution layer (2D)
    """

    def __init__(self):
        self.layers = []
        self.pool_size = None
        pass

    def predict(self, file_name=None):
        """
        make a prediction of image
        :param file_name:
            name of file with weight
        :return:
            label
        """
        pass

    def Layer2D(self, data, kernel, kernel_size, layer_id, strides=(1, 1), border_mode='valid'):
        """
        Create convolution layer
        :param data:
            image 28x28
        :param kernel_size:
            size of filter
        :param strides:
            filter "step"
        :param border_mode:
            mode which says how filter will work with map
            if "valid":
                take map and dont add some fields to it
            if "full":
                add to map some fields(with 0) for filter

        layer structure:
            layer(list of tuples)[
                    {number: [bias, [filters], [new map]]},
                ]
        :return:
            new convolution map
        """
        # Создать слой
        # если нет айди то создать
        # если есть изменить значение
        # 1) фильтра
        # 2) новое поле
        # 3) биас

        if layer_id in self.layers:
            pass
        else:
            # edit current layer
            # self.layers[layer_od]
            pass
        # self.layers.append(np.random.rand(kernel, kernel_size))
        # bias =
        # Create empty map
        if border_mode == 'valid':
            pass
        elif border_mode == 'full':
            pass
        else:
            pass

        # Проходить по всем елементам всей карты и собирать сумму в новую карту
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


    # plt.imshow(test, cmap='Greys')
    #
    # plt.show()