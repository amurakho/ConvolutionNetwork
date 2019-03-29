"""

Create Convolution Network with hardcode
It will be really simple, becose i make it only for training
So this network will categorize... numbers, yeap
again numbers(with ENG alphabet)

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

def share_image():
    """
    reshape image - make in same size(28x28)
    separate to channels
    """
    pass

class Conv():
    pass


if __name__ == '__main__':

    df = pd.read_csv('data/train_images.csv')

    test = np.array(df.iloc[0]).reshape(28, 28)

    plt.imshow(test, cmap='Greys')

    plt.show()