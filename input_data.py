from mlxtend.data import loadlocal_mnist
import numpy as np
import pandas as pd

X, y = loadlocal_mnist(
        images_path='data/train-images-idx3-ubyte',
        labels_path='data/train-labels-idx1-ubyte')

# print('Dimensions: %s x %s' % (X.shape[0], X.shape[1]))
# print('\n1st row', X[0])
# print('**********************')


test = X[0].reshape(28, 28)

print(np.shape(test))
# np.savetxt(fname='data/train_images.csv',
#            X=X, delimiter=',', fmt='%d')
# np.savetxt(fname='data/train_labels.csv',
#            X=y, delimiter=',', fmt='%d')

# df = pd.read_csv('data/train_images.csv')

# print(df)