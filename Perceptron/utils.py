import numpy as np
import pandas as pd
from sklearn.utils import shuffle

def parse_data():
    train_data = pd.read_csv("data/mnist_train.csv", header=None, sep=",")
    test_data = pd.read_csv("data/mnist_test.csv", header=None, sep=",")

    #train_data = shuffle(train_data)
    #test_data = shuffle(test_data)

    tr_labels = train_data.iloc[:, 0]
    train_data /= 255
    train_data.iloc[:,0] = 1.0
    tr_features = train_data

    ts_labels = test_data.iloc[:, 0]
    test_data /= 255
    test_data.iloc[:,0] = 1.0
    ts_features = test_data

    tr_labels = np.array(tr_labels).reshape(60000, 1)
    ts_labels = np.array(ts_labels).reshape(10000, 1)


    return np.array(tr_features), tr_labels, np.array(ts_features), ts_labels

def one_hot_encode(labels):
    labels_size = len(labels)
    num_of_classes = 10
    one_hot_encode = np.zeros((labels_size, num_of_classes))
    one_hot_encode[np.arange(labels_size), labels] = 1
    return one_hot_encode


'''
:param inputs: Training input data i.e array containing arrays of pixel values
:param labels: target values to compare predictions to

1. Load an array of pixel data for a MNIST digit, and its corresponding label
2. Loop through all 10 perceptrons and set:
    1. Loop through inputs and labels
    1. Calculate perceptron's output (predict function)
    2. Calculate loss and update weights
3. Get classification using the index of the highest output
4. Repeat for next image
'''