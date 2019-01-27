import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from timeit import default_timer as timer

img_size = 784
h_size = 10

def main():
    tr_features, tr_labels, ts_features, ts_labels = parse_data()
    network = Network(img_size,h_size,10)
    w1 = network.in_hidden_weights
    w2 = network.hidden_out_weights

    assert(w1.shape == (h_size, img_size+1))
    assert(w2.shape == (10, h_size+1))


class Network(object):
    def __init__(self, input_size, hidden_size, output_size):
        self.in_hidden_weights = self.init_weights(hidden_size, input_size+1)
        self.hidden_out_weights = self.init_weights(output_size, hidden_size+1)

    def init_weights(self, r, c):
        w = np.random.uniform(low=-0.05, high=0.05, size=(r, c))
        w = np.around(w, decimals=2)
        return w

    def compute_target(self, O, label):
        T = []
        for index in O.shape[0]:
            t = 0.9 if index==label else 0.1
            T.append(t)
        return T

    # Feed in numpy array activation values, then return index of output node with maximum value (Prediciton)
    def get_prediction_index(A):
        max = np.argmax(A)
        return max

    def propagate(self, input, label, learning_rate, momentum):
        # Propagate forward
        H = [1]
        O = []
        z = np.dot(self.in_hidden_weights, input.T)
        for w, in self.in_hidden_weights:
            z = np.dot(w, input.T)
            H.append(sigmoid(z))
        H = np.array(H).reshape((h_size,1))

        for w in self.hidden_out_weights:
            z = np.dot(w, H.T)
            O.append(sigmoid(z))
        O = np.array(O).reshape((10,1))

        # Calculate error terms
        O_errors = []
        for i, output in enumerate(O):
            T = self.compute_target(O, label)
            for t in T:
                error = output * (1-output) * (t - output)
                O_errors.append(error)
            O_errors = np.array(O_errors).reshape((10,1))

        H_errors = []
        for h, w in zip(H, self.hidden_out_weights):
            error = h* (1-h) * (np.dot(w,O_errors))
            H_errors.append(error)

        # Update weights
        # Calculate delta_w
        delta_ws = []
        prev_delta_w = 0
        for h, h_error in zip(H, H_errors):
            delta_w = learning_rate * h_error * h + momentum * prev_delta_w
            delta_ws.append(delta_w)
            prev_delta_w = delta_w
        for weights in self.hidden_out_weights:
            for w in weights:
                w = w +





    def train(self, learning_rate, tr_inputs, tr_labels, ts_inputs, ts_labels, nb_epoch):
        for epoch in range(nb_epoch+1):
            hidden_activation = np.ones((1, h_size+1)) # Vector to store activation values of hidden layer
            for input, label in zip(tr_inputs, tr_labels):










def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def parse_data():
    train_data = pd.read_csv('data/mnist_train.csv', header=None, sep=',', engine='c', na_filter= False, low_memory=False)
    test_data = pd.read_csv('data/mnist_train.csv', header=None, sep=',', engine='c', na_filter=False, low_memory=False)

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

if __name__ == '__main__':
    main()