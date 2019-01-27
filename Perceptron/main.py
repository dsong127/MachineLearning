import numpy as np
import pandas as pd
from timeit import default_timer as timer
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle

img_size = 784

def main():
    tr_features, tr_labels, ts_features, ts_labels = parse_data()

    network = Network(10)
    network.Perceptron(img_size)
    network.train(tr_features, tr_labels, ts_features, ts_labels)

def parse_data():
    '''
    Input datas are extracted from csv files and scaled by 255.
    First column is stored in labels variable, then replaced by a column of 1's (Bias)
    :return: Preprocessed inputs and labels
    '''
    train_data = pd.read_csv("data/mnist_train.csv", header=None, sep=",")
    test_data = pd.read_csv("data/mnist_test.csv", header=None, sep=",")

    train_data = shuffle(train_data)
    test_data = shuffle(test_data)

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

class Network(object):
    class Perceptron(object):
        def __init__(self, input_size):
            self.weights = self.init_weights(input_size)

        def init_weights(self, input_size):
            w = np.random.uniform(low=-0.05, high=0.05, size=(input_size+1))
            w = np.around(w, decimals = 2)
            return w

    def __init__(self, nb_perceptrons):
        self.nb_perceptrons = nb_perceptrons
        self.perceptrons = [self.Perceptron(img_size) for _ in range(self.nb_perceptrons)]

    def output(self, sum):
        if sum > 0:
            y = 1
        else:
            y = 0
        return y

    def dot_product(self, perceptron, inputs):
        x = perceptron.weights
        w = inputs
        sum = np.dot(w, x)
        return sum

    def target(self, perceptron_class, label):
        if perceptron_class == label:
            t = 1
        else:
            t = 0
        return t

    def train(self, inputs, labels, ts_inputs, ts_labels, learning_rate = 1, nb_epoch=50):
        tr_acc_data = []
        ts_acc_data = []
        prediction_data = []
        for epoch in range(nb_epoch+1):
            start = timer()
            tr_incorrect =0
            ts_incorrect = 0

            for input, label in zip(inputs, labels):
                tr_max = {"output": 0, "index": 0}
                tr_prediction = 0

                # 1. For each training example, loop through all 10 perceptrons to compute wx, y, t
                # 2. Predict by finding the perceptron with max wx, and storing its class (Index)
                # Note: Following for loop can be replaced by performing dot product of each input row with weights of all perceptrons

                for idx, p in enumerate(self.perceptrons):
                    s = self.dot_product(p, input)
                    y = self.output(s)
                    t = self.target(idx, label)

                    # Predict output
                    if s > tr_max["output"]:
                        tr_max["output"] = s
                        tr_max["index"] = idx
                    # Adjust weights ie. stochastic gradient (After 0th epoch)
                    if epoch > 0:
                       for i in range(p.weights.size):
                            p.weights[i] += learning_rate * (t - y) * input[i]
                tr_prediction = tr_max["index"]
                # Was the prediction correct?
                if tr_prediction != label:
                    tr_incorrect += 1

            end = timer()
            tr_acc = ((labels.size - tr_incorrect) / labels.size) * 100
            tr_acc_data.append(tr_acc)
            print("Time elasped: {}".format(end - start))
            print("Epoch {}: # of incorrects: {}, labels size: {} accuracy: {}".format(epoch, tr_incorrect, labels.size,  tr_acc))

            # Calculate accuracy on test set
            start = timer()
            for ts_input, ts_label in zip(ts_inputs, ts_labels):
                ts_max = {"output": 0, "index": 0}
                ts_prediction = 0
                for idx, p in enumerate(self.perceptrons):
                    s = self.dot_product(p, ts_input)
                    y = self.output(s)
                    t = self.target(idx, ts_label)
                    if s > ts_max["output"]:
                        ts_max["output"] = s
                        ts_max["index"] = idx
                ts_prediction = ts_max["index"]
                if epoch == nb_epoch:
                    prediction_data.append(ts_prediction)
                # Was the prediction correct?
                if ts_prediction != ts_label:
                    ts_incorrect += 1

            end = timer()
            ts_acc = ((ts_labels.size - ts_incorrect) / ts_labels.size) * 100
            ts_acc_data.append(ts_acc)
            print("Time elasped: {}".format(end - start))
            print("Epoch {} (Test set): # of incorrects: {}, labels size: {} accuracy: {}".format(epoch, ts_incorrect, ts_labels.size, ts_acc))

        # Confusion matrix and plots
        cm = confusion_matrix(ts_labels, np.array(prediction_data))
        print(cm)
        epoch_data = range(nb_epoch+1)
        plt.title("Accuracy for learning rate: {}".format(learning_rate))
        plt.plot(epoch_data, tr_acc_data, label = "Training")
        plt.plot(epoch_data, ts_acc_data, label="Testing")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy %")
        plt.legend()
        plt.show()

if __name__ == '__main__':
    main()