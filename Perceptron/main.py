import numpy as np
from timeit import default_timer as timer
from utils import parse_data
from matplotlib import pyplot as plt

img_size = 784

def main():
    tr_features, tr_labels, ts_features, ts_labels = parse_data()

    #one_hot_encode(tr_labels)
    #one_hot_encode(ts_labels)

    network = Network(10)
    network.Perceptron(img_size)
    network.train(ts_features, ts_labels)

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

    def train(self, inputs, labels, learning_rate = 0.01, nb_epoch=10):
        tr_acc_data = []
        ts_acc_data = []
        for epoch in range(nb_epoch+1):
            start = timer()
            incorrect = 0

            for input, label in zip(inputs, labels):
                max = {"output": 0, "index": 0}
                prediction = 0
                # 1. For each training example, loop through all 10 perceptrons to compute wx, y, t
                # 2. Predict by finding the perceptron with max wx, and storing its class (Number)

                for idx, p in enumerate(self.perceptrons):
                    s = self.dot_product(p, input)
                    y = self.output(s)
                    t = self.target(idx, label)

                    # Predict output
                    if s > max["output"]:
                        max["output"] = s
                        max["index"] = idx

                    # Adjust weights ie. stochastic gradient (After 0th epoch)
                    if epoch > 0:
                        for i in range(p.weights.size):
                            p.weights[i] += learning_rate * (t - y) * input[i]

                prediction = max["index"]
                # Was the prediction correct?
                if prediction != label:
                    incorrect += 1

            end = timer()
            print("Time elasped: {}".format(end - start))
            acc = ((labels.size - incorrect) / labels.size) * 100
            tr_acc_data.append(acc)
            print("Epoch {}: # of incorrects: {}, labels size: {} accuracy: {}".format(epoch, incorrect, labels.size,  acc))

        epoch_data = range(nb_epoch+1)
        plt.plot(epoch_data, tr_acc_data, label = "Training")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy %")
        plt.legend()
        plt.show()



if __name__ == '__main__':
    main()