import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
from timeit import default_timer as timer

img_size = 784
h_size = 100

m = 15000
ts_m = 10000

def main():
    start = timer()
    print("--------Parsing data----------------------")
    tr_features, tr_labels, ts_features, ts_labels_cm = parse_data()
    ts_labels = one_hot_encode(ts_labels_cm)
    tr_labels = one_hot_encode(tr_labels)
    print("Training input shape: {}".format(tr_features.shape))
    print("Training labels shape: {}".format(tr_labels.shape))
    end = timer()
    print("Parse data complete. Time taken: {} seconds".format(end-start))
    print("------------------------------------------")

    network_10 = Network(img_size,h_size,10)

    network_10.train(0.1, 0.9, tr_features, tr_labels, ts_features, ts_labels, ts_labels_cm, 50)

class Network(object):
    def __init__(self, input_size, hidden_size, output_size):
        self.in_hidden_weights = self.init_weights(input_size+1, hidden_size)
        self.hidden_out_weights = self.init_weights(hidden_size+1, output_size)
        # Store prev weights for delta w calculations
        self.prev_w_ih = np.zeros(self.in_hidden_weights.shape)
        self.prev_w_ho = np.zeros(self.hidden_out_weights.shape)

    def init_weights(self, r, c):
        w = np.random.uniform(low=-0.05, high=0.05, size=(r, c))
        w = np.around(w, decimals=2)
        return w

    def compute_target_values(self, label):
        T = []
        for value in label:
            t = 0.9 if value==1 else 0.1
            T.append(t)
        return np.array(T)

    # Feed in numpy array activation values from output layer
    # then return index of output node with maximum value (Prediciton)
    def get_prediction_index(self, O):
        max = np.argmax(O)
        #return one_hot_encode(max)
        return max

    def feed_forward(self, x):
        # Input to hidden layer
        Zh = np.dot(x, self.in_hidden_weights)
        H = sigmoid(Zh)
        H = np.insert(H, 0, 1)  # Prepend 1 for bias
        H = H.reshape((1,h_size+1)) # 2D -> 1D array
        H = np.ravel(H)

        # Hidden to output layer
        Zo = np.dot(H, self.hidden_out_weights)
        O = sigmoid(Zo)
        O = O.reshape((1,10))

        return H, O

    def back_propagation(self, O, H, label):
        # Get target values
        T = self.compute_target_values(label)

        # Compute output error terms
        Eo = O * (1 - O) * (T - O)
        assert(Eo.shape == ((1,10)))

        # Compute hidden error terms
        dot = np.dot(self.hidden_out_weights[1:], Eo.T)
    
        sig_prime = (H[1:] * (1 - H[1:]))
        sig_prime = sig_prime.reshape((h_size,1))
        
        Eh = sig_prime.T * dot.T

        return Eo, Eh

    def update_weights(self, Eo, Eh, H, X, learning_rate, momentum):
        #Compute delta, update weights, save current delta for next iteration
        delta_w = (learning_rate * Eo.T * H).T + (momentum * self.prev_w_ho)

        self.hidden_out_weights += delta_w
        self.prev_w_ho = delta_w

        # Update input to hidden
        delta_w = (learning_rate * Eh.T * X).T + (momentum * self.prev_w_ih)
        self.in_hidden_weights += delta_w
        self.prev_w_ih = delta_w

    def train(self, learning_rate, momentum, tr_inputs, tr_labels, ts_inputs, ts_labels, ts_labels_cm, nb_epoch):
        tr_acc_data = []
        ts_acc_data = []
        prediction_data = []
        for epoch in range(nb_epoch+1):
            tr_incorrect = 0
            ts_incorrect = 0
            start = timer()
            # Loop Through each example
            for input, label in zip(tr_inputs, tr_labels):
                H, O = self.feed_forward(input)
                prediction = self.get_prediction_index(O)
                if prediction != one_hot_to_number(label):
                    tr_incorrect += 1
                if epoch>0:
                    Eo, Eh = self.back_propagation(O, H, label)
                    input = input.reshape((1, 785))
                    self.update_weights(Eo, Eh, H, input, learning_rate, momentum)
            # Accuracy on test set
            for input, label in zip(ts_inputs, ts_labels):
                H, O = self.feed_forward(input)
                prediction = self.get_prediction_index(O)
                # For confusion matrix (Runs on last epoch)
                if epoch == nb_epoch:
                    prediction_data.append(prediction)
                if prediction != one_hot_to_number(label):
                    ts_incorrect += 1
            end = timer()
            # Time elapsed
            print("Epoch {} \t time elapsed: {}".format(epoch, end-start))
            tr_accuracy = ((m - tr_incorrect) / m) * 100
            ts_accuracy = ((ts_m - ts_incorrect) / ts_m) * 100
            tr_acc_data.append(tr_accuracy)
            ts_acc_data.append(ts_accuracy)

            # Evaluate training accuracy
            print("Training set accuracy: {} %".format(tr_accuracy))
            print("Testing set accuracy: {} %".format(ts_accuracy))
            print("------------------------------------")

        cm = confusion_matrix(ts_labels_cm, np.array(prediction_data))
        df_cm = pd.DataFrame(cm, index=[i for i in "0123456789"],
                             columns=[i for i in "0123456789"])
        plt.figure(figsize=(10, 10))
        sn.heatmap(df_cm, annot=True, fmt = '.1f')

        plt.figure(figsize=(10,10))
        epoch_data = range(nb_epoch+1)
        plt.title("Accuracy for learning rate: {}".format(learning_rate))
        plt.plot(epoch_data, tr_acc_data, label = "Training")
        plt.plot(epoch_data, ts_acc_data, label="Testing")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy %")
        plt.legend()
        plt.show()


def one_hot_encode(labels):
    nb_labels = len(labels)
    nb_categories = 10
    one_hot = np.zeros((nb_labels, nb_categories))
    one_hot[np.arange(nb_labels), labels] = 1
    return one_hot

def one_hot_to_number(label):
    return np.argmax(label)

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def parse_data():
    train_data = pd.read_csv('data/mnist_train.csv', header=None, sep=',', engine='c', na_filter= False, low_memory=False)
    test_data = pd.read_csv('data/mnist_test.csv', header=None, sep=',', engine='c', na_filter=False, low_memory=False)

    tr_labels = train_data.iloc[:, 0]
    tr_labels = tr_labels[:15000]
    print(tr_labels.value_counts())    # Check dataset is balanced
    train_data /= 255
    train_data.iloc[:,0] = 1.0
    #tr_features = train_data
    tr_features = train_data[:15000]    #Only use half of training set

    ts_labels = test_data.iloc[:, 0]
    test_data /= 255
    test_data.iloc[:,0] = 1.0
    ts_features = test_data

    return np.array(tr_features), np.array(tr_labels), np.array(ts_features), np.array(ts_labels)

if __name__ == '__main__':
    main()