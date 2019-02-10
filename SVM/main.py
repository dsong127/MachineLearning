import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_curve
import random
import warnings
warnings.filterwarnings("ignore")

def main():
    tr_x, tr_y, ts_x, ts_y = preprocess()
    assert(tr_x.shape == (2300,57))
    assert(tr_y.shape == (2300, 1))
    assert(ts_x.shape == (2301, 57))
    assert(ts_y.shape == (2301, 1))

    print("---------------------------------")
    print("-------EXP 1---------------------")
    print("---------------------------------")

    # Train a SVM model
    clf = svm.SVC(kernel='linear',probability=True)
    clf.fit(tr_x, tr_y)
    prediction = clf.predict(ts_x)
    probs = clf.predict_proba(ts_x)

    # Accuracy, precision, recall, and ROC curve
    acc = accuracy_score(ts_y, prediction) * 100
    precision = precision_score(ts_y, prediction) * 100
    recall = recall_score(ts_y, prediction) * 100

    probs = probs[:,1]
    fpr, tpr, thresh = roc_curve(ts_y, probs)

    print("Accuracy: {} %".format(acc))
    print("Precision: {} %".format(precision))
    print("Recall: {} %".format(recall))

    # Plot ROC Curve and save plot
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('roc_exp1.png')
    plt.clf()

    print("---------------------------------")
    print("-------EXP 2---------------------")
    print("---------------------------------")
    # Compute weight vector w
    w = np.dot(clf.dual_coef_, clf.support_vectors_)
    w = w.reshape((57,))
    w = np.absolute(w)

    svm_m = svm.SVC(kernel='linear')
    acc_data = []

    # Get new features with highest wight vector value
    for m in range(2, 58):
        new_tr_x, new_ts_x = select_features(w, m, tr_x, ts_x)
        svm_m.fit(new_tr_x, tr_y)
        predict_m = svm_m.predict(new_ts_x)
        acc = accuracy_score(ts_y, predict_m) * 100
        acc_data.append(acc)
        print("M: {}, \t Accuracy: {}".format(m, acc))

    # Plot Accuracy
    plt.title('Number of features vs Accuracy')
    plt.plot(range(2,58), acc_data)
    plt.ylabel('Accuracy %')
    plt.xlim([0, 57])
    plt.ylim([30, 100])
    plt.xlabel('number of features')
    plt.savefig('acc_exp2.png')
    plt.clf()

    print("---------------------------------")
    print("-------EXP 3---------------------")
    print("---------------------------------")

    svm_random = svm.SVC(kernel='linear')
    acc_data_random = []

    for m in range(2, 58):
        random_tr_x, random_ts_x = select_random_features(m, tr_x, ts_x)
        svm_random.fit(random_tr_x, tr_y)
        predict_random = svm_random.predict(random_ts_x)
        acc_random = accuracy_score(ts_y, predict_random) * 100
        acc_data_random.append(acc_random)
        print("M: {}, \t Accuracy: {}".format(m, acc_random))

        # Plot Accuracy
    plt.title('Number of features (Random) vs Accuracy')
    plt.plot(range(2, 58), acc_data_random)
    plt.ylabel('Accuracy %')
    plt.xlim([0, 57])
    plt.ylim([30, 100])
    plt.xlabel('number of features')
    plt.savefig('rand_acc_exp3.png')
    plt.clf()

def preprocess():
    data = pd.read_csv('spambase.data', header=None, sep=',', engine='c', na_filter= False, low_memory=False)
    labels = data.iloc[:, 57]
    # 1 1813
    # 0 2788

    # Split train test 50-50. Make sure # of classes are balanced in each
    tr_x = pd.concat([data.head(906), data.tail(1394)])
    tr_y = pd.concat([labels.head(906), labels.tail(1394)])

    ts_x = pd.concat([data[906:1813], data[1813:3207]])
    ts_y = pd.concat([labels[906:1813], labels[1813:3207]])

    # Remove last column in features
    tr_x = tr_x.drop([57], axis = 1)
    ts_x = ts_x.drop([57], axis = 1)

    # Scale by subtracting mean and dividing by std for each feature
    scaler = preprocessing.StandardScaler()
    tr_x = scaler.fit_transform(tr_x)

    # Scale test data using mean and std computed from training features
    ts_x = scaler.transform(ts_x)

    # If for any feature std is zero, set all values of that feature to zero
    tr_std = np.std(tr_x, axis=1)
    if (np.any(tr_std ==0)):
        std_zero_indices = np.where(tr_std == 0)[0]
        for i in std_zero_indices:
            tr_x[i] = 0
            ts_x[i] = 0

    return np.array(tr_x), np.array(tr_y).reshape(2300,1), np.array(ts_x), np.array(ts_y).reshape(2301,1)

def select_features(arr, m, tr_x, ts_x):
    highest_w_ind = (-arr).argsort()[:m]
    new_tr_x = tr_x[:, highest_w_ind]
    new_ts_x = ts_x[:, highest_w_ind]

    return new_tr_x, new_ts_x

def select_random_features(m, tr_x, ts_x):
    for x in range(m+1):
        ind_random = (random.sample(range(0,57), m))
        # Make sure there's no duplicate
        assert(len(tuple(ind_random)) == len(ind_random))
    random_tr_x = tr_x[:, ind_random]
    random_ts_x = ts_x[:, ind_random]
    return random_tr_x, random_ts_x

if __name__ == '__main__':
    main()