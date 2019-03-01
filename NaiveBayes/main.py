import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
import seaborn as sn
import matplotlib.pyplot as plt

nb_tr = 2300
nb_ts = 2301


def main():
    tr_x, tr_y, ts_x, ts_y = preprocess()
    pred_data = classify_data(tr_x, tr_y, ts_x)

    evaluate(pred_data, ts_y, 'nb')

    # Logistic Regression
    clf = LogisticRegression(solver='liblinear')
    tr_y = np.reshape(tr_y, (2300,))
    clf.fit(tr_x, tr_y)
    prediction =  clf.predict(ts_x)

    evaluate(prediction, ts_y, 'lr')

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

    #print(ts_y.value_counts())

    return np.array(tr_x), np.array(tr_y).reshape(2300,1), np.array(ts_x), np.array(ts_y).reshape(2301,1)

def prob_model(tr_x, tr_y):
    tr_prob_model, c1 = np.unique(tr_y, return_counts=True)
    p_not = c1[0] / nb_tr
    p_spam = c1[1] / nb_tr

    # Check if index slicing is doing-------------------
    std_spam = np.std(tr_x[0:906], axis = 0)
    std_not = np.std(tr_x[906:2300], axis = 0)

    # Add 0.0001 to std to avoid 0s
    std_spam += 0.0001
    std_not += 0.0001

    mean_spam = np.mean(tr_x[0:906], axis = 0)
    mean_not = np.mean(tr_x[906:2300], axis=0)

    return p_spam, p_not, std_spam, std_not, mean_spam, mean_not

def classify_data(tr_x, tr_y, ts_x):
    p_spam, p_not, std_spam, std_not, mean_spam, mean_not = prob_model(tr_x, tr_y)
    spam_predict = np.empty((0,58))
    not_predict = np.empty((0,58))

    first_spam = np.log(compute_first(std_spam))
    first_not = np.log(compute_first(std_not))

    # ln(First) + (x-mean)^2 / (2(std)^2)
    # Compute PDFs
    # Spam
    for x in ts_x:
        second = compute_second(x, mean_spam, std_spam)
        pdf = first_spam + second
        #Prepend ln(P(Spam))
        pdf = np.insert(pdf, 0, np.log(p_spam))
        pdf = np.reshape(pdf, (1, 58))
        spam_predict = np.append(spam_predict, pdf, axis = 0)

    # Not Spam
    for x in ts_x:
        second = compute_second(x, mean_not, std_not)
        pdf = first_not + second
        pdf = np.insert(pdf, 0, np.log(p_not))
        pdf = np.reshape(pdf, (1, 58))
        not_predict = np.append(not_predict, pdf, axis=0)

    # Classification
    final_predict_spam = np.sum(spam_predict, axis =1)
    final_predict_not = np.sum(not_predict, axis=1)

    prediction_data = []
    for x, y in zip(final_predict_spam, final_predict_not):
        if x > y:
            prediction_data.append(1)
        else:
            prediction_data.append(0)

    return np.array(prediction_data)

def evaluate(prediction, label, alg):
    correct = 0
    for x, y in zip(prediction, label):
        if x == y:
            correct += 1

    cm = confusion_matrix(label, prediction)
    tn, fp, fn, tp = cm.ravel()
    accuracy = (correct / 2301) * 100
    precision = (tp / (tp + fp)) * 100
    recall = (tp / (tp + fn)) * 100

    print("Accuracy: {}".format(accuracy))
    print("Precision: {} \t Recall: {}".format(precision, recall))

    plt.figure(figsize=(10,7))
    sn.heatmap(cm, annot=True, fmt = '.1f')
    plt.title('1 = spam 0 = not spam')
    plt.xlabel('Predicted')
    plt.ylabel('True label')
    if alg == 'nb':
        plt.savefig('plot.png', bbox_inches='tight')
    else:
        plt.savefig('plot_lr.png', bbox_inches='tight')


def compute_first(std):
    return (1 / (np.sqrt(2* np.pi) * std))

def compute_second(x, mean, std):
    return -1 * (np.square(x - mean) / (2 * np.square(std)))


if __name__ == '__main__':
    main()