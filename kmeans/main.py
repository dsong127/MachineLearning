import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt
from PIL import Image
np.set_printoptions(suppress=True)

path_tr = 'optdigits/optdigits.train'
path_ts = 'optdigits/optdigits.test'
nb_clutters = 10

class Clutter(object):
    def __init__(self, data):
        self.center = self.random_centroid(data)
        self.members = []
        self.center_same = False
        self.final_members = np.empty((0,0))

    def add_member(self, member):
        self.members.append(member)

    def random_centroid(self, data):
        random_point = data[np.random.randint(data.shape[0])]
        return random_point

    def update_center(self, data):
        member_data = data[self.members]
        new_center = np.mean(member_data, axis = 0)
        if np.array_equal(new_center, self.center):
            self.center_same = True
            self.final_members = self.members
        else:
            self.center = new_center

    def reset_members(self):
        self.members = []

    def __repr__(self):
        return '{} \t final members: {}\n center: {}'.format(self.__class__.__name__, len(self.final_members), self.center)

def evaluate_kmeans(clutters, data, labels, k):
    # Calculate mean square error
    mse_data = []
    for c in clutters:
        member_data = data[c.final_members]
        mse_vector = []
        for x in member_data:
            dist = np.linalg.norm(x- c.center)
            mse_vector.append(dist)
        mse_vector = np.array(mse_vector)
        total_mse = np.mean(mse_vector)
        mse_data.append(total_mse)

    avg_mse = np.mean(mse_data)
    print('average mse: {}'.format(avg_mse))

    # Calculate mean square separation
    mss_data = []
    for i in range(k):
        for j in range(i+1, k):
            dist = np.linalg.norm(clutters[i].center - clutters[j].center)
            mss_data.append(dist)

    mss_sum = np.sum(mss_data)
    mss = mss_sum / ((k*(k-1))/2)
    print('mss: {}'.format(mss))

    entropy_data = []
    total_data = []

    # Convert final members in each clutters to classes(0-9)
    # Then calculate entropy
    for c in clutters:
        classes = labels[c.final_members]
        c_counts = np.array(np.unique(classes, return_counts = True))
        print(c_counts)
        print('------------------------------------')
        total= np.sum(c_counts[1])
        total_data.append(total)
        #print(total)
        entropy = []
        for i in c_counts[1]:
            e = (i/total) * (np.log(i/total)/np.log(2))
            entropy.append(e)
        entropy = -1 * np.sum(entropy)
        entropy_data.append(entropy)
    total_data = np.array(total_data) / data.shape[0]
    mean_entropy = np.dot(total_data, entropy_data)
    print('mean entropy: {}'.format(mean_entropy))

def load_data(path):
    data = pd.read_csv(path, header=None, sep=',', engine='c', na_filter=False, low_memory=False)
    labels = data.iloc[:,-1]
    data = data.drop([64], axis = 1)

    return np.array(data), np.array(labels)

def train_kmeans(k, data, labels):
    clutters = np.array([Clutter(data) for _ in range(k)])
    repeat = True
    counter = 0

    while repeat:
        print('iteration: {}'.format(counter))
        counter += 1
        # Assign points to closest clutter
        for i, x in enumerate(data):
            distances = []
            for clutter in clutters:
                dist = np.linalg.norm(x-clutter.center)
                distances.append(dist)
            closest_index = np.argmin(distances)
            clutters[closest_index].add_member(i)

        nb_no_changes = 0
        # Update center
        for c in clutters:
            c.update_center(data)
            c.reset_members()
            if c.center_same == True:
                nb_no_changes += 1
            # If all clutters stop moving, stop
            if nb_no_changes == k:
                repeat = False
                break

    # Save circles
    centers = np.empty((0,64))
    for c in clutters:
        centers = np.vstack([centers, c.center])
    np.save('centers', centers)

    evaluate_kmeans(clutters, data, labels, k)

def test_kmeans(k, data, labels):
    saved_centers = np.load('centers.npy')
    clutters = np.array([Clutter(data) for _ in range(k)])
    # k = 10
    dict = {'0':5, '1':3, '2':7, '3':2, '4':8, '5':0, '6':4, '7':1, '8':9, '9':6}
    # k = 30
    '''
    dict = {'0':4, '1':4, '2':5, '3':1, '4':2, '5':9, '6':2, '7':7, '8':0, '9':5,
            '10':7, '11':8, '12':6, '13':4, '14':9, '15':5, '16':7, '17':2, '18':2, '19':7,
            '20':6, '21':1, '22':3, '23':4, '24':1, '25':3, '26':9, '27':8, '28':9, '29':1}
            '''
    incorrect = 0

    for c, center in zip(clutters, saved_centers):
        c.center = center

    # Assign each test example to the closest clutter
    for i, x in enumerate(data):
        distances = []
        for clutter in clutters:
            dist = np.linalg.norm(x - clutter.center)
            distances.append(dist)
        closest_index = np.argmin(distances)
        clutters[closest_index].add_member(i)

    # For each clutter, look at all assigned members and compare
    # them to its assigned class
    for i, c in enumerate(clutters):
        predicts = labels[c.members]
        for p in predicts:
            if dict[str(i)] != p:
                incorrect +=1

    accuracy = ((1797-incorrect) / 1797) * 100
    print('accuracy {}'.format(accuracy))

    predictions = np.empty((0,0))
    actual = np.empty((0, 0))
    for i, c in enumerate(clutters):
        nb_predictions = len(c.members)
        arr = np.full((1, nb_predictions), dict[str(i)])
        predictions = np.append(predictions, arr)
        actual_arr = labels[c.members]
        actual = np.append(actual, actual_arr)

    print(predictions.shape)
    print(actual.shape)

    cm = confusion_matrix(actual, predictions)

    plt.figure(figsize=(10, 7))
    sn.heatmap(cm, annot=True, fmt='.1f')
    plt.xlabel('Predicted')
    plt.ylabel('True label')
    plt.savefig('AH_CHAOS.png', bbox_inches='tight')

def create_grayscale():
    saved_centers = np.load('centers.npy')
    for n, c in enumerate(saved_centers):
        c = c.reshape(8,8)
        img = Image.new('L', (8,8))
        for i in range(img.size[0]):
            for j in range(img.size[0]):
                img.putpixel((j,i), int(c[i][j] * 16))
        img = img.resize((50,50))
        name = 'clutter' + str(n) + '.png'
        img.save(name)

if __name__ == "__main__":
    x_tr, labels_tr= load_data(path_tr)
    #train_kmeans(nb_clutters,x_tr, labels_tr)
    x_ts, labels_ts = load_data(path_ts)
    test_kmeans(nb_clutters, x_ts, labels_ts)
    create_grayscale()




