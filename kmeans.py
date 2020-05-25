import argparse
import numpy as np
from csv import reader
from scipy.spatial.distance import cityblock


class K_Means:
    def __init__(self, k=2, tol=0, max_iter=10000):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, data):

        self.centroids = {}

        for i in range(self.k):
            random_index = np.random.randint(0, data.shape[0])
            self.centroids[i] = data[random_index]

        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            for featureset in X:
                if distance_metric == 'Manhattan':
                    distances = [cityblock(featureset, self.centroids[centroid]) for centroid in self.centroids]
                else:
                    distances = [np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid - original_centroid) / original_centroid * 100.0) > self.tol:
                    optimized = False

            if optimized:
                break

    def predict(self, data):
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification


# Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


def remove_label(dataset):
    for row in dataset:
        row.pop()
    return dataset


def get_label(dataset):
    y = []
    for row in dataset:
        y.append(row[-1])
    return y


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run k-means clustering. Please provide absolute path to dataset '
                                                 'and distance metric(optional).')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--distance', required=False,
                        help='not specifying the --distance option will default to Euclidean distance')
    args = parser.parse_args()

    filepath = args.dataset
    distance_metric = args.distance
    dataset = load_csv(filepath)
    # remove header
    dataset.pop(0)
    # convert into float
    for i in range(len(dataset[0])):
        str_column_to_float(dataset, i)
    y = np.array(get_label(dataset))
    X = np.array(remove_label(dataset))

    clf = K_Means()
    clf.fit(X)

    correct = 0
    for i in range(len(X)):
        predict_me = np.array(X[i].astype(float))
        predict_me = predict_me.reshape(-1, len(predict_me))
        prediction = clf.predict(predict_me)
        # print("Compare: ", prediction, y[i])
        if prediction == y[i]:
            correct += 1

    print("correction rate: ", correct / len(X))
