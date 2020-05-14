from csv import reader
from math import sqrt
import random


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


# Calculate the Euclidean distance between two rows
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1) - 1):
        distance += (row1[i] - row2[i]) ** 2
    return sqrt(distance)


# Find the most similar neighbors
def get_neighbors(train, test, n_neighbors):
    distances = list()
    for train_row in train:
        dist = euclidean_distance(test, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(n_neighbors):
        neighbors.append(distances[i][0])
    return neighbors


# Make a prediction with neighbors
def predict_classification(train, test, n_neighbors):
    neighbors = get_neighbors(train, test, n_neighbors)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction


def main():
    filename = input('Path to file: ')
    dataset = load_csv(filename)
    # remove header
    dataset.pop(0)
    # convert into float
    for i in range(len(dataset[0])):
        str_column_to_float(dataset, i)
    testscale = int(round(0.2 * len(dataset)))
    shuffled = dataset[:]
    random.shuffle(shuffled)
    train_dataset = shuffled[testscale:]
    test_dataset = shuffled[:testscale]
    # define number of k
    n_neighbors = int(input('Number of k: '))
    n_misclassified = 0
    # predict the label in the test set
    for row in test_dataset:
        predicted_label = predict_classification(train_dataset, row, n_neighbors)
        if predicted_label != row[-1]:
            n_misclassified += 1

    print('Finished running KNN with 80% of data as training set and 20% of data as test set.')
    print('Error rate:', "{:.2%}".format((n_misclassified / len(test_dataset))))


if __name__ == '__main__':
    main()
