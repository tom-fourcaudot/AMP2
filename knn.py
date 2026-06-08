import pandas as pd
from collections import Counter

from numpy.ma.core import mean

from utils.data_split import split_data_proportional


def squared_euclidian_distance(a: pd.Series, b: pd.Series) -> float:
    """
    Calculate the  squared Euclidean distance between two 4 dimensional vectors.
    :param a: the first vector
    :param b: the second vector
    :return: the squared Euclidean distance
    """
    return ((a['sepal length'] - b['sepal length']) ** 2
            + (a['sepal width'] - b['sepal width']) ** 2
            + (a['petal length'] - b['petal length']) ** 2
            + (a['petal width'] - b['petal width']) ** 2)


def knn(df_train: pd.DataFrame, data_test: pd.Series, k: int = 3) -> tuple[str, float]:
    """
    Create and test the k-nearest neighbors of 1 data
    :param df_train: the training dataframe
    :param data_test: the test data
    :param k: number of nearest neighbors
    :return: the prediction of the model with its confidence percentage
    """
    distances = []
    for index, row in df_train.iterrows():
        distance = squared_euclidian_distance(row, data_test)
        distances.append((distance, row['class']))
    distances.sort(key=lambda x: x[0])
    k_neighbors = distances[:k]
    neighbors_predictions = Counter(tup[1] for tup in k_neighbors)
    prediction = neighbors_predictions.most_common(1)
    return prediction[0][0], prediction[0][1] / k


def x_times_knn(data: pd.DataFrame, k: int, n: int = 50) -> float:
    """
    Run the knn on all test data many times to test the accuracy of the model
    :param data: the row dataset
    :param k: the number of nearest neighbors
    :param n: the number of times to run the knn
    :return: the accuracy of the model
    """
    accuracies = []
    for _ in range(n):
        training, testing = split_data_proportional(data)

        result = []
        for index, row in testing.iterrows():
            prediction = knn(training, row, k)
            result.append((prediction, row['class'], prediction[0] == row['class']))

        corrects = sum(1 if d[2] else 0 for d in result)
        accuracy = corrects / len(training)
        accuracies.append(accuracy)
    return mean(accuracies)
