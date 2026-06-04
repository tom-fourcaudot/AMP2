import pandas as pd
from collections import Counter

from numpy.ma.core import mean

from utils.data_split import split_data


def squared_euclidian_distance(a: pd.Series, b: pd.Series):
    return ((a['sepal length'] - b['sepal length'])**2
            + (a['sepal width'] - b['sepal width'])**2
            + (a['petal length'] - b['petal length'])**2
            + (a['petal width'] - b['petal width'])**2)

def knn(df_train: pd.DataFrame, data_test: pd.Series, k: int = 3):
    distances = []
    for index, row in df_train.iterrows():
        distance = squared_euclidian_distance(row, data_test)
        distances.append((distance, row['class']))
    distances.sort(key=lambda x: x[0])
    k_neighbors = distances[:k]
    neighbors_predictions = Counter(tup[1] for tup in k_neighbors)
    prediction = neighbors_predictions.most_common(1)
    return prediction[0][0], prediction[0][1]/k

def x_times_knn(data: pd.DataFrame, k: int, n: int = 50):
    accuracies = []
    for _ in range(n):
        training, testing = split_data(data)

        result = []
        for index, row in testing.iterrows():
            prediction = knn(training, row, k)
            result.append((prediction, row['class'], prediction[0] == row['class']))

        corrects = sum(1 if d[2] else 0 for d in result)
        accuracy = corrects/len(training)
        accuracies.append(accuracy)
    return mean(accuracies)