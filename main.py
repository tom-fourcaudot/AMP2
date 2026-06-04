import pandas as pd
from numpy.ma.core import mean
from ucimlrepo import fetch_ucirepo

from utils.data_split import split_data
from knn import knn

# fetch dataset
iris = fetch_ucirepo(id=53)

# data (as pandas dataframes)
x = iris.data.features
y = iris.data.targets
data = x.join(y, how='inner')

try_K = [1, 2, 3, 5, 7, 10, 15]
iterations = 50

accuracies = {}


for k in try_K:
    tmp = []
    for _ in range(iterations):
        training, testing = split_data(data)

        result = []
        for index, row in testing.iterrows():
            prediction = knn(training, row, k)
            result.append((prediction, row['class'], prediction[0] == row['class']))

        corrects = sum(1 if d[2] else 0 for d in result)
        accuracy = corrects/len(training)
        tmp.append(accuracy)
    print(f'Average accuracy for K = {k} -> {mean(tmp)}')
    accuracies[k] = mean(tmp)

print(accuracies)