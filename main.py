from statistics import mean

import pandas as pd
from ucimlrepo import fetch_ucirepo

from fisher import fisher
from knn import x_times_knn

# fetch dataset
iris = fetch_ucirepo(id=53)

# data (as pandas dataframes)
x:pd.DataFrame = iris.data.features
y:pd.DataFrame = iris.data.targets
data:pd.DataFrame = x.join(y, how='inner')


# test knn
def test_knn(iterations: int = 20):
    try_k = [1, 2, 3, 5, 7, 10, 15]
    accuracies = {}
    for k in try_k:
        accuracies[k] = x_times_knn(data, k=k, n=iterations)

    print(accuracies)

def test_fisher(iterations: int = 20):
    accuracies = []
    for _ in range(iterations):
        result = fisher(data)
        good_predictions = sum(1 for ground_true, prediction, _ in result if ground_true == prediction)
        accuracy = good_predictions / len(result)
        accuracies.append(accuracy)
    fisher_accuracy = mean(accuracies)
    print(fisher_accuracy)

if __name__ == '__main__':
    test_knn(iterations=20)
    test_fisher(iterations=20)