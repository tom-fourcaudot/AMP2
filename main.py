from statistics import mean

import pandas as pd
from ucimlrepo import fetch_ucirepo

from fisher import fisher, x_times_fisher
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

    return accuracies

def test_fisher(iterations: int = 20):
    accuracies = x_times_fisher(data, iterations)
    return accuracies

if __name__ == '__main__':
    knn_accuracies = test_knn(iterations=20)
    fisher_accuracies = test_fisher(iterations=20)