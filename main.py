from ucimlrepo import fetch_ucirepo

from knn import x_times_knn

# fetch dataset
iris = fetch_ucirepo(id=53)

# data (as pandas dataframes)
x = iris.data.features
y = iris.data.targets
data = x.join(y, how='inner')

# test knn
def test_knn():
    try_k = [1, 2, 3, 5, 7, 10, 15]
    iterations = 20

    accuracies = {}
    for k in try_k:
        accuracies[k] = x_times_knn(data, k=k, n=iterations)

    print(accuracies)