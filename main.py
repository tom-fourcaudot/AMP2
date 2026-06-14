import logging

import pandas as pd
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo

from fisher import x_times_fisher
from knn import x_times_knn
from utils.analisys import get_stats

# fetch dataset
iris = fetch_ucirepo(id=53)

# data (as pandas dataframes)
x:pd.DataFrame = iris.data.features
y:pd.DataFrame = iris.data.targets
data:pd.DataFrame = x.join(y, how='inner')


# test knn
def test_knn(iterations: int = 20):
    try_k = list(range(1, 16))
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
    classifiers_stats: dict[str, tuple[float, float]] = {}
    for k, v in knn_accuracies.items():
        classifiers_stats[f"knn {k}"] = get_stats(v)
    classifiers_stats["fisher"] = get_stats(fisher_accuracies)
    x: list[str] = list(classifiers_stats.keys())
    y, y_deviation = zip(*classifiers_stats.values())
    plt.errorbar(x, y, yerr=y_deviation, fmt='o')
    plt.title("Mean and Standard Deviation of each classifier")
    plt.xlabel("Classifier")
    plt.ylabel("Accuracy")
    plt.savefig("figures/mean_and_stdev.png")
    logging.info(f"Graph figures/mean_and_stdev.png have been generated")
    accuracies_dataframe = pd.DataFrame({
        'classifier': x,
        'accuracy': y,
        'std_deviation': y_deviation
    })
    accuracies_dataframe.to_csv("figures/mean_and_stdev.csv")
    logging.info(f"CSV figures/mean_and_stdev.csv have been generated")
