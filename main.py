import pandas as pd
from ucimlrepo import fetch_ucirepo

from fisher import get_projection_vector
from knn import x_times_knn
from utils.data_split import by_class_separation, split_data

import numpy as np

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


training, testing = split_data(data)

df1, df2, df3 = by_class_separation(training)
class1 = df1['class'].unique()[0]
class2 = df2['class'].unique()[0]
class3 = df3['class'].unique()[0]

classifiers = {class1: get_projection_vector(df1, pd.concat([df2, df3])),
               class2: get_projection_vector(df2, pd.concat([df3, df1])),
               class3: get_projection_vector(df3, pd.concat([df1, df2]))}

print(classifiers)

for index, row in testing.iterrows():
    c = row['class']
    data = row.drop('class').to_numpy()

    for model_class, classifier in classifiers.items():
        projection_vector, threshold, order = classifier
        projection = data @ projection_vector
        prediction = order[0] if projection < threshold else order[1]
