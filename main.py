import pandas as pd
from ucimlrepo import fetch_ucirepo

from utils.data_split import split_data

# fetch dataset
iris = fetch_ucirepo(id=53)

# data (as pandas dataframes)
x = iris.data.features
y = iris.data.targets
data = x.join(y, how='inner')

training, testing = split_data(data)

print(training)
print(testing)