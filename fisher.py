from statistics import mean

import numpy as np
import pandas as pd

from utils.data_split import split_data_proportional, by_class_separation


def get_projection_vector(df1: pd.DataFrame, df2: pd.DataFrame) -> tuple[list[float], float]:
    """
    Calculate the projection vector between two dataframes
    :param df1: first dataframe
    :param df2: second dataframe
    :return: projection vector
    """
    df1 = df1.drop(columns=['class'])
    df2 = df2.drop(columns=['class'])

    mean1 = df1.mean(numeric_only=True)
    mean2 = df2.mean(numeric_only=True)

    centered1 = df1 - mean1
    centered2 = df2 - mean2

    scatter1 = centered1.T @ centered1
    scatter2 = centered2.T @ centered2

    sw = scatter1 + scatter2
    mean_difference  = mean1 - mean2

    w = np.linalg.inv(sw) @ mean_difference

    mean1_projected = mean1 @ w
    mean2_projected = mean2 @ w
    threshold = (mean1_projected + mean2_projected) / 2

    if mean1_projected < mean2_projected:
        w = -w
        threshold = (mean1 @ w + mean2 @ w) / 2
    return w, threshold

def fisher(data: pd.DataFrame):
    training, testing = split_data_proportional(data)

    df1, df2, df3 = by_class_separation(training)
    class1 = df1['class'].unique()[0]
    class2 = df2['class'].unique()[0]
    class3 = df3['class'].unique()[0]

    classifiers = {class1: get_projection_vector(df1, pd.concat([df2, df3])),
                   class2: get_projection_vector(df2, pd.concat([df3, df1])),
                   class3: get_projection_vector(df3, pd.concat([df1, df2]))}

    fisher_results = []
    for index, row in testing.iterrows():
        c = row['class']
        row_features = row.drop('class').to_numpy()

        predictions: dict = {}
        for model_class, classifier in classifiers.items():
            projection_vector, threshold = classifier
            projection = row_features @ projection_vector
            predictions[model_class] = (projection - threshold)
        p = max(predictions, key=lambda k: predictions[k])
        fisher_results.append((c, p, predictions[p]))
    return fisher_results

def x_times_fisher(data: pd.DataFrame, iterations: int = 20):
    accuracies = []
    for _ in range(iterations):
        result = fisher(data)
        good_predictions = sum(1 for ground_true, prediction, _ in result if ground_true == prediction)
        accuracy = good_predictions / len(result)
        accuracies.append(accuracy)
    return accuracies