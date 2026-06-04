from typing import Any

import numpy as np
import pandas as pd
from numpy import complexfloating, dtype, floating, ndarray


def get_projection_vector(df1: pd.DataFrame, df2: pd.DataFrame) -> ndarray[float, float, float, float]:
    """
    Calculate the projection vector between two dataframes
    :param df1: first dataframe
    :param df2: second dataframe
    :return: projection vector
    """
    class1 = df1['class'].unique()[0]
    class2 = df2['class'].unique()[0]

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
    return w