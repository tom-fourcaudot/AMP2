import pandas as pd

def split_data(df: pd.DataFrame, ratio: float = 0.5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test
    :param df: the row dataframe
    :param ratio: the ratio of train and test between 0 and 1
    :return: tuple of train and test dataframes
    """
    training = df.sample(frac=ratio)
    test = df.drop(training.index)
    if  test is None:
        raise  Exception("This split result no test data")
    if  training is None:
        raise  Exception("This split result no training data")
    return training, test

def by_class_separation(df: pd.DataFrame)  -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into the 3 different class labels
    :param df: the row dataframe
    :return: tuple of the 3 different class labels
    """
    df1, df2, df3 = [x for _, x in df.groupby('class')]
    return df1, df2, df3