import pandas as pd

def split_data_random(df: pd.DataFrame, ratio: float = 0.5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test. This slitting split randomly without taking care of classes.
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

def split_data_proportional(df: pd.DataFrame, ratio: float = 0.5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train and test. This slitting split ensure you have the same amount of samples of each class
    :param df: the row dataframe
    :param ratio: the ratio of train and test between 0 and 1
    :return: tuple of train and test dataframes
    """
    df_by_class = by_class_separation(df)
    trainings = []
    testings = []
    for class_df in df_by_class:
        df_train, df_test = split_data_random(class_df, ratio)
        trainings.append(df_train)
        testings.append(df_test)
    training = pd.concat(trainings)
    testing = pd.concat(testings)
    return training, testing
