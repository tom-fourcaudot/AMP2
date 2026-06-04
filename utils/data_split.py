import pandas as pd

def split_data(df: pd.DataFrame, ratio: float = 0.5):
    training = df.sample(frac=ratio)
    test = df.drop(training.index)
    return training, test

def by_class_separation(df: pd.DataFrame):
    df1, df2, df3 = [x for _, x in df.groupby('class')]
    return df1, df2, df3