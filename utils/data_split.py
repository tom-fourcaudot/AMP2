import pandas as pd

def split_data(df: pd.DataFrame, ratio: float = 0.5):
    training = df.sample(frac=ratio)
    test = df.drop(training.index)
    return training, test