import pandas as pd
from sklearn.model_selection import train_test_split


def create_train_test(df: pd.DataFrame) -> [pd.DataFrame, pd.DataFrame]:
    """
        Stratify the data and split it into train and test sets

    :param df: DataFrame to generate training and testing
    :return: two DataFrames, train and test
    """
    train, test = train_test_split(df, stratify=df[["label", "station", "Officer-defined ethnicity",
                                                    "Gender", "Age range"]],
                                   test_size=0.25, random_state=123)

    return train, test
