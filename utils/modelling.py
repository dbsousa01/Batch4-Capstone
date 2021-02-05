import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def create_train_test(
    df: pd.DataFrame,
) -> [pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
        Stratify the data and split it into train and test sets

    :param df: DataFrame to generate training and testing
    :return: two DataFrames, train and test
    """
    train, test = train_test_split(
        df,
        stratify=df[["label", "station"]],
        test_size=0.20,
        random_state=123,
    )

    # Remove label columns and columns that we are not going to receive in requests and can induce to some leaking
    X_train = train.drop(
        columns=[
            "label",
            "Outcome",
            "Outcome linked to object of search",
            "Removal of more than just outer clothing",
        ]
    )
    y_train = train[["label"]]

    # Test set
    X_test = test.drop(
        columns=[
            "label",
            "Outcome",
            "Outcome linked to object of search",
            "Removal of more than just outer clothing",
        ]
    )
    y_test = test[["label"]]

    return X_train, X_test, y_train.to_numpy(), y_test.to_numpy()
