import json
import pickle
import joblib
import numpy as np
import pandas as pd

from typing import List
from datetime import date
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
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


def feature_importance(feature_names: List[str], clf: Pipeline, save_path: str) -> None:
    """
        Function to create a plot that orders features and their importances for the trained model.
    Currently it's only implemented for LinearSVC since it is the best performing model

    :param feature_names: Name of features that are being used
    :param clf: The classifier object
    :param save_path: Path of the file for the image to be saved
    :return:
    """

    # clf name
    classifier_name = clf.named_steps["classifier"].__class__.__name__

    # today date
    today_date = date.today().strftime("%d%m%Y")

    # named_steps to get the classifier from the Pipeline object
    imp = clf.named_steps["classifier"].feature_importances_
    imp, names = zip(*sorted(zip(imp, feature_names)))

    plt.barh(range(len(names)), imp, align="center")
    plt.yticks(range(len(names)), names)
    plt.tight_layout()
    plt.savefig(
        save_path + "{0}_feature_importance_{1}.png".format(classifier_name, today_date)
    )
    plt.show()
    return


def save_model(pipeline: Pipeline, X_train: pd.DataFrame) -> None:
    """
        Saves the model, columns and dtypes of the training set

    :param pipeline: pipeline to be saved
    :param X_train: DataFrame that was used to train the pipeline
    :return:
    """

    # save the model, columns and dtypes
    with open('./models/columns.json', 'w') as fh:
        json.dump(X_train.columns.tolist(), fh)

    with open('./models/dtypes.pickle', 'wb') as fh:
        pickle.dump(X_train.dtypes, fh)

    joblib.dump(pipeline, './models/pipeline.pickle')
    return
