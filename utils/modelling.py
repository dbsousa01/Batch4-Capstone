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

# File variables
models_path = "./models/"
images_path = "./images/"


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


def calculate_prediction(pipeline: Pipeline, X_test: pd.DataFrame, decision_value: float = 0.1) -> np.asarray:
    """
        Function to calculate predictions based on probabilities. If proba of being true class es greater than
    decision_value, then it's the true class, else it's the false class.

    :param pipeline: pipeline
    :param X_test: DataFrame to be predicted
    :param decision_value: integer to be compared with

    :return: np.asarray with predictions
    """
    # get all the probas of being successful
    probas = pipeline.predict_proba(X_test)[:, 1]

    # If proba is greater then decision_value then it's true, else false
    predictions = np.where(probas > decision_value, 1, 0)

    return predictions


def feature_importance(feature_names: List[str], clf: Pipeline) -> None:
    """
        Function to create a plot that orders features and their importances for the trained model.
    Currently it's only implemented for LinearSVC since it is the best performing model

    :param feature_names: Name of features that are being used
    :param clf: The classifier object
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
        images_path + "{0}_feature_importance_{1}.png".format(classifier_name, today_date)
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
    with open(models_path + 'columns.json', 'w') as fh:
        json.dump(X_train.columns.tolist(), fh)

    with open(models_path + 'dtypes.pickle', 'wb') as fh:
        pickle.dump(X_train.dtypes, fh)

    joblib.dump(pipeline, models_path + 'pipeline.pickle')
    return


def load_model() -> [Pipeline, List[str], pd.Series]:
    """
        Loads the model into memory and returns the pipeline, columns array and their dtypes

    :return:
    """
    with open(models_path + 'columns.json') as fh:
        columns = json.load(fh)

    with open(models_path + 'dtypes.pickle', 'rb') as fh:
        dtypes = pickle.load(fh)

    pipeline = joblib.load(models_path + 'pipeline.pickle')

    return pipeline, columns, dtypes
