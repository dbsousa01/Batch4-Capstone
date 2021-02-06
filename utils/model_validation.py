import numpy as np
import pandas as pd

from sklearn.metrics import precision_score


def verify_no_discrimination(
    X_test,
    y_true,
    y_pred,
    sensitive_columns_tuple=("Officer-defined ethnicity", "Gender"),
    max_diff=0.05,
    min_samples=30,
):
    """
            Verifies that no station has discrimination in between protected column tuples (ethnicity, gender)

    :param X_test: DataFrame with the testing data
    :param y_true: np.array with the true labels
    :param y_pred: np.array with the predicted labels
    :param sensitive_columns_tuple: tuple with the sensitive_columns
    :param max_diff: max difference between columns, if greater then there is discrimination
    :param min_samples: minimum nb of samples to be considered
    :return:
    """

    first_sensitive_column, second_sensitive_column = sensitive_columns_tuple

    stations = X_test["station"].unique()
    sensitive_classes = X_test[first_sensitive_column].unique()
    second_sensitive_classes = X_test[second_sensitive_column].unique()

    is_satisfied = True
    problematic_stations = []
    good_stations = []
    ignored_stations = []
    # For every station
    for station in stations:
        precisions = {}
        # For every classes
        for sensitive_class in sensitive_classes:
            for second_sensitive_class in second_sensitive_classes:

                # Create a mask that filters according to class values
                mask = (
                    (X_test[first_sensitive_column] == sensitive_class)
                    & (X_test["station"] == station)
                    & (X_test[second_sensitive_column] == second_sensitive_class)
                )

                # if the dataframe filtered with the mask has more than 30 rows
                if np.sum(mask) > min_samples:

                    # generate the dict key with the two classes
                    key = "{0} - {1}".format(sensitive_class, second_sensitive_class)
                    precisions[key] = precision_score(
                        y_true[mask], y_pred[mask], pos_label=1
                    )

        if len(precisions) > 1:
            diff = np.max(list(precisions.values())) - np.min(list(precisions.values()))

            if diff > max_diff:
                is_satisfied = False
                problematic_stations.append((station, diff, precisions))
            else:
                good_stations.append((station, diff, precisions))
        else:
            ignored_stations.append((station, None, []))

    global_precisions = {}
    for station in stations:
        for sensitive_class in sensitive_classes:
            for second_sensitive_class in second_sensitive_classes:
                mask = (
                    (X_test[first_sensitive_column] == sensitive_class)
                    & (X_test[second_sensitive_column] == second_sensitive_class)
                    & (X_test["station"] == station)
                )

                if np.sum(mask) > min_samples:
                    # key to filter the dictionary
                    key = "{0} - {1}".format(sensitive_class, second_sensitive_class)
                    global_precisions[key] = precision_score(
                        y_true[mask], y_pred[mask], pos_label=1
                    )

    if len(precisions) > 1:
        diff = np.max(list(precisions.values())) - np.min(list(precisions.values()))
        if diff > max_diff:
            is_satisfied = False

    return is_satisfied, problematic_stations, good_stations, global_precisions


def comparison_between_stations(X_test, y_true, y_pred, min_samples=30):
    """
        Function that returns the precision score for each station

    :param X_test: DataFrame with the test set that we want to use for comparison
    :param y_true: np.array with the true y labels
    :param y_pred: np.array with the predicted y labels
    :param min_samples: minimum samples for the entry to be considered
    :return: sorted dict with the stations and their precisions
    """
    stations = X_test["station"].unique()
    precisions = {}

    # For every station
    for station in stations:

        # Create a mask that filters according to station
        mask = (X_test["station"] == station)

        # if the dataframe filtered with the mask has more than 30 rows
        if np.sum(mask) > min_samples:
            # generate the dict key with the two classes
            precisions[station] = precision_score(
                y_true[mask], y_pred[mask], pos_label=1
            )
    return dict(sorted(precisions.items(), key=lambda item: item[1]))
