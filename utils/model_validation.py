import numpy as np

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
    for station in stations:
        precisions = {}
        for sensitive_class in sensitive_classes:
            mask = (X_test[first_sensitive_column] == sensitive_class) & (
                X_test["station"] == station
            )
            if np.sum(mask) > min_samples:
                precisions[sensitive_class] = precision_score(
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
    for sensitive_class in sensitive_classes:
        mask = X_test[first_sensitive_column] == sensitive_class
        if np.sum(mask) > min_samples:
            global_precisions[sensitive_class] = precision_score(
                y_true[mask], y_pred[mask], pos_label=1
            )

    if len(precisions) > 1:
        diff = np.max(list(precisions.values())) - np.min(list(precisions.values()))
        if diff > max_diff:
            is_satisfied = False

    return is_satisfied, problematic_stations, good_stations, global_precisions
