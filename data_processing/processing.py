import os
import numpy as np
import pandas as pd

# File variables
dir_path = os.path.dirname(os.path.realpath(__file__))


def load_data() -> pd.DataFrame:
    """
        Loads the data and returns it in a dataframe
    :return:
    """
    file_path = "{}/../data/train.csv".format(dir_path)
    df = pd.read_csv(file_path, index_col="observation_id")

    # fill na values - missing values are imputed, not better values for now
    df.fillna({
        "Type": "missing",
        "Part of a policing operation": False,
        "Legislation": "missing",
    }, inplace=True)

    return df


def build_outcome_label(df: pd.DataFrame) -> pd.DataFrame:
    """
        Created a new column in the dataframe that is the outcome of the search -> successful or not

    :param df: DataFrame to be processed
    :return: df with the new label column
    """

    df_label = df.copy()
    sucessful_outcomes = ["Local resolution", "Community resolution", "Offender given drugs possession warning",
                          "Khat or Cannabis warning", "Caution (simple or conditional)",
                          "Offender given penalty notice", "Arrest", "Penalty Notice for Disorder",
                          "Suspected psychoactive substances seized - No further action", "Summons / charged by post",
                          "Article found - Detailed outcome unavailable", "Offender cautioned", "Suspect arrested",
                          "Suspect summonsed to court"]

    df_label["label"] = df_label["Outcome"]
    df_label["label"] = np.where(df_label["Outcome"].isin(sucessful_outcomes), 1, 0)
    return df_label


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
        Create time features for the dataset

    :param df: DataFrame to be processed
    :return: df with the new time features as columns
    """

    df_time = df.copy()

    df_time['Date'] = pd.to_datetime(df_time['Date'], format='%Y/%m/%d')

    # get the hour and day of the week, maybe they will be useful
    df_time['hour'] = df_time['Date'].dt.hour
    df_time['month'] = df_time['Date'].dt.month
    df_time["year"] = df_time["Date"].dt.year
    df_time['day_of_week'] = df_time['Date'].dt.day_name()
    df_time["year-quarter"] = df_time["Date"].dt.year.astype(str) + 'Q' + df_time["Date"].dt.quarter.astype(str)

    return df_time
