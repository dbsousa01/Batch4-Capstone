import os
import numpy as np
import pandas as pd

# File variables
dir_path = os.path.dirname(os.path.abspath(__file__))


def load_data() -> pd.DataFrame:
    """
        Loads the data and returns it in a dataframe
    :return:
    """
    file_path = "{}/../data/train.csv".format(dir_path)
    df = pd.read_csv(file_path, index_col="observation_id")

    # fill na values - missing values are imputed, not better values for now
    df.fillna(
        {
            "Type": "missing",
            "Part of a policing operation": False,
            "Legislation": "missing",
            "Outcome linked to object of search": False,
        },
        inplace=True,
    )
    # drop metropolitan station due to only NaNs
    df = df[df["station"] != "metropolitan"]
    df = df[df["Gender"] != "Other"]
    df = df[df["Age range"] != "under 10"]
    df = df[(df["Officer-defined ethnicity"] != "Mixed") & df["Officer-defined ethnicity"] != "Other"]

    return df


def build_outcome_label(df: pd.DataFrame) -> pd.DataFrame:
    """
        Created a new column in the dataframe that is the outcome of the search -> successful or not

    :param df: DataFrame to be processed
    :return: df with the new label column
    """

    df_label = df.copy()
    sucessful_outcomes = [
        "Local resolution",
        "Community resolution",
        "Offender given drugs possession warning",
        "Khat or Cannabis warning",
        "Caution (simple or conditional)",
        "Offender given penalty notice",
        "Arrest",
        "Penalty Notice for Disorder",
        "Suspected psychoactive substances seized - No further action",
        "Summons / charged by post",
        "Article found - Detailed outcome unavailable",
        "Offender cautioned",
        "Suspect arrested",
        "Suspect summonsed to court",
    ]

    df_label["label"] = df_label["Outcome"]
    df_label["label"] = np.where(
        (
            df_label["Outcome"].isin(sucessful_outcomes)
            & df_label["Outcome linked to object " "of search"]
        ),
        1,
        0,
    )
    return df_label


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
        Create time features for the dataset

    :param df: DataFrame to be processed
    :return: df with the new time features as columns
    """

    df_time = df.copy()

    df_time["Date"] = pd.to_datetime(df_time["Date"], format="%Y/%m/%d")

    # get the hour and day of the week, maybe they will be useful
    df_time["hour"] = df_time["Date"].dt.hour
    df_time["month"] = df_time["Date"].dt.month
    df_time["year"] = df_time["Date"].dt.year
    df_time["day_of_week"] = df_time["Date"].dt.day_name()
    df_time["year-quarter"] = (
        df_time["Date"].dt.year.astype(str)
        + "Q"
        + df_time["Date"].dt.quarter.astype(str)
    )

    return df_time


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df_copy = df.copy()

    df_copy["Part of a policing operation"] = np.where(df_copy["Part of a policing operation"], 1, 0)

    series = df_copy.groupby("Legislation").count()["Type"] > 3000
    legislations_series = series[series != False]
    legislations_to_keep = list(legislations_series.index)

    df_copy["Legislation"] = np.where(df["Legislation"].isin(legislations_to_keep), df["Legislation"], "Other")

    series = df_copy.groupby("Object of search").count()["Type"] > 3000
    obj_search_series = series[series != False]
    obj_search_list = list(obj_search_series.index)
    df_copy["Object of search"] = np.where(df["Object of search"].isin(obj_search_list),
                                           df["Object of search"],
                                           "Other")
    return df_copy
