import os
import pandas as pd

# File variables
dir_path = os.path.dirname(os.path.realpath(__file__))


def load_data():
    """
        Loads the data and returns it in a dataframe
    :return:
    """
    file_path = "{}/../data/train.csv".format(dir_path)
    df = pd.read_csv(file_path, index_col="observation_id")

    return df
