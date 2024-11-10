import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset


def load_rating_wo_text(path):
    """
    Load a ratings dataset without the text column
    :param path: the path to the ratings.csv
    :return: the loaded dataframe without the text column
    """
    return pd.read_csv(path, usecols=lambda col: col != "text")


def load_user_data(
    ba_path="../data/BeerAdvocate/ratings/BA_ratings.csv",
    rb_path="../data/RateBeer/ratings/ratings.csv",
):
    """
    Loads the users.csv for both datasets
    :param ba_path: Path to the BeerAdvocate users.csv
    :param rb_path: Path to the RateBeer users.csv
    :return:
    """
    df_ba_users = pd.read_csv(ba_path)
    df_rb_users = pd.read_csv(rb_path)
    return df_ba_users, df_rb_users


def load_brewery_data(brewery_path="./data/RateBeer/breweries.csv"):
    """
    Loading the brewery dataset.
    CAUTION: The location attribute is renamed to brewery_location.
    :param brewery_path: Path to the breweries.csv
    :return: the brewery dataset in a pandas df
    """
    df_brew = pd.read_csv(brewery_path)
    df_brew.rename(
        columns={"id": "brewery_id", "location": "brewery_location"}, inplace=True
    )
    return df_brew


def load_rating_data():
    """
    Loads rating data in pandas dataframes
    :return: these dataframes
    """
    df_rb_ratings = pd.read_csv(
        "C:/Users/nette/Dateistruktur/RWTH/Auslandssemester/ImSemester/CS-401 Applied Data "
        "Analysis/Project/P2/Git/ada-2024-project-thefivedatawranglers/src/data/RateBeer/ratings/ratings.csv"
    )
    df_ba_ratings = pd.read_csv(
        "C:/Users/nette/Dateistruktur/RWTH/Auslandssemester/ImSemester/CS-401 Applied Data "
        "Analysis/Project/P2/Git/ada-2024-project-thefivedatawranglers/src/data/BeerAdvocate/ratings/BA_ratings.csv"
    )
    return df_rb_ratings, df_ba_ratings


class SomeDataset(Dataset):
    """
    A dataset implements 2 functions
        - __len__  (returns the number of samples in our dataset)
        - __getitem__ (returns a sample from the dataset at the given index idx)
    """

    def __init__(self, dataset_parameters, **kwargs):
        super().__init__()
        ...


class SomeDatamodule(DataLoader):
    """
    Allows you to sample train/val/test data, to later do training with models.
    """

    def __init__(self):
        super().__init__()
