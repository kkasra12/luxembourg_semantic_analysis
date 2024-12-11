import glob
import os

import pandas as pd


def check_downloaded_data(social_media: str):
    """
    Normally, we will have a folder as same as the given `social_media` name.
    in this folder, we will have several csv files which are the downloaded data.
    This function will check if the data is downloaded or not.
    if the data is downloaded, it will combine all the csv files into one csv file and return the dataframe.

    :param social_media: str
    :return: pd.DataFrame or None if the data is not downloaded
    """
    print(social_media)
    if social_media not in os.listdir(os.path.dirname(__file__)):
        return
    print(os.path.dirname(__file__))

    all_dfs = []
    for file in glob.glob(
        os.path.join(os.path.dirname(__file__), social_media, "*.csv")
    ):
        all_dfs.append(pd.read_csv(file))

    df = pd.concat(all_dfs, ignore_index=True)
    return df
