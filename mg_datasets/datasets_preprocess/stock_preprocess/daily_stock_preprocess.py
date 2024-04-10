# -*- coding: utf-8 -*-
# @Time    : 2024/4/9 16:12
# @Author  : Karry Ren

""" Please follow the `README.md` to download the 1-day stock dataset,
      and get the directory like:
        /Users/username/.qlib/qlib_data/qlib_data/
            ├── cn_data
                ├── calendars
                ├── features
                └── instruments
Please make sure you have this directory and the dataset is right !!!

Because I want to make the code clear and beautiful, so I need you to do some directory creation by hand !!!
    You need to create the following directory structure `BY HAND`: `CSI300_STOCK_DATASET_PATH/`

During the preprocessing, we wil do operations by the following steps:
    1. Change the `PATH` based on your situation.
    2. Load the daily feature and label. (665 stocks data in 3081 days)
    3. Swip the index and sort the df.
    4. Save the dataframe to `.pkl` file (a df with `datetime-stock` index).

All in all, after downloading the file from the web, you need:
    1. Create the directory structure `BY HAND` following the top comment.
    2. Run this file by `python daily_stock_preprocess.py` and you will ge the following directory structure:
        CSI300_STOCK_DATASET_PATH/1_day.pkl

"""

import pandas as pd
from typing import List, Tuple
import qlib
from qlib.data import D


def load_daily_stock_dataset(market: str = "csi300") -> Tuple[pd.DataFrame, List[str], List[str]]:
    """ Load the daily stock dataset.

    :param market: the market type, you have only 1 choice now:
        - `csi300` (665 stocks, 3081 trading dates from `2007-02-16` to `2020-01-01`)

    returns:
        - daily_stock_df: the daily stock data dataframe, index is (instrument-datetime)
        - feature_name_list: the feature name list
        - label_name_list: the label name list

    """

    # ---- Check the market ---- #
    assert market in ["csi300"], f"The market {market} is not supported now !!"

    # ---- Step 1. Load features ---- #
    # build up the feature field list and feature name list
    feature_field_list, feature_name_list = [], []
    feature_field_list += ["$open", "$close"]
    feature_name_list += ["open", "close"]
    # - feature 1. the `daily_high/daily_close` and `daily_high (shift k days)/daily_close` (20)
    feature_field_list += ["$high/$close"]  # ATTENTION: Ref($high, 0) != $high !!
    feature_field_list += [f"Ref($high, {day})/$close" for day in range(1, 20)]  # Ref means shift back if day > 0, else shift front
    feature_name_list += [f"HIGH{day}" for day in range(0, 20)]  # append the name
    # - feature 2. the `daily_open/daily_close` and `daily_open (shift k days)/daily_close` (20)
    feature_field_list += ["$open/$close"]  # ATTENTION: Ref($open, 0) != $open !!
    feature_field_list += [f"Ref($open, {day})/$close" for day in range(1, 20)]  # Ref means shift back if day > 0, else shift front
    feature_name_list += [f"OPEN{day}" for day in range(0, 20)]  # append the name
    # - feature 3. the `daily_low/daily_close` and `daily_low (shift k days)/daily_close` (20)
    feature_field_list += ["$low/$close"]  # ATTENTION: Ref($low, 0) != $low !!
    feature_field_list += [f"Ref($low, {day})/$close" for day in range(1, 20)]  # Ref means shift back if day > 0, else shift front
    feature_name_list += [f"LOW{day}" for day in range(0, 20)]  # append the name
    # - feature 4. the `daily_close/daily_close` and `daily_close (shift k days)/daily_close` (20)
    feature_field_list += ["$close/$close"]  # ATTENTION: Ref($low, 0) != $low !! and it must be 1.
    feature_field_list += [f"Ref($close, {day})/$close" for day in range(1, 20)]  # Ref means shift back if day > 0, else shift front
    feature_name_list += [f"CLOSE{day}" for day in range(0, 20)]  # append the name
    # - feature 5. the `daily_volume/daily_volume` and `daily_volume (shift k days)/daily_volume` (20)
    feature_field_list += ["$volume/$volume"]  # ATTENTION: Ref($volume, 0) != $volume !! and it must be 1.
    feature_field_list += [f"Ref($volume, {day})/$volume" for day in range(1, 20)]  # Ref means shift back if day > 0, else shift front
    feature_name_list += [f"VOLUME{day}" for day in range(0, 20)]  # append the name
    # use qlib to load features
    print("|| LOAD FEATURES NOW ||")
    daily_stock_df = D.features(D.instruments(market), fields=feature_field_list, start_time="2007-02-16", end_time="2020-01-01", freq="day")
    daily_stock_df.columns = feature_name_list  # change the name
    print("|| LOAD FEATURES OVER ||")

    # ---- Step 2. Load label ---- #
    # build up the label field list and label name list
    label_field_list = ["(Ref($close, -2) + Ref($open, -2))/(Ref($close, -1) + Ref($open, -1)) - 1"]
    label_name_list = ["LABEL"]
    print("|| LOAD LABEL NOW ||")
    df_labels = D.features(D.instruments(market), fields=label_field_list, start_time="2007-02-16", end_time="2020-01-01", freq="day")
    df_labels.columns = label_name_list  # change the name
    daily_stock_df[label_name_list] = df_labels  # append label column to df
    print("|| LOAD LABEL OVER ||")

    # ---- Step 3. Do the summary of df ---- #
    stock_code_set, trading_date_set = set(), set()  # define the empty set
    print("|| DO SUMMARY NOW ||")
    for idx in daily_stock_df.index:
        stock_code_set.add(idx[0])  # add stock idx
        trading_date_set.add(idx[1])  # add trading date
    print(f"|| The `{market}` dataset includes `{len(stock_code_set)}` stock_code and `{len(trading_date_set)}` trading_date. ||")
    print("|| DO SUMMARY OVER ||")

    # ---- Return ---- #
    return daily_stock_df, feature_name_list, label_name_list


if __name__ == '__main__':
    # ---- Step 1. Change the `PATH` based on your situation ---- #
    CSI300_STOCK_DATASET_PATH = "../../../../CMLF_Dataset/CSI300"
    print("************************** BEGIN DAILY CSI300 STOCK DATASET PREPROCESSING **************************")

    # ---- Step 2. Load the daily feature and label ---- #
    print("************************** BEGIN LOADING DAILY STOCK DATASET **************************")
    qlib.init(provider_uri={"day": "/Users/karry/.qlib/qlib_data/cn_data"})  # init the qlib
    daily_stock_df, _, _ = load_daily_stock_dataset(market="csi300")  # load the stock dataset
    print("************************** FINISH LOADING DAILY STOCK DATASET **************************")

    # ---- Step 3. Swip the index and sort the df ---- #
    print("************************** BEGIN SORTING DAILY STOCK DATASET **************************")
    # swip the index from (instrument-datetime) to (datetime-instrument)
    daily_stock_df.index = daily_stock_df.index.swaplevel()
    # sort the stock by datetime
    daily_stock_df = daily_stock_df.sort_index()
    print("************************** FINISH SORTING DAILY STOCK DATASET **************************")

    # ---- Step 4. Save the dataframe to `.pkl` file ---- #
    print("************************** BEGIN SAVING DAILY STOCK DATASET **************************")
    daily_stock_df.to_pickle(f"{CSI300_STOCK_DATASET_PATH}/1_day.pkl")
    print("************************** FINISH SAVING DAILY STOCK DATASET **************************")
