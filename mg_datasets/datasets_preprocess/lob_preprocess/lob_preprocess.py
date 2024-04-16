# -*- coding: utf-8 -*-
# @Time    : 2024/4/15 16:07
# @Author  : Karry Ren

""" The preprocess code of raw Future LOB dataset (download from Qlib following `README.md`).

During the preprocessing, we wil do operations by the following steps:
    1. Change the `PATH` based on your situation.
    2. For-loop all `.csv` files of 0.5 seconds lob.
        Computing other granularity data: 1 second, 10 seconds, 30 seconds, 1 minute.

All in all, after downloading the file from the web, you need:
    1. Change and Create the `PATH` based on your situation.
    2. Run this file by `python lob_preprocess.py` and you will ge the following directory structure:
        FUTURE_LOB_DATASET_PATH/
            ├── 0.5_seconds
                ├── 05_seconds_20220104.csv
                ├── 05_seconds_20220105.csv
                ├── ...
                └── 05_seconds_20221230.csv
            ├── 1_second
                ├── 1_second_20220104.csv
                ├── 1_second_20220105.csv
                ├── ...
                └── 1_second_20221230.csv
            ├── 10_seconds
                ├── 10_seconds_20220104.csv
                ├── 10_seconds_20220105.csv
                ├── ...
                └── 10_seconds_20221230.csv
            ├── 30_seconds
                ├── 30_seconds_20220104.csv
                ├── 30_seconds_20220105.csv
                ├── ...
                └── 30_seconds_20221230.csv
            └── 1_minute
                ├── 1_minute_20220104.csv
                ├── 1_minute_20220105.csv
                ├── ...
                └── 1_minute_20221230.csv

"""

import pandas as pd
import numpy as np
import os
from price_alignment_features.cal_paf import cal_bid_paf_mat, cal_ask_paf_mat

# ---- Define the Params ---- #
# define the number of trading dates and tick
TRADING_DATES_NUM = 242  # totally 242 days
TICK_NUM = 28800  # the num of total ticks in one day
# define the gap of each granularity
SECOND_1_TICK_GAP, SECOND_10_TICK_GAP, SECOND_30_TICK_GAP, MINUTE_1_TICK_GAP = 2, 20, 60, 120
# define the LOB COLUMN
LOB_COLUMNS = [
    "BidPrice1", "BidVolume1", "AskPrice1", "AskVolume1",
    "BidPrice2", "BidVolume2", "AskPrice2", "AskVolume2",
    "BidPrice3", "BidVolume3", "AskPrice3", "AskVolume3",
    "BidPrice4", "BidVolume4", "AskPrice4", "AskVolume4",
    "BidPrice5", "BidVolume5", "AskPrice5", "AskVolume5"
]

# ---- Step 1. Change and Create the `PATH` based on your situation ---- #
# the path of download raw data
LOB_DOWNLOAD_FILE_PATH = "../../../../Data/Future_LOB_dataset/IF_M0/download_data"
assert os.path.exists(LOB_DOWNLOAD_FILE_PATH), f"Please DOWNLOAD the Future LOB dataset and move it to {LOB_DOWNLOAD_FILE_PATH} !"
# the path of 0.5 seconds files
LOB_05_SECOND_FILE_PATH = "../../../../Data/Future_LOB_dataset/IF_M0/0.5_seconds"
assert os.path.exists(LOB_05_SECOND_FILE_PATH), f"Please CREATE the path of 0.5 seconds files {LOB_05_SECOND_FILE_PATH} !"
# the path of 1 second files
LOB_1_SECOND_FILE_PATH = "../../../../Data/Future_LOB_dataset/IF_M0/1_second"
assert os.path.exists(LOB_1_SECOND_FILE_PATH), f"Please CREATE the path of 1 second files {LOB_1_SECOND_FILE_PATH} !"
# the path of 10 seconds files
LOB_10_SECOND_FILE_PATH = "../../../../Data/Future_LOB_dataset/IF_M0/10_seconds"
assert os.path.exists(LOB_10_SECOND_FILE_PATH), f"Please CREATE the path of 10 seconds files {LOB_10_SECOND_FILE_PATH} !"
# the path of 30 seconds files
LOB_30_SECOND_FILE_PATH = "../../../../Data/Future_LOB_dataset/IF_M0/30_seconds"
assert os.path.exists(LOB_30_SECOND_FILE_PATH), f"Please CREATE the path of 30 seconds files {LOB_30_SECOND_FILE_PATH} !"
# the path of 1 minute files
LOB_1_MINUTE_FILE_PATH = "../../../../Data/Future_LOB_dataset/IF_M0/1_minute"
assert os.path.exists(LOB_1_MINUTE_FILE_PATH), f"Please CREATE the path of 1 minute files {LOB_1_MINUTE_FILE_PATH} !"
# the path of 1 minute labels
LOB_1_MINUTE_LABEL_PATH = "../../../../Data/Future_LOB_dataset/IF_M0/1_minute_label"
assert os.path.exists(LOB_1_MINUTE_LABEL_PATH), f"Please CREATE the path of 1 minute label {LOB_1_MINUTE_LABEL_PATH} !"

# ---- Step 2. For-Loop all `.csv` files of 0.5 seconds lob ---- #
lob_file_list = sorted(os.listdir(LOB_DOWNLOAD_FILE_PATH))  # get the list of all raw lob files
assert len(lob_file_list) == TRADING_DATES_NUM, f"{TRADING_DATES_NUM} ERROR ! There are{len(lob_file_list)} trading days in lob dataset !"
for file_idx, lob_file in enumerate(lob_file_list):  # Computing other granularity data.
    print(f"***** `{file_idx + 1} | {TRADING_DATES_NUM}` START PROCESSING `{lob_file}` NOW !!! *****")

    # - 2.0 Read the raw 0.5 seconds lob data
    lob_data_df = pd.read_csv(f"{LOB_DOWNLOAD_FILE_PATH}/{lob_file}")
    # load the price and volume
    bid_price = lob_data_df[["BidPrice1", "BidPrice2", "BidPrice3", "BidPrice4", "BidPrice5"]].values  # shape=(28800, 5)
    ask_price = lob_data_df[["AskPrice1", "AskPrice2", "AskPrice3", "AskPrice4", "AskPrice5"]].values  # shape=(28800, 5)
    bid_volume = lob_data_df[["BidVolume1", "BidVolume2", "BidVolume3", "BidVolume4", "BidVolume5"]].values  # shape=(28800, 5)
    ask_volume = lob_data_df[["AskVolume1", "AskVolume2", "AskVolume3", "AskVolume4", "AskVolume5"]].values  # shape=(28800, 5)
    # transfer price to int, be suitable for the paf interfaces
    bid_price = (np.round(bid_price * 100, 0)).astype(np.int32)
    ask_price = (np.round(ask_price * 100, 0)).astype(np.int32)
    # add one dim for volume
    bid_volume = bid_volume.reshape(TICK_NUM, 5, 1)
    ask_volume = ask_volume.reshape(TICK_NUM, 5, 1)

    # - 2.1 get the 0.5 seconds lob data, keep raw
    # define the empty 1-second lob data, shape=(28800, 5, 2, 2), (T, L, D, F)
    lob_data_05_second = np.zeros((TICK_NUM, 5, 2, 2))  # reshape to (28800, 20)
    lob_data_05_second[:, :, 0, 0] = bid_price  # bid price
    lob_data_05_second[:, :, 0, 1] = bid_volume[:, :, 0]  # bid volume
    lob_data_05_second[:, :, 1, 0] = ask_price  # ask price
    lob_data_05_second[:, :, 1, 1] = ask_volume[:, :, 0]  # ask volume
    lob_data_05_second = lob_data_05_second.reshape(TICK_NUM, -1)  # reshape to (28800, 20)
    lob_data_05_second_df = pd.DataFrame(lob_data_05_second, columns=LOB_COLUMNS)  # construct the df
    lob_data_05_second_df.to_csv(f"{LOB_05_SECOND_FILE_PATH}/05_seconds_{lob_file}", index=False)  # save the df
    print(f"   - 1. FINISH 0.5 SECONDS !!!")

    # - 2.2 compute the 1 seconds lob data, align by price during 2 ticks and sum volume
    # define the empty 1-second lob data, shape=(28800/2, 5, 2, 2), (T, L, D, F)
    lob_data_1_second = np.zeros((TICK_NUM // SECOND_1_TICK_GAP, 5, 2, 2))
    # for loop the tick by `SECOND_1_TICK_GAP` to gen all PA Feature of each tick
    tick_i = 0
    for tick_i in range(SECOND_1_TICK_GAP - 1, TICK_NUM, SECOND_1_TICK_GAP):
        # compute the price range and paf-volume
        bid_price_range, bid_paf_volume = cal_bid_paf_mat(
            bid_price=bid_price[tick_i + 1 - SECOND_1_TICK_GAP:tick_i + 1],
            bid_feature=bid_volume[tick_i + 1 - SECOND_1_TICK_GAP:tick_i + 1],
        )  # bid price range and bid paf-volume
        ask_price_range, ask_paf_volume = cal_ask_paf_mat(
            ask_price=ask_price[tick_i + 1 - SECOND_1_TICK_GAP:tick_i + 1],
            ask_feature=ask_volume[tick_i + 1 - SECOND_1_TICK_GAP:tick_i + 1],
        )  # bid price range and bid paf-volume
        # set data
        lob_data_1_second[tick_i // SECOND_1_TICK_GAP, :, 0, 0] = bid_price_range[:5]  # bid price
        lob_data_1_second[tick_i // SECOND_1_TICK_GAP, :, 0, 1] = bid_paf_volume.sum(axis=(0, 2))[:5]  # bid volume
        lob_data_1_second[tick_i // SECOND_1_TICK_GAP, :, 1, 0] = ask_price_range[:5]  # bid price
        lob_data_1_second[tick_i // SECOND_1_TICK_GAP, :, 1, 1] = ask_paf_volume.sum(axis=(0, 2))[:5]  # bid volume
    assert tick_i == TICK_NUM - 1, "tick_i ERROR !"  # test the tick_i
    lob_data_1_second = lob_data_1_second.reshape(TICK_NUM // SECOND_1_TICK_GAP, -1)  # reshape to (28800/2, 20)
    lob_data_1_second_df = pd.DataFrame(lob_data_1_second, columns=LOB_COLUMNS)  # construct the df
    lob_data_1_second_df.to_csv(f"{LOB_1_SECOND_FILE_PATH}/1_second_{lob_file}", index=False)
    print(f"   - 2. FINISH 1 SECOND !!!")

    # - 2.3 compute the 10 seconds lob data, align by price during 20 ticks and sum volume
    # define the empty 10-second lob data, shape=(28800/20, 5, 2, 2), (T, L, D, F)
    lob_data_10_second = np.zeros((TICK_NUM // SECOND_10_TICK_GAP, 5, 2, 2))
    # for loop the tick by `SECOND_10_TICK_GAP` to gen all PA Feature of each tick
    tick_i = 0
    for tick_i in range(SECOND_10_TICK_GAP - 1, TICK_NUM, SECOND_10_TICK_GAP):
        # compute the price range and paf-volume
        bid_price_range, bid_paf_volume = cal_bid_paf_mat(
            bid_price=bid_price[tick_i + 1 - SECOND_10_TICK_GAP:tick_i + 1],
            bid_feature=bid_volume[tick_i + 1 - SECOND_10_TICK_GAP:tick_i + 1],
        )  # bid price range and bid paf-volume
        ask_price_range, ask_paf_volume = cal_ask_paf_mat(
            ask_price=ask_price[tick_i + 1 - SECOND_10_TICK_GAP:tick_i + 1],
            ask_feature=ask_volume[tick_i + 1 - SECOND_10_TICK_GAP:tick_i + 1],
        )  # bid price range and bid paf-volume
        # set data
        lob_data_10_second[tick_i // SECOND_10_TICK_GAP, :, 0, 0] = bid_price_range[:5]  # bid price
        lob_data_10_second[tick_i // SECOND_10_TICK_GAP, :, 0, 1] = bid_paf_volume.sum(axis=(0, 2))[:5]  # bid volume
        lob_data_10_second[tick_i // SECOND_10_TICK_GAP, :, 1, 0] = ask_price_range[:5]  # bid price
        lob_data_10_second[tick_i // SECOND_10_TICK_GAP, :, 1, 1] = ask_paf_volume.sum(axis=(0, 2))[:5]  # bid volume
    assert tick_i == TICK_NUM - 1, "tick_i ERROR !"  # test the tick_i
    lob_data_10_second = lob_data_10_second.reshape(TICK_NUM // SECOND_10_TICK_GAP, -1)  # reshape to (28800/20, 20)
    lob_data_10_second_df = pd.DataFrame(lob_data_10_second, columns=LOB_COLUMNS)  # construct the df
    lob_data_10_second_df.to_csv(f"{LOB_10_SECOND_FILE_PATH}/10_seconds_{lob_file}", index=False)
    print(f"   - 3. FINISH 10 SECONDS !!!")

    # - 2.4 compute the 30 seconds lob data, align by price during 60 ticks and sum volume
    # define the empty 30-second lob data, shape=(28800/60, 5, 2, 2), (T, L, D, F)
    lob_data_30_second = np.zeros((TICK_NUM // SECOND_30_TICK_GAP, 5, 2, 2))
    # for loop the tick by `HF` to gen all PA Feature of each tick
    tick_i = 0
    for tick_i in range(SECOND_30_TICK_GAP - 1, TICK_NUM, SECOND_30_TICK_GAP):
        # compute the price range and paf-volume
        bid_price_range, bid_paf_volume = cal_bid_paf_mat(
            bid_price=bid_price[tick_i + 1 - SECOND_30_TICK_GAP:tick_i + 1],
            bid_feature=bid_volume[tick_i + 1 - SECOND_30_TICK_GAP:tick_i + 1],
        )  # bid price range and bid paf-volume
        ask_price_range, ask_paf_volume = cal_ask_paf_mat(
            ask_price=ask_price[tick_i + 1 - SECOND_30_TICK_GAP:tick_i + 1],
            ask_feature=ask_volume[tick_i + 1 - SECOND_30_TICK_GAP:tick_i + 1],
        )  # bid price range and bid paf-volume
        # set data
        lob_data_30_second[tick_i // SECOND_30_TICK_GAP, :, 0, 0] = bid_price_range[:5]  # bid price
        lob_data_30_second[tick_i // SECOND_30_TICK_GAP, :, 0, 1] = bid_paf_volume.sum(axis=(0, 2))[:5]  # bid volume
        lob_data_30_second[tick_i // SECOND_30_TICK_GAP, :, 1, 0] = ask_price_range[:5]  # bid price
        lob_data_30_second[tick_i // SECOND_30_TICK_GAP, :, 1, 1] = ask_paf_volume.sum(axis=(0, 2))[:5]  # bid volume
    assert tick_i == TICK_NUM - 1, "tick_i ERROR !"  # test the tick_i
    lob_data_30_second = lob_data_30_second.reshape(TICK_NUM // SECOND_30_TICK_GAP, -1)  # reshape to (28800/20, 20)
    lob_data_30_second_df = pd.DataFrame(lob_data_30_second, columns=LOB_COLUMNS)  # construct the df
    lob_data_30_second_df.to_csv(f"{LOB_30_SECOND_FILE_PATH}/30_seconds_{lob_file}", index=False)
    print(f"   - 4. FINISH 30 SECONDS !!!")

    # - 2.5 compute the 1-minute lob data, align by price during 120 ticks and sum volume
    # define the empty 1-minute lob data, shape=(28800/120, 5, 2, 2), (T, L, D, F)
    lob_data_1_minute = np.zeros((TICK_NUM // MINUTE_1_TICK_GAP, 5, 2, 2))
    # for loop the tick by `MINUTE_1_TICK_GAP` to gen all PA Feature of each tick
    tick_i = 0
    for tick_i in range(MINUTE_1_TICK_GAP - 1, TICK_NUM, MINUTE_1_TICK_GAP):
        # compute the price range and paf-volume
        bid_price_range, bid_paf_volume = cal_bid_paf_mat(
            bid_price=bid_price[tick_i + 1 - MINUTE_1_TICK_GAP:tick_i + 1],
            bid_feature=bid_volume[tick_i + 1 - MINUTE_1_TICK_GAP:tick_i + 1],
        )  # bid price range and bid paf-volume
        ask_price_range, ask_paf_volume = cal_ask_paf_mat(
            ask_price=ask_price[tick_i + 1 - MINUTE_1_TICK_GAP:tick_i + 1],
            ask_feature=ask_volume[tick_i + 1 - MINUTE_1_TICK_GAP:tick_i + 1],
        )  # bid price range and bid paf-volume
        # set data
        lob_data_1_minute[tick_i // MINUTE_1_TICK_GAP, :, 0, 0] = bid_price_range[:5]  # bid price
        lob_data_1_minute[tick_i // MINUTE_1_TICK_GAP, :, 0, 1] = bid_paf_volume.sum(axis=(0, 2))[:5]  # bid volume
        lob_data_1_minute[tick_i // MINUTE_1_TICK_GAP, :, 1, 0] = ask_price_range[:5]  # bid price
        lob_data_1_minute[tick_i // MINUTE_1_TICK_GAP, :, 1, 1] = ask_paf_volume.sum(axis=(0, 2))[:5]  # bid volume
    assert tick_i == TICK_NUM - 1, "tick_i ERROR !"  # test the tick_i
    lob_data_1_minute = lob_data_1_minute.reshape(TICK_NUM // MINUTE_1_TICK_GAP, -1)  # reshape to (28800/20, 20)
    lob_data_1_minute_df = pd.DataFrame(lob_data_1_minute, columns=LOB_COLUMNS)  # construct the df
    lob_data_1_minute_df.to_csv(f"{LOB_1_MINUTE_FILE_PATH}/1_minute_{lob_file}", index=False)
    print(f"   - 5. FINISH 1 MINUTE !!!")

    # - 2.6 compute the 1-minute lob label
    # compute the mid-price
    mid_price = (lob_data_1_minute_df["BidPrice1"] + lob_data_1_minute_df["AskPrice1"]) / 2
    # set the label
    lob_label_1_minute = np.log((mid_price.shift(-1) / mid_price).values) * 1e4
    # construct the df
    lob_label_1_minute_df = pd.DataFrame(lob_label_1_minute, columns=["1_minute_label"]).fillna(0)  # construct the df
    lob_label_1_minute_df.to_csv(f"{LOB_1_MINUTE_LABEL_PATH}/1_minute_label_{lob_file}", index=False)
    print(f"   - 6. FINISH 1 MINUTE LABEL !!!")
