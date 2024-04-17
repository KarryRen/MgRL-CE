# -*- coding: utf-8 -*-
# @Time    : 2024/4/15 23:15
# @Author  : Karry Ren

""" The preprocess code of raw CSI300 index dataset (download from web following `README.md`).

DESCRIPTION OF RAW `.CSV` DATASET:
    In the `1_minute.csv` file, you can get the index trading data of each 1 minute over a `8-year period from 2016 to 2023`:
        - Totally 1945 days:
                ~ 2016: 244 days
                ~ 2017: 244 days
                ~ 2018: 243 days
                ~ 2019: 244 days
                ~ 2020: 243 days
                ~ 2021: 243 days
                ~ 2022: 242 days
                ~ 2023: 242 days
            and there are 466880 rows data (1 minutes data, each day present 240 rows (4*60)).
        - Each column represents one feature,
            and there are 6 features: OPEN, HIGH, LOW, CLOSE, VOLUME, AMT

Because I want to make the code clear and beautiful, so I need you to do some directory creation by hand !!!

During the preprocessing, we wil do operations by the following steps:
    1. Change the `PATH` based on your situation.
    2. Run this file by `python index_preprocess.py` and you will ge the following directory structure:
        INDEX_DATASET_PATH/
            ├── Train
               ├── 1_day_label.csv
               ├── 1_minute.csv
               ├── 5_minutes.csv
               ├── 15_minutes.csv
               ├── 1_hour.csv
               └── 1_day.csv
            ├── Valid
            └── Test

"""

import pandas as pd
import os

# ---- Define the PARAMS ---- #
TRAIN_DAYS = 244 + 244 + 243 + 244 + 243 + 243  # 2016-244, 2017-244, 2018-243, 2019-244, 2020-243, 2021-243 (1461 days)
VALID_DAYS = 242  # 2022-242
TEST_DAYS = 242  # 2023-242

# ---- Step 1. Change the `PATH` based on your situation ---- #
# the path of download raw data
INDEX_DOWNLOAD_FILE_PATH = "../../../Data/CSI300_index_dataset/1_minute_download.csv"
assert os.path.exists(INDEX_DOWNLOAD_FILE_PATH), f"Please DOWNLOAD the CSI300 index dataset and move it to {INDEX_DOWNLOAD_FILE_PATH} !"
# the path of index dataset
INDEX_DATASET_PATH = "../../../Data/CSI300_index_dataset/dataset"
assert os.path.exists(INDEX_DATASET_PATH), f"Please create the directory structure {INDEX_DATASET_PATH} `BY HAND` !"

# ---- Step 2. Split to Train & Valid & Test, and save the 1_minute.csv ---- #
index_data_1_min = pd.read_csv(INDEX_DOWNLOAD_FILE_PATH)  # read the raw file
train_index_data_1_min = index_data_1_min[:TRAIN_DAYS * 240]  # Train (6 years)
train_index_data_1_min.to_csv(f"{INDEX_DATASET_PATH}/Train/1_minute.csv", index=False)
valid_index_data_15_min = index_data_1_min[TRAIN_DAYS * 240:(TRAIN_DAYS + VALID_DAYS) * 240]  # Valid (1 year)
valid_index_data_15_min.to_csv(f"{INDEX_DATASET_PATH}/Valid/1_minute.csv", index=False)
test_index_data_15_min = index_data_1_min[(TRAIN_DAYS + VALID_DAYS) * 240:]  # Test (1 year)
test_index_data_15_min.to_csv(f"{INDEX_DATASET_PATH}/Test/1_minute.csv", index=False)
print("************************** STEP 2. FINISH 1 MINUTE SPLITTING **************************")

# ---- Step 3. Down-granularity algorithm (5-minutes, 15-minutes, 1-hour and 1-day) csi300 index data ---- #
for data_type in ["Train", "Valid", "Test"]:
    print(f"|| Down-granularity {data_type} ||")
    # Read the 1-minute data, from the saved `.csv` file
    index_data_1_min = pd.read_csv(f"{INDEX_DATASET_PATH}/{data_type}/1_minute.csv")
    # Compute the 5-minute data, 5-`1 minute` group
    index_data_5_min = index_data_1_min.groupby(index_data_1_min.index // 5).agg({
        "SYMBOL": "last", "TRADING_DATE": "last", "TRADING_TIME": "last",
        "OPEN": "first", "HIGH": "max", "LOW": "min", "CLOSE": "last",
        "VOLUME": "sum", "AMT": "sum"
    })
    index_data_5_min.to_csv(f"{INDEX_DATASET_PATH}/{data_type}/5_minutes.csv", index=False)
    # Compute the 15-minute data, 15-`1 minute` group
    index_data_15_min = index_data_1_min.groupby(index_data_1_min.index // 15).agg({
        "SYMBOL": "last", "TRADING_DATE": "last", "TRADING_TIME": "last",
        "OPEN": "first", "HIGH": "max", "LOW": "min", "CLOSE": "last",
        "VOLUME": "sum", "AMT": "sum"
    })
    index_data_15_min.to_csv(f"{INDEX_DATASET_PATH}/{data_type}/15_minutes.csv", index=False)
    # Compute the 1-hour data, 60-`1 minute` group
    index_data_1_hour = index_data_1_min.groupby(index_data_1_min.index // 60).agg({
        "SYMBOL": "last", "TRADING_DATE": "last", "TRADING_TIME": "last",
        "OPEN": "first", "HIGH": "max", "LOW": "min", "CLOSE": "last",
        "VOLUME": "sum", "AMT": "sum"
    })
    index_data_1_hour.to_csv(f"{INDEX_DATASET_PATH}/{data_type}/1_hour.csv", index=False)
    # Compute the 1-day data, 240-`1 minute` group
    index_data_1_day = index_data_1_min.groupby(index_data_1_min.index // 240).agg({
        "SYMBOL": "last", "TRADING_DATE": "last", "TRADING_TIME": "last",
        "OPEN": "first", "HIGH": "max", "LOW": "min", "CLOSE": "last",
        "VOLUME": "sum", "AMT": "sum"
    })
    index_data_1_day.to_csv(f"{INDEX_DATASET_PATH}/{data_type}/1_day.csv", index=False)
    # Compute 1-day label
    close_price = index_data_1_day["CLOSE"]
    # set the label
    index_label_1_day_data = (close_price.shift(-2) / close_price.shift(-1)).values - 1
    # construct the df
    index_label_1_day = pd.DataFrame(index_label_1_day_data, columns=["LABEL"]).fillna(0)  # construct the df
    index_label_1_day["SYMBOL"] = index_data_1_day["SYMBOL"]
    index_label_1_day["TRADING_DATE"] = index_data_1_day["TRADING_DATE"]
    index_label_1_day["TRADING_TIME"] = index_data_1_day["TRADING_TIME"]
    index_label_1_day = index_label_1_day.reindex(columns=["SYMBOL", "TRADING_DATE", "TRADING_TIME", "LABEL"])
    index_label_1_day.to_csv(f"{INDEX_DATASET_PATH}/{data_type}/1_day_label.csv", index=False)
    # Assert for detection
    if data_type == "Train":
        DAYS = TRAIN_DAYS
    elif data_type == "Valid":
        DAYS = VALID_DAYS
    else:
        DAYS = TEST_DAYS
    assert len(index_data_1_day) == DAYS, f"{data_type} 1 day error !!"
    assert len(index_data_1_hour) == DAYS * 4, f"{data_type} 1 hour error !!"
    assert len(index_data_15_min) == DAYS * 16, f"{data_type} 15 minutes error !!"
    assert len(index_data_5_min) == DAYS * 48, f"{data_type} 5 minutes error !!"
    assert len(index_data_1_min) == DAYS * 240, f"{data_type} 1 minute error !!"
    assert len(index_label_1_day) == DAYS, f"{data_type} 1 day labels error !!"

print("************************** FINISH CSI300 INDEX DATASET PREPROCESSING **************************")
