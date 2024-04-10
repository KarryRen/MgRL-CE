# -*- coding: utf-8 -*-
# @Time    : 2024/4/9 16:13
# @Author  : Karry Ren

""" Please follow the `README.md` to download the 1-day and 1-min stock dataset,
      and get the directory like:
        /Users/username/.qlib/qlib_data/qlib_data/
            ├── cn_data
                ├── calendars
                ├── features
                └── instruments
            ├── cn_data_1min
                ├── calendars
                ├── features
                └── instruments
Please make sure you have this directory and the dataset is right !!!
Please run `python daily_stock_preprocess.py` and get the daily dataset, firstly !!!

During the preprocessing, we wil do operations by the following steps:
    1. Change the `PATH` and `daily_stock_df_name` based on your situation.
    2. Read the preprocessed daily stock data.
    3. Prepare the param for hf stock data.
    4. For-Loop to compute high-frequency data.

"""

import numpy as np
import pandas as pd
import argparse
from functools import reduce
from tqdm import tqdm
import copy
import datetime as dt
import qlib
from qlib.data import D

# ---- Build up the parser ---- #
parser = argparse.ArgumentParser()
parser.add_argument("--hf", type=str, default="15_min",
                    help="The high frequency, you have only 4 choices: `1_min`, `5_min`, `15_min`, `1_hour` !!")
args = parser.parse_args()

# ---- Step 1. Change the `PATH` and `daily_stock_df_name` based on your situation ---- #
# please make sure the `PATH` and `daily_stock_df_name` are same as `daily_stock_preprocess.py` !!!
CSI300_STOCK_DATASET_PATH, daily_stock_df_name = "../../../../CMLF_Dataset/CSI300", "1_day.pkl"
# read the high_frequency
high_frequency = args.hf
print(f"************************** BEGIN HIGH-FREQUENCY `{high_frequency}` CSI300 STOCK DATASET PREPROCESSING **************************")

# ---- Step 2. Read the preprocessed daily stock data ---- #
print(f"************************** BEGIN READING THE DAILY STOCK DATA **************************")
label_name_list = ["LABEL"]
daily_stock_df = pd.read_pickle(f"{CSI300_STOCK_DATASET_PATH}/{daily_stock_df_name}")
daily_stock_labels = daily_stock_df[label_name_list]  # read the daily labels
# EXCELLENT STOCK-CODE, MIN-TRADING-DATE, MAX-TRADING-DATE !!!
stock_code_trading_date_df = daily_stock_labels.reset_index().groupby("instrument")["datetime"].agg(["min", "max"])
print(f"************************** FINISH READING THE DAILY STOCK DATA **************************")

# ---- Step 3. Prepare the param for hf stock data ---- #
print(f"************************** BEGIN PREPARING THE PARAM **************************")
# set the tick gap based on the high_frequency
if high_frequency == "1_min":
    tick_gap = 1  # 1 min
elif high_frequency == "5_min":
    tick_gap = 5  # 5 min
elif high_frequency == "15_min":
    tick_gap = 15  # 15 min
elif high_frequency == "1_hour":
    tick_gap = 60  # 1 hour
else:
    raise TypeError(high_frequency)
# compute the tick num of one day
tick_num = 240 // tick_gap
print(f"************************** BEGIN PREPARING THE PARAM, TICK_NUM={tick_num} **************************")

# ---- Step 4. For-Loop to compute high frequency data ---- #
print(f"************************** BEGIN COMPUTING THE HIGH FREQUENCY DATA **************************")
qlib.init(provider_uri={"1min": "/Users/karry/.qlib/qlib_data/cn_data_1min"})  # init the q lib
for stock_code, (start_date, end_date) in tqdm(stock_code_trading_date_df.iterrows()):  # for loop to get each stock code
    # read the minute stock
    minute_stock_df = D.features(
        instruments=[stock_code],
        fields=['$high', '$low', '$close', '$open', '$volume'],
        # start_time=f"{str(start_date)[:10]} 09:31:00", end_time=f"{str(end_date)[:10]} 15:00:00",
        freq="1min", disk_cache=0
    )
    # test have the stock or not
    if minute_stock_df.empty:
        print(f"WARNING: The minute stock data of {stock_code} not found !!!")
        continue
    # just get the minute stock df
    minute_stock_df = minute_stock_df.loc[stock_code]
    if tick_gap > 1:  # other tick gap
        hf_stock_df = minute_stock_df.groupby(minute_stock_df.index.ceil(f"{tick_gap}min")).agg(
            {"$high": "max", "$low": "min", "$close": "last", "$open": "first", "$volume": "sum"}
        )
    else:
        hf_stock_df = minute_stock_df  # keep raw
    print(hf_stock_df)
