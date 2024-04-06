# -*- coding: utf-8 -*-
# @Time    : 2024/4/1 10:47
# @Author  : Karry Ren

""" The pre-process code of UCI electricity dataset (download from web).

Please `DOWNLOAD` UCI electricity dataset following `README.md`, and you will get the `LD2011_2014.txt`
    file from the web, please move the file to the path `UCI_DOWNLOAD_FILE_PATH`.
In this `.txt` file, you can get the data from 2011 to 2014:
    - Totally 1461 days (2011-365, 2012-366, 2013-365, 2014-365),
        and 140256 rows data (15 minutes data, all days present 96 measures (24*4)).
    - Each column represent one client.

Because we want to make the code clear and beautiful, so we need you to do some directory creation !!!
    you need to create the following directory structure `BY HAND`:
        UCI_ELECT_DATASET_PATH/
            ├── Train
            ├── Valid
            └── Test

During the preprocessing, we wil do the following things:
    1. Set the path.
    2. Intercept the raw elect data for the three-year period `from 2012 to 2014`, totally 1096 days and 105216 15_minutes.
    3. Exclude the samples with more than 1 day of missing data, and just keep 321 clients, while change the unit of data.
    4. The scale of distribution of electricity data varies GREATLY from one client to another !
          We need to adjust the scale of different data distributions !
    5. Split the raw data to Train/Valid/Test and save to `15_minutes.csv` file as following:
        - Train (24 months, 366+365=1096 days, and 70176 rows of data)
        - Valid (6 months, 31+28+31+30+31+30=181 days, and 17376 rows of data)
        - Test (6 months, 31+31+30+31+30+31=184 days, and 17664 rows of data)
    6. Use the down-granularity algorithm to Compute other granularity (1-hour, 4-hours, 12-hours and 1-day) uci electricity data.
       And compute the daily label.

All in all, after downloading the file from the web, you need:
    1. Change the `UCI_DOWNLOAD_FILE_PATH` and `UCI_PROCESS_FILE_PATH` based on your situation.
    2. Create the directory structure `BY HAND` following the top comment.
    3. Run this file by `python elect_preprocess.py` and you will ge the following directory structure:
        UCI_ELECT_DATASET_PATH/
            ├── Train
               ├── 15_minutes.csv
               ├── 1_hour.csv
               ├── 4_hours.csv
               ├── 12_hours.csv
               └── 1_day.csv
            ├── Valid
            └── Test

"""

import pandas as pd
import numpy as np
import os

# ---- Define the PARAMS ---- #
TRAIN_DAYS, VALID_DAYS, TEST_DAYS = 731, 181, 184

# ---- Step 1. Change the `PATH` based on your situation ---- #
UCI_ELECT_DOWNLOAD_FILE_PATH = "../../../Data/UCI_electricity_dataset/LD2011_2014.txt"
UCI_ELECT_DATASET_PATH = "../../../Data/UCI_electricity_dataset/dataset"
print("************************** BEGIN UCI ELECTRICITY DATASET PREPROCESSING **************************")

# ---- Step 2. Intercept the raw data for the three-year period `from 2012 to 2014`, while change kW*15min to kW*h ---- #
elect_data = pd.read_csv(UCI_ELECT_DOWNLOAD_FILE_PATH, sep=";", parse_dates=True, decimal=',')  # read the data
elect_data = elect_data.rename(columns={"Unnamed: 0": "Time"})  # change time column name
pd.to_datetime(elect_data["Time"])  # change Time type (format)
elect_data = elect_data[365 * 96:]  # Intercepts from `2012 to 2014` (cut off the 2011)
print("************************** FINISH INTERCEPTING **************************")

# ---- Step 3. Exclude the samples with more than 1 day of missing data ---- #
elect_data_1_day = elect_data.groupby(elect_data.index // 96).transform("sum")  # group sum
elect_data_1_day["Time"] = elect_data["Time"]  # change time column
elect_data_1_day = elect_data_1_day[elect_data_1_day.index % 96 == 95]  # just keep 1-day data
elect_data_1_day = elect_data_1_day[:1]  # 2012-01-01 to 2012-01-10
keep_clients_list = []  # the clients list going to keep
for client in elect_data_1_day.columns[1:]:  # for-loop all clients
    if np.sum(elect_data_1_day[client].values == 0) < 1:  # no more than 1 day
        keep_clients_list.append(client)  # append the right clients
elect_data = elect_data[["Time"] + keep_clients_list]  # exclude clients and keep `Time` column of 15 minutes
print(f"************************** FINISH EXCLUDING AND GET {len(keep_clients_list)} CLIENTS **************************")
# change the unit of data (from kW*15min to kW*h)
elect_data[keep_clients_list] = elect_data[keep_clients_list] / 4.0
print(f"************************** FINISH CHANGE kW*15min to kW*h **************************")

# ---- Step 4. Adjust the scale of different data distributions ---- #
# get the daily elect data of the first day
elect_data_1_day_of_first_day = elect_data_1_day[["Time"] + keep_clients_list][:1]
# save for future using
elect_data_1_day_of_first_day.to_csv(f"{UCI_ELECT_DATASET_PATH}/elect_data_1_day_of_first_day.csv", index=False)
# change the scale by the first day data
elect_data[keep_clients_list] = elect_data[keep_clients_list].values / elect_data_1_day_of_first_day[keep_clients_list].values
print(f"************************** FINISH ADJUSTING SCALE **************************")

# ---- Step 5. Split to Train & Valid & Test, and save the 15_minutes.csv ---- #
train_elect_data = elect_data[:70176]  # Train (24 months, 366+365=731 days, and 70176 rows of data)
train_elect_data.to_csv(f"{UCI_ELECT_DATASET_PATH}/Train/15_minutes.csv", index=False)
valid_elect_data = elect_data[70176:87552]  # Valid (6 months, 31+28+31+30+31+30=181 days, and 17376 rows of data)
valid_elect_data.to_csv(f"{UCI_ELECT_DATASET_PATH}/Valid/15_minutes.csv", index=False)
test_elect_data = elect_data[87552:]  # Test (6 months, 31+31+30+31+30+31=184 days, and 17664 rows of data)
test_elect_data.to_csv(f"{UCI_ELECT_DATASET_PATH}/Test/15_minutes.csv", index=False)
print("************************** FINISH 15 MINUTES SPLITTING **************************")

# ---- Step 6. Down-granularity algorithm (1-hour, 4-hours, 12-hours and 1-day) uci electricity data ---- #
for data_type in ["Train", "Valid", "Test"]:
    print(data_type)
    # Read the 15 minutes data, from the saved `.csv` file
    elect_data_15_min = pd.read_csv(f"{UCI_ELECT_DATASET_PATH}/{data_type}/15_minutes.csv")
    # Compute the 1-hour data, 4-`15 minutes` group
    elect_data_1_hour = elect_data_15_min.groupby(elect_data_15_min.index // 4).transform("sum")
    elect_data_1_hour["Time"] = elect_data_15_min["Time"]  # change time column
    elect_data_1_hour = elect_data_1_hour[elect_data_1_hour.index % 4 == 3]  # just keep 1-hour data
    elect_data_1_hour.to_csv(f"{UCI_ELECT_DATASET_PATH}/{data_type}/1_hour.csv", index=False)
    # Compute the 4-hours data, 16-`15 minutes` group
    elect_data_4_hours = elect_data_15_min.groupby(elect_data_15_min.index // 16).transform("sum")
    elect_data_4_hours["Time"] = elect_data_15_min["Time"]  # change time column
    elect_data_4_hours = elect_data_4_hours[elect_data_4_hours.index % 16 == 15]  # just keep 4-hours data
    elect_data_4_hours.to_csv(f"{UCI_ELECT_DATASET_PATH}/{data_type}/4_hours.csv", index=False)
    # Compute the 12-hours data, 48-`15 minutes` group
    elect_data_12_hours = elect_data_15_min.groupby(elect_data_15_min.index // 48).transform("sum")
    elect_data_12_hours["Time"] = elect_data_15_min["Time"]  # change time column
    elect_data_12_hours = elect_data_12_hours[elect_data_12_hours.index % 48 == 47]  # just keep 12-hours data
    elect_data_12_hours.to_csv(f"{UCI_ELECT_DATASET_PATH}/{data_type}/12_hours.csv", index=False)
    # Compute the 1-day data, 96-`15 minutes` group
    elect_data_1_day = elect_data_15_min.groupby(elect_data_15_min.index // 96).transform("sum")
    elect_data_1_day["Time"] = elect_data_15_min["Time"]  # change time column
    elect_data_1_day = elect_data_1_day[elect_data_1_day.index % 96 == 95]  # just keep 1-day data
    elect_data_1_day.to_csv(f"{UCI_ELECT_DATASET_PATH}/{data_type}/1_day.csv", index=False)
    # Get the label
    elect_label_1_day = elect_data_1_day.shift(-1)
    elect_label_1_day.to_csv(f"{UCI_ELECT_DATASET_PATH}/{data_type}/label.csv", index=False)
    # Assert for detection
    if data_type == "Train":
        DAYS = TRAIN_DAYS
    elif data_type == "Valid":
        DAYS = VALID_DAYS
    else:
        DAYS = TEST_DAYS
    assert len(elect_data_1_day) == DAYS, f"{data_type} days error !!"
    assert len(elect_data_12_hours) == DAYS * 2, f"{data_type} 12 hours error !!"
    assert len(elect_data_4_hours) == DAYS * 6, f"{data_type} 4 hours error !!"
    assert len(elect_data_1_hour) == DAYS * 24, f"{data_type} 1 hour error !!"
    assert len(elect_data_15_min) == DAYS * 96, f"{data_type} 15 minutes error !!"
    assert len(elect_label_1_day) == DAYS, f"{data_type} labels error !!"
    print(f"|| {data_type} FINISH !! ||")
print("************************** FINISH UCI ELECTRICITY DATASET PREPROCESSING **************************")
