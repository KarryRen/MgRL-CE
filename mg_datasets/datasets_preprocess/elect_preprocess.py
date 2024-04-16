# -*- coding: utf-8 -*-
# @Time    : 2024/4/1 10:47
# @Author  : Karry Ren

""" The preprocess code of raw UCI electricity dataset (download from web following `README.md`).

Please `DOWNLOAD` UCI electricity dataset following `README.md`, and you will get the `LD2011_2014.txt`
    file from the web, please move the file to the path `UCI_DOWNLOAD_FILE_PATH` in Step 1.

DESCRIPTION OF RAW .TXT DATASET:
    In the `LD2011_2014.txt` file, you can get the electricity consumption data of each 15 minutes (unit: kW*15min) from
      a total of `370` clients over a `4-year period from 2011 to 2014`:
        - Totally 1461 days (2011-365, 2012-366, 2013-365, 2014-365),
            and there are 140256 rows data (15 minutes data, each day present 96 rows (24*4)).
        - Each column represents one client,
            and there are 370 clients from MT_001 to MT_369

Because I want to make the code clear and beautiful, so I need you to do some directory creation by hand !!!
    You need to create the following directory structure `BY HAND`:
        UCI_ELECT_DATASET_PATH/
            ├── Train
            ├── Valid
            └── Test

During the preprocessing, we wil do operations by the following steps:
    1. Change the `PATH` based on your situation.
    2. Intercept the raw elect data for the 3-year period `from 2012 to 2014`, totally 1096 days and 105216 15_minutes.
    3. Exclude the clients with more than 1 day of missing data, and just keep 321 clients, while changing the unit of data.
    4. The scale of distribution of electricity data varies GREATLY from one client to another.
          We need to adjust the scale of data distribution !
    5. Split the raw data to Train/Valid/Test and save to `15_minutes.csv` file as following:
        - Train (24 months, 366+365=731 days, and 70176 rows of data)
        - Valid (6 months, 31+28+31+30+31+30=181 days, and 17376 rows of data)
        - Test (6 months, 31+31+30+31+30+31=184 days, and 17664 rows of data)
        Here, you may wonder why not just divide the set by date (as will be used for the stock and futures data) ?
        This is because the electricity data is available throughout the day, so there would be a 0/24 criticality issue involved,
            and here we have chosen to divide by rows in order to keep the code more concise.
        Admittedly, this is a very inflexible and error-prone way of dividing the data,
            but due to time constraints, this can only be fixed in subsequent code.
    6. Use the down-granularity algorithm to Compute other granularity (1-hour, 4-hours, 12-hours and 1-day) uci electricity data,
           while computing the daily label.

All in all, after downloading the file from the web, you need:
    1. Create the directory structure `BY HAND` following the top comment.
    2. Change the `UCI_DOWNLOAD_FILE_PATH` and `UCI_PROCESS_FILE_PATH` based on your situation.
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
import matplotlib.pyplot as plt
import seaborn as sns

# ---- Define the PARAMS ---- #
TRAIN_DAYS = 365 + 366  # 2012-366, 2013-365
VALID_DAYS = 181  # 1 to 6 month of 2014: 31+28+31+30+31+30=181 days
TEST_DAYS = 184  # 7 to 12 month of 2014: 31+31+30+31+30+31=184 days
DRAW_VERBOSE = True  # whether draw the data distribution png images

# ---- Step 1. Change the `PATH` based on your situation ---- #
UCI_ELECT_DOWNLOAD_FILE_PATH = "../../../Data/UCI_electricity_dataset/LD2011_2014.txt"
assert os.path.exists(UCI_ELECT_DOWNLOAD_FILE_PATH), \
    f"Please DOWNLOAD the UCI electricity dataset following `README.md` ! And move `LD2011_2014.txt` to {UCI_ELECT_DOWNLOAD_FILE_PATH} !"
UCI_ELECT_DATASET_PATH = "../../../Data/UCI_electricity_dataset/dataset"
assert os.path.exists(UCI_ELECT_DOWNLOAD_FILE_PATH), \
    f"Please create the directory structure `BY HAND` following top comment: UCI_ELECT_DATASET_PATH/ ├── Train ├── Valid └── Test !"
print("************************** BEGIN UCI ELECTRICITY DATASET PREPROCESSING **************************")

# ---- Step 2. Intercept the raw 15_min data for the three-year period `from 2012 to 2014` ---- #
# use the `pd.read_csv()` interface to read `.txt` file, the `sep` is `;` and change the `,` to `.` !
# use `parse_dates` to change dates
elect_data_15_min = pd.read_csv(UCI_ELECT_DOWNLOAD_FILE_PATH, sep=";", parse_dates=True, decimal=',')
# rename the Time column
elect_data_15_min = elect_data_15_min.rename(columns={"Unnamed: 0": "Time"})
# intercept the raw elect data for the 3-year period `from 2012 to 2014`
elect_data_15_min = elect_data_15_min[365 * 96:]  # cut off 2011 (365 * 96 rows)
assert len(elect_data_15_min) == 105216, f"elect data of 15 minutes length ERROR, should be 105216, now is {len(elect_data_15_min)} !"
print("************************** STEP 2. FINISH INTERCEPTING **************************")

# ---- Step 3. Exclude the clients with more than 1 day of missing data, while changing the unit of data ---- #
# compute the 1_day electricity data, using group by 96 15_minutes
elect_data_1_day = elect_data_15_min.groupby(elect_data_15_min.index // 96).transform("sum")
elect_data_1_day["Time"] = elect_data_15_min["Time"]  # change the time column value
elect_data_1_day = elect_data_1_day[elect_data_1_day.index % 96 == 95]  # just keep 1-day data
assert len(elect_data_1_day) == 1096, f"elect data of 1 day length ERROR, should be 1096, now is {len(elect_data_1_day)} !"
# check the missing data of the first day, and get the keep_clients_list
elect_data_1_day = elect_data_1_day[:1]  # 2012-01-01 daily data (just 1 day)
keep_clients_list = []  # the clients list going to keep
for client in elect_data_1_day.columns[1:]:  # for-loop all clients
    if np.sum(elect_data_1_day[client].values == 0) < 1:  # no more than 1 day
        keep_clients_list.append(client)  # append the right clients
# exclude the clients
elect_data_15_min = elect_data_15_min[["Time"] + keep_clients_list]
elect_data_1_day = elect_data_1_day[["Time"] + keep_clients_list]
# draw raw_elect_distribution(kW*15min)
if DRAW_VERBOSE:
    plt.figure()
    sns.distplot(elect_data_15_min[keep_clients_list].values.flatten(), hist=True, label="15 mins elect (kW*15min)", color="#FFCC33")
    plt.legend()
    plt.xlabel("")
    plt.savefig(f"{UCI_ELECT_DATASET_PATH}/raw_elect_distribution(kW*15min).png", dpi=200, bbox_inches="tight")
print(f"************************** STEP 3.1 FINISH EXCLUDING AND GET `{len(keep_clients_list)}` CLIENTS **************************")
# change the unit of data (from kW*15min to kW*h)
elect_data_15_min[keep_clients_list] = elect_data_15_min[keep_clients_list] / 4.0
# raw_elect_distribution(kW*h)
if DRAW_VERBOSE:
    plt.figure()
    sns.distplot(elect_data_15_min[keep_clients_list].values.flatten(), hist=True, label="15 mins elect (kW*h)", color="#FFCC33")
    plt.legend()
    plt.xlabel("")
    plt.savefig(f"{UCI_ELECT_DATASET_PATH}/raw_elect_distribution(kW*h).png", dpi=200, bbox_inches="tight")
print(f"************************** STEP 3.2 FINISH CHANGE kW*15min to kW*h **************************")

# ---- Step 4. Adjust the scale of different data distributions ---- #
# get the daily elect data of the first day
elect_data_1_day_of_first_day = elect_data_1_day[:1]
# save for future using (when do prediction, use the `.csv` file to get data of raw distribution)
elect_data_1_day_of_first_day.to_csv(f"{UCI_ELECT_DATASET_PATH}/elect_data_1_day_of_first_day.csv", index=False)
# change the scale by dividing each client data by their daily electricity consumption on the first day
elect_data_15_min[keep_clients_list] = elect_data_15_min[keep_clients_list].values / elect_data_1_day_of_first_day[keep_clients_list].values
if DRAW_VERBOSE:
    plt.figure()
    sns.distplot(elect_data_15_min[keep_clients_list].values.flatten(), hist=True, label="15 mins elect", color="#FFCC33")
    plt.legend()
    plt.xlabel("")
    plt.savefig(f"{UCI_ELECT_DATASET_PATH}/adj_elect_distribution.png", dpi=200, bbox_inches="tight")
    plt.close()
print(f"************************** STEP 4 FINISH ADJUSTING SCALE **************************")

# ---- Step 5. Split to Train & Valid & Test, and save the 15_minutes.csv ---- #
train_elect_data_15_min = elect_data_15_min[:70176]  # Train (24 months, 366+365=731 days, and 70176 rows of data)
train_elect_data_15_min.to_csv(f"{UCI_ELECT_DATASET_PATH}/Train/15_minutes.csv", index=False)
valid_elect_data_15_min = elect_data_15_min[70176:87552]  # Valid (6 months, 31+28+31+30+31+30=181 days, and 17376 rows of data)
valid_elect_data_15_min.to_csv(f"{UCI_ELECT_DATASET_PATH}/Valid/15_minutes.csv", index=False)
test_elect_data_15_min = elect_data_15_min[87552:]  # Test (6 months, 31+31+30+31+30+31=184 days, and 17664 rows of data)
test_elect_data_15_min.to_csv(f"{UCI_ELECT_DATASET_PATH}/Test/15_minutes.csv", index=False)
print("************************** STEP 5. FINISH 15 MINUTES SPLITTING **************************")

# ---- Step 6. Down-granularity algorithm (1-hour, 4-hours, 12-hours and 1-day) uci electricity data ---- #
for data_type in ["Train", "Valid", "Test"]:
    print(f"|| Down-granularity {data_type} ||")
    # Read the 15 minutes data, from the saved `.csv` file
    elect_data_15_min = pd.read_csv(f"{UCI_ELECT_DATASET_PATH}/{data_type}/15_minutes.csv")
    # Compute the 1-hour data, 4-`15 minutes` group
    elect_data_1_hour = elect_data_15_min.groupby(elect_data_15_min.index // 4).transform("sum")
    elect_data_1_hour["Time"] = elect_data_15_min["Time"]  # change time column
    elect_data_1_hour = elect_data_1_hour[elect_data_1_hour.index % 4 == 3]  # just keep 1-hour data
    elect_data_1_hour.to_csv(f"{UCI_ELECT_DATASET_PATH}/{data_type}/1_hour.csv", index=False)
    # Compute the 4-hours data, 16-`15 minutes` group
    elect_data_4_hour = elect_data_15_min.groupby(elect_data_15_min.index // 16).transform("sum")
    elect_data_4_hour["Time"] = elect_data_15_min["Time"]  # change time column
    elect_data_4_hour = elect_data_4_hour[elect_data_4_hour.index % 16 == 15]  # just keep 4-hours data
    elect_data_4_hour.to_csv(f"{UCI_ELECT_DATASET_PATH}/{data_type}/4_hours.csv", index=False)
    # Compute the 12-hours data, 48-`15 minutes` group
    elect_data_12_hour = elect_data_15_min.groupby(elect_data_15_min.index // 48).transform("sum")
    elect_data_12_hour["Time"] = elect_data_15_min["Time"]  # change time column
    elect_data_12_hour = elect_data_12_hour[elect_data_12_hour.index % 48 == 47]  # just keep 12-hours data
    elect_data_12_hour.to_csv(f"{UCI_ELECT_DATASET_PATH}/{data_type}/12_hours.csv", index=False)
    # Compute the 1-day data, 96-`15 minutes` group
    elect_data_1_day = elect_data_15_min.groupby(elect_data_15_min.index // 96).transform("sum")
    elect_data_1_day["Time"] = elect_data_15_min["Time"]  # change time column
    elect_data_1_day = elect_data_1_day[elect_data_1_day.index % 96 == 95]  # just keep 1-day data
    elect_data_1_day.to_csv(f"{UCI_ELECT_DATASET_PATH}/{data_type}/1_day.csv", index=False)
    # Compute the label
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
    assert len(elect_data_12_hour) == DAYS * 2, f"{data_type} 12 hours error !!"
    assert len(elect_data_4_hour) == DAYS * 6, f"{data_type} 4 hours error !!"
    assert len(elect_data_1_hour) == DAYS * 24, f"{data_type} 1 hour error !!"
    assert len(elect_data_15_min) == DAYS * 96, f"{data_type} 15 minutes error !!"
    assert len(elect_label_1_day) == DAYS, f"{data_type} labels error !!"
    print(f"|| STEP 6. {data_type} FINISH !! ||")
print("************************** FINISH UCI ELECTRICITY DATASET PREPROCESSING **************************")
