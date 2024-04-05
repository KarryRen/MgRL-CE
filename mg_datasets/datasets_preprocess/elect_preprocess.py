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
               ├── 15_minutes.csv
               ├── 1_hour.csv
               ├── 4_hours.csv
               ├── 12_hours.csv
               └── 1_day.csv
            ├── Valid
            └── Test

During the preprocessing, we wil do the following things:
    1. Change all `,` in .txt file to `.`, in fact the raw .txt data has some errors, all `,` should be `.` !
    2. Split the raw data to Train/Valid/Test and save to `15_minutes.csv` file as following:
        - Train (36 months, 365+366+365=1096 days, and 105216 rows of data)
        - Valid (6 months, 31+28+31+30+31+30=181 days, and 17376 rows of data)
        - Test (6 months, 31+31+30+31+30+31=184 days, and 17664 rows of data)
    3. Compute other frequency (1-hour, 4-hours, 12-hours and 1-day) uci electricity data (re-frequency).
    4. Compute the daily label.

All in all, after downloading the file from the web, you need:
    1. Change the `UCI_DOWNLOAD_FILE_PATH` and `UCI_PROCESS_FILE_PATH` based on your situation.
    2. Create the directory structure.
    3. Run this file by `python uci_preprocess.py`

"""

import pandas as pd

# ---- Define the PARAMS ---- #
TRAIN_DAYS, VALID_DAYS, TEST_DAYS = 1096, 181, 184

# ---- Step 1. Change the `PATH` based on your situation ---- #
UCI_ELECT_DOWNLOAD_FILE_PATH = "../../../Data/UCI_electricity_dataset/LD2011_2014.txt"
UCI_ELECT_DOWNLOAD_FIX_FILE_PATH = "../../../Data/UCI_electricity_dataset/LD2011_2014(Fix).txt"
UCI_ELECT_DATASET_PATH = "../../../Data/UCI_electricity_dataset/dataset"
print("************************** BEGIN UCI ELECTRICITY DATASET PREPROCESSING **************************")

# ---- Step 2. Change all `,` in .txt file to `.` ---- #
with open(UCI_ELECT_DOWNLOAD_FILE_PATH, "r") as f:  # read the raw data
    wrong_line_list = f.readlines()  # all lines are wrong
    right_line_list = []  # right line list is empty
    for wrong_line in wrong_line_list:  # for loop to change wrong line to right line
        if "," in wrong_line:
            right_line = wrong_line.replace(",", ".")  # change `,` to `.`
        else:
            right_line = wrong_line  # have no `,`, just keep raw
        right_line_list.append(right_line)  # append all lines
with open(UCI_ELECT_DOWNLOAD_FIX_FILE_PATH, "w") as f:  # write the right line to file
    for right_line in right_line_list:
        f.writelines(right_line)
print("************************** FINISH `,` to `.` REPLACING **************************")

# ---- Step 3. Read the raw data, and change the Time type ---- #
uci_data = pd.read_table(UCI_ELECT_DOWNLOAD_FIX_FILE_PATH, delimiter=";")  # read data
uci_data = uci_data.rename(columns={"Unnamed: 0": "Time"})  # change volume name
pd.to_datetime(uci_data["Time"])  # change Time type
print("************************** FINISH RAW FILE READING **************************")

# ---- Step 4. Split to Train & Valid & Test, and save the 15_minutes.csv ---- #
train_uci_data = uci_data[:105216]  # Train (36 months, 365+366+365=1096 days, and 105216 rows of data)
train_uci_data.to_csv(f"{UCI_ELECT_DATASET_PATH}/Train/15_minutes.csv", index=False)
valid_uci_data = uci_data[105216:122592]  # Valid (6 months, 31+28+31+30+31+30=181 days, and 17376 rows of data)
valid_uci_data.to_csv(f"{UCI_ELECT_DATASET_PATH}/Valid/15_minutes.csv", index=False)
test_uci_data = uci_data[122592:]  # Test (6 months, 31+31+30+31+30+31=184 days, and 17664 rows of data)
test_uci_data.to_csv(f"{UCI_ELECT_DATASET_PATH}/Test/15_minutes.csv", index=False)
print("************************** FINISH 15 MINUTES SPLITTING **************************")

# ---- Step 5. Compute other frequency (1-hour, 4-hours, 12-hours and 1-day) uci electricity data ---- #
for data_type in ["Train", "Valid", "Test"]:
    print(data_type)
    # Read the 15 minutes data, from the saved `.csv` file
    uci_data_15_min = pd.read_csv(f"{UCI_ELECT_DATASET_PATH}/{data_type}/15_minutes.csv")
    # Compute the 1-hour data, 4-`15 minutes` group
    uci_data_1_hour = uci_data_15_min.groupby(uci_data_15_min.index // 4).transform("sum")
    uci_data_1_hour["Time"] = uci_data_15_min["Time"]  # change time column
    uci_data_1_hour = uci_data_1_hour[uci_data_1_hour.index % 4 == 3]  # just keep 1-hour data
    uci_data_1_hour.to_csv(f"{UCI_ELECT_DATASET_PATH}/{data_type}/1_hour.csv", index=False)
    # Compute the 4-hours data, 16-`15 minutes` group
    uci_data_4_hours = uci_data_15_min.groupby(uci_data_15_min.index // 16).transform("sum")
    uci_data_4_hours["Time"] = uci_data_15_min["Time"]  # change time column
    uci_data_4_hours = uci_data_4_hours[uci_data_4_hours.index % 16 == 15]  # just keep 4-hours data
    uci_data_4_hours.to_csv(f"{UCI_ELECT_DATASET_PATH}/{data_type}/4_hours.csv", index=False)
    # Compute the 12-hours data, 48-`15 minutes` group
    uci_data_12_hours = uci_data_15_min.groupby(uci_data_15_min.index // 48).transform("sum")
    uci_data_12_hours["Time"] = uci_data_15_min["Time"]  # change time column
    uci_data_12_hours = uci_data_12_hours[uci_data_12_hours.index % 48 == 47]  # just keep 12-hours data
    uci_data_12_hours.to_csv(f"{UCI_ELECT_DATASET_PATH}/{data_type}/12_hours.csv", index=False)
    # Compute the 1-day data, 96-`15 minutes` group
    uci_data_1_day = uci_data_15_min.groupby(uci_data_15_min.index // 96).transform("sum")
    uci_data_1_day["Time"] = uci_data_15_min["Time"]  # change time column
    uci_data_1_day = uci_data_1_day[uci_data_1_day.index % 96 == 95]  # just keep 1-day data
    uci_data_1_day.to_csv(f"{UCI_ELECT_DATASET_PATH}/{data_type}/1_day.csv", index=False)
    # Get the label
    uci_label_1_day = uci_data_1_day.shift(-1)
    uci_label_1_day.to_csv(f"{UCI_ELECT_DATASET_PATH}/{data_type}/label.csv", index=False)
    # Assert for detection
    if data_type == "Train":
        DAYS = TRAIN_DAYS
    elif data_type == "Valid":
        DAYS = VALID_DAYS
    else:
        DAYS = TEST_DAYS
    assert len(uci_data_1_day) == DAYS, f"{data_type} days error !!"
    assert len(uci_data_12_hours) == DAYS * 2, f"{data_type} 12 hours error !!"
    assert len(uci_data_4_hours) == DAYS * 6, f"{data_type} 4 hours error !!"
    assert len(uci_data_1_hour) == DAYS * 24, f"{data_type} 1 hour error !!"
    assert len(uci_data_15_min) == DAYS * 96, f"{data_type} 15 minutes error !!"
    assert len(uci_label_1_day) == DAYS, f"{data_type} labels error !!"
    print(f"{data_type} FINISH !!")

print("************************** FINISH UCI ELECTRICITY DATASET PREPROCESSING **************************")
