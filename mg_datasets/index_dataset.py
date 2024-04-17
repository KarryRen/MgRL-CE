# -*- coding: utf-8 -*-
# @Time    : 2024/4/17 11:11
# @Author  : Karry Ren

""" The torch.Dataset of CSI300 index dataset.

After the preprocessing raw CSI300 index dataset (download from web) by
    run `python index_preprocess.py` you will get the following UCI electricity dataset directory:
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

In this dataset:
    - during `__init__()`, we will READ all `.csv` files of multi-granularity data to memory.
    - during `__getitem__()`, we will READ 1 item with multi-granularity data and lag it by `DAY`.

"""

from torch.utils import data
import pandas as pd
import numpy as np


class INDEXDataset(data.Dataset):
    """ The torch.Dataset of CSI300 index dataset. """

    def __init__(self, root_path: str, data_type: str = "Train", time_steps: int = 2):
        """ The init function of ELECTDataset. Will READ all `.csv` files of multi-granularity data to memory.
        For this dataset, the task is predicting the next day index return, so let the daily data be core !!

        :param root_path: the root path of CSI300 index dataset
        :param data_type: the data_type of dataset, you have 3 choices now:
            - "Train" for train dataset
            - "Valid" for valid dataset
            - "Test" for test dataset
        :param time_steps: the time steps (lag steps)

        """

        assert data_type in ["Train", "Valid", "Test"], "data_type ERROR !"

        # ---- Step 0. Set the params ---- #
        self.T = time_steps  # time steps (seq len)
        needed_features = ["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME", "AMT"]

        # ---- Step 2. Read the label and feature ---- #
        self.label = pd.read_csv(f"{root_path}/{data_type}/1_day_label.csv")[["LABEL"]].values  # label, shape=(T, 1)
        self.mg_features_list_dict = {
            "feature_1_day": pd.read_csv(f"{root_path}/{data_type}/1_day.csv")[needed_features].values,
            "feature_1_hour": pd.read_csv(f"{root_path}/{data_type}/1_hour.csv")[needed_features].values,
            "feature_15_minutes": pd.read_csv(f"{root_path}/{data_type}/15_minutes.csv")[needed_features].values,
            "feature_5_minute": pd.read_csv(f"{root_path}/{data_type}/5_minutes.csv")[needed_features].values,
            "feature_1_minute": pd.read_csv(f"{root_path}/{data_type}/1_minute.csv")[needed_features].values
        }  # features key-value pair, each item of dict is a feature data

        # ---- Step 3. Get the total num of date
        self.total_day_num = len(self.label)  # the total number of days
        print(self.mg_features_list_dict)

    def __len__(self):
        """ Get the length of dataset. """

        return self.total_day_num

    def __getitem__(self, idx: int):
        """ Get the item based on idx, and lag the item.

        return: item_data (one lagged day sample)
            - `mg_features`: the multi-granularity (5 kinds of granularity) features of CSI300 index dataset, the format is:
                {
                    "g1": , shape=(time_steps, 1, 6), # feature_1_day
                    "g2": , shape=(time_steps, 4, 6), # feature_1_hour
                    "g3": , shape=(time_steps, 16, 1), # feature_15_minutes
                    "g4": , shape=(time_steps, 48, 1), # feature_5_minutes
                    "g5": , shape=(time_steps, 240, 1) # feature_1_minute
                } shape is (T, K^g, D), please make sure REMEMBER the true time period of each granularity !!!
            - `label`: the return label, shape=(1, )
            - `weight`: the weight, shape=(1, )

        """

        # ---- Compute the index pair day_idx to locate data ---- #
        day_idx = idx  # get the day index to locate the day of daily data
        hour_1_idx = (day_idx + 1) * 4 - 1  # get the 1-hour index
        minute_15_idx = (day_idx + 1) * 16 - 1  # get the 15 minutes index
        minute_5_idx = (day_idx + 1) * 48 - 1  # get the 5 minutes index
        minute_1_idx = (day_idx + 1) * 240 - 1  # get the 1-minute index

        # ---- Get the multi-granularity features, label and weight ---- #
        # feature dict, each item is a list of ndarray with shape=(time_steps, feature_shape)
        mg_features_dict = {"g1": None, "g2": None, "g3": None, "g4": None, "g5": None}
        # meaningless data, features are made to all zeros, erasing the front and tail data
        if day_idx < self.T - 1 or day_idx >= self.total_day_num - 1:
            # set features, all zeros, shape is different from granularity to granularity
            mg_features_dict["g1"] = np.zeros((self.T, 1, 1))  # 1_day granularity
            mg_features_dict["g2"] = np.zeros((self.T, 2, 1))  # 12_hours granularity
            mg_features_dict["g3"] = np.zeros((self.T, 6, 1))  # 4_hours granularity
            mg_features_dict["g4"] = np.zeros((self.T, 24, 1))  # 1_hour granularity
            mg_features_dict["g5"] = np.zeros((self.T, 96, 1))  # 15_minutes granularity
            # `label = 0.0` for loss computation, shape=(1)
            label = np.zeros(1)
            # `weight = 0.0` means data is meaningless, shape=(1)
            weight = np.zeros(1)
        # meaningful data, load the true feature and label
        else:
            # load features, shape is based on granularity, (T, K^g, D)
            mg_features_dict["g1"] = self.mg_features_list_dict["feature_1_day"][client_idx][day_idx - self.T + 1:
                                                                                             day_idx + 1].reshape(self.T, 1, 1)
            mg_features_dict["g2"] = self.mg_features_list_dict["feature_12_hours"][client_idx][hour_12_idx - self.T * 2 + 1:
                                                                                                hour_12_idx + 1].reshape(self.T, 2, 1)
            mg_features_dict["g3"] = self.mg_features_list_dict["feature_4_hours"][client_idx][hour_4_idx - self.T * 6 + 1:
                                                                                               hour_4_idx + 1].reshape(self.T, 6, 1)
            mg_features_dict["g4"] = self.mg_features_list_dict["feature_1_hour"][client_idx][hour_1_idx - self.T * 24 + 1:
                                                                                              hour_1_idx + 1].reshape(self.T, 24, 1)
            mg_features_dict["g5"] = self.mg_features_list_dict["feature_15_minutes"][client_idx][minute_15_idx - self.T * 96 + 1:
                                                                                                  minute_15_idx + 1].reshape(self.T, 96, 1)
            # get the label, shape=(1)
            label = self.label_list[client_idx][day_idx].reshape(1)
            # set `the weight = 1`, shape=(1)
            weight = np.ones(1)

        # ---- Construct item data ---- #
        item_data = {
            "mg_features": mg_features_dict,
            "label": label,
            "weight": weight
        }

        return item_data


if __name__ == "__main__":  # a demo using INDEXDataset
    INDEX_DATASET_PATH = "../../Data/CSI300_index_dataset/dataset"
    data_set = INDEXDataset(INDEX_DATASET_PATH, data_type="Test", time_steps=7)
    print(len(data_set))
