# -*- coding: utf-8 -*-
# @Time    : 2024/4/1 10:45
# @Author  : Karry Ren

""" The torch.Dataset of UCI electricity dataset.

After the preprocessing raw UCI electricity dataset (download from web) by
    run `python uci_preprocess.py` you will get the following I-SPY1 dataset directory:
        UCI_ELECT_DATASET_PATH/
            ├── Train
               ├── 15_minutes.csv
               ├── 1_hour.csv
               ├── 4_hours.csv
               ├── 12_hours.csv
               └── 1_day.csv
            ├── Valid
            └── Test

In this dataset:
    - during `__init__()`, we will READ all `.csv` files of multi-granularity data to memory.
    - during `__getitem__()`, we will READ 1 item with multi-granularity data and lag it by day.

"""

from torch.utils import data
import pandas as pd
import numpy as np


class ELECTDataset(data.Dataset):
    """ The torch.Dataset of UCI electricity dataset. """

    def __init__(self, root_path: str, data_type: str = "Train", time_steps: int = 2):
        """ The init function of UCIDataset. Will READ all `.csv` files of multi-granularity data to memory.
        For this dataset, the task is predicting the daily consumption of each client, so let the daily data be core !!

        :param root_path: the root path of UCI electricity dataset.
        :param data_type: the data_type of dataset, you have 3 choices now:
            - "Train" for train dataset
            - "Valid" for valid dataset
            - "Test" for test dataset
        :param time_steps: the time steps (lag steps)

        """

        # ---- Step 0. Set the params and load all data to memory ---- #
        self.T = time_steps
        feature_1_day = pd.read_csv(f"{root_path}/{data_type}/1_day.csv", index_col=0)  # g1
        feature_12_hours = pd.read_csv(f"{root_path}/{data_type}/12_hours.csv", index_col=0)  # g2
        feature_4_hours = pd.read_csv(f"{root_path}/{data_type}/4_hours.csv", index_col=0)  # g3
        feature_1_hour = pd.read_csv(f"{root_path}/{data_type}/1_hour.csv", index_col=0)  # g4
        feature_15_minutes = pd.read_csv(f"{root_path}/{data_type}/15_minutes.csv", index_col=0)  # g5
        label = pd.read_csv(f"{root_path}/{data_type}/label.csv", index_col=0)

        # ---- Step 1. Read some params from 1-day feature ---- #
        client_list = feature_1_day.columns  # each column represents a client
        self.total_day_nums = len(feature_1_day)  # the total number of days
        self.total_client_nums = len(client_list)  # the total number of clients

        # ---- Step 3. Read all `.csv` files of multi-granularity data to memory. ---- #
        self.label_list = []  # label list, each item is a daily label array (T, 1) for one client
        self.mg_features_list_dict = {
            "feature_1_day": [],
            "feature_12_hours": [],
            "feature_4_hours": [],
            "feature_1_hour": [],
            "feature_15_minutes": []
        }  # features key-value pair, each item of dict is a list of features
        for client in client_list:  # for-loop to append label and feature
            self.label_list.append(label[client].values)
            self.mg_features_list_dict["feature_1_day"].append(feature_1_day[client].values)
            self.mg_features_list_dict["feature_12_hours"].append(feature_12_hours[client].values)
            self.mg_features_list_dict["feature_4_hours"].append(feature_4_hours[client].values)
            self.mg_features_list_dict["feature_1_hour"].append(feature_1_hour[client].values)
            self.mg_features_list_dict["feature_15_minutes"].append(feature_15_minutes[client].values)

    def __len__(self):
        """ Get the length of dataset. """

        return self.total_client_nums * self.total_day_nums

    def __getitem__(self, idx: int):
        """ Get the item based on idx, and lag the item.

        return: item_data (one day of one client)
            - `mg_features`: the multi-granularity (5 kinds of granularity) features of UCI electricity dataset, the format is:
                {
                    "feature_1_day": , shape=(time_steps, 1, 1),
                    "feature_12_hours": , shape=(time_steps, 2, 1),
                    "feature_4_hours": , shape=(time_steps, 6, 1),
                    "feature_1_hour": , shape=(time_steps, 24, 1),
                    "feature_15_minutes": , shape=(time_steps, 96, 1)
                } shape is (T, K^g, D)
            - `label`: the return label, shape=(1)
            - `weight`: the weight, shape=(1)

        """

        # ---- Compute the index pair [client_idx, day_idx] to locate data ---- #
        client_idx = idx // self.total_day_nums  # get the client index to locate the client of data
        day_idx = idx % self.total_day_nums  # get the day index to locate the day of daily data
        hour_12_idx = (day_idx + 1) * 2 - 1  # get the 12 hours index
        hour_4_idx = (day_idx + 1) * 6 - 1  # get the 4 hours index
        hour_1_idx = (day_idx + 1) * 24 - 1  # get the 1 hour idx
        minute_15_idx = (day_idx + 1) * 96 - 1  # get thr

        # ---- Get the multi-granularity features, label and weight ---- #
        mg_features_dict = {
            "feature_1_day": [],
            "feature_12_hours": [],
            "feature_4_hours": [],
            "feature_1_hour": [],
            "feature_15_minutes": []
        }  # feature dict, each item shape=(time_steps, feature_shape)
        # meaningless data, features are made to all zeros, erasing the front and tail data
        if day_idx < self.T - 1 or day_idx >= self.total_day_nums - 1:
            # set features, all zeros, shape is different from granularity to granularity
            mg_features_dict["feature_1_day"] = np.zeros((self.T, 1, 1))
            mg_features_dict["feature_12_hours"] = np.zeros((self.T, 2, 1))
            mg_features_dict["feature_4_hours"] = np.zeros((self.T, 6, 1))
            mg_features_dict["feature_1_hour"] = np.zeros((self.T, 24, 1))
            mg_features_dict["feature_15_minutes"] = np.zeros((self.T, 96, 1))
            # `label = 0.0` for loss computation, shape=(1)
            label = np.zeros(1)
            # `weight = 0.0` means data is meaningless, shape=(1)
            weight = np.zeros(1)
        # meaningful data, load the true feature and label
        else:
            # load features, shape is based on granularity, (T, K^g, D)
            mg_features_dict["feature_1_day"] = self.mg_features_list_dict["feature_1_day"][client_idx][
                                                day_idx - self.T + 1:day_idx + 1].reshape(self.T, 1, 1)
            mg_features_dict["feature_12_hours"] = self.mg_features_list_dict["feature_12_hours"][client_idx][
                                                   hour_12_idx - self.T * 2 + 1:hour_12_idx + 1].reshape(self.T, 2, 1)
            mg_features_dict["feature_4_hours"] = self.mg_features_list_dict["feature_4_hours"][client_idx][
                                                  hour_4_idx - self.T * 6 + 1:hour_4_idx + 1].reshape(self.T, 6, 1)
            mg_features_dict["feature_1_hour"] = self.mg_features_list_dict["feature_1_hour"][client_idx][
                                                 hour_1_idx - self.T * 24 + 1:hour_1_idx + 1].reshape(self.T, 24, 1)
            mg_features_dict["feature_15_minutes"] = self.mg_features_list_dict["feature_15_minutes"][client_idx][
                                                     minute_15_idx - self.T * 96 + 1:minute_15_idx + 1].reshape(self.T, 96, 1)
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


if __name__ == "__main__":  # a demo using UCIDataset
    UCI_ELECT_DATASET_PATH = ("/Users/karry/KarryRen/Scientific-Projects/"
                              "2023-SCU-Graduation-Paper/Code/Data/UCI_electricity_dataset/dataset")

    data_set = ELECTDataset(UCI_ELECT_DATASET_PATH, data_type="Valid", time_steps=1)
    # weight_1_sum = 0
    # for i in range(len(data_set)):
    #     uci_data = data_set[i]
    #     if uci_data["weight"] != [0]:
    #         weight_1_sum += 1
    # print(weight_1_sum)
    print(data_set[0])
