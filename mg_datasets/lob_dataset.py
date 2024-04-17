# -*- coding: utf-8 -*-
# @Time    : 2024/4/16 16:41
# @Author  : Karry Ren

""" The torch.Dataset of Future LOB dataset.

After the preprocessing raw Future LOB dataset (download from Qlib following `README.md`) by
    run `python lob_preprocess.py` you will get the following Future LOB dataset directory:
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

In this dataset:
    - during `__init__()`, we will READ target `.csv` files of multi-granularity data to memory.
    - during `__getitem__()`, we will READ 1 item with multi-granularity data and lag it by `MINUTE`.

"""
import os

from torch.utils import data
import pandas as pd
import numpy as np


class LOBDataset(data.Dataset):
    """ The torch.Dataset of Future LOB dataset. """

    def __init__(self, root_path: str, start_date: str, end_date: str, time_steps: int = 2):
        """ The init function of LOBDataset. Will READ target `.csv` files of multi-granularity data to memory.
        For this dataset, the task is predicting the 1-minute return, so let the 1-minute data be core !!

        :param root_path: the root path of Future LOB dataset
        :param start_date: the start date, format should be "yyyymmdd"
        :param end_date: the end date, format should be "yyyymmdd"
        :param time_steps: the time steps (lag steps)

        NOTE:
            - the start_date and end_date will be [start_date, end_date] meaning start close and end close.
                ~ for train: the start_date and end_date is ["20220101", "20220831"] meaning 8 months (161 days)
                ~ for valid: the start_date and end_date is ["20220901", "20221031"] meaning 2 months (37 days)
                ~ for test: the start_date and end_date is ["20221101", "20221231"] meaning 2 months (44 days)

        """

        assert start_date < end_date, f"end_date muse be greater than start_date !!!"

        # ---- Step 0. Set the params ---- #
        self.T = time_steps  # time steps (seq len)
        self.date_num, self.total_minute_num = 0, 240  # the date_num and tick_num

        # ---- Step 1. Define the feature and label list ---- #
        self.label_list = []  # label list, each item is a daily label array (T, 1) for one date
        self.mg_features_list_dict = {
            "feature_1_minute": [],
            "feature_30_seconds": [],
            "feature_10_seconds": [],
            "feature_1_second": [],
            "feature_0.5_seconds": []
        }  # features key-value pair, each item of dict is a list of features for one date

        # ---- Step 2. Read target `.csv` data date by date ---- #
        file_feature_1_minute_list = sorted(os.listdir(f"{root_path}/1_minute"))  # read all 1_minute files
        for file in file_feature_1_minute_list:  # for loop the date to read
            date = file.split("_")[-1].split(".")[0]  # get the date
            if start_date <= date <= end_date:  # select the date in [start_date, end_date]
                # append the label
                self.label_list.append(pd.read_csv(f"{root_path}/1_minute_label/1_minute_label_{date}.csv").values)
                # append the 1-minute feature
                self.mg_features_list_dict["feature_1_minute"].append(pd.read_csv(f"{root_path}/1_minute/1_minute_{date}.csv").values)
                # append the 30-seconds feature
                self.mg_features_list_dict["feature_30_seconds"].append(pd.read_csv(f"{root_path}/30_seconds/30_seconds_{date}.csv").values)
                # append the 10-seconds feature
                self.mg_features_list_dict["feature_10_seconds"].append(pd.read_csv(f"{root_path}/10_seconds/10_seconds_{date}.csv").values)
                # append the 1-second feature
                self.mg_features_list_dict["feature_1_second"].append(pd.read_csv(f"{root_path}/1_second/1_second_{date}.csv").values)
                # append the 0.5-seconds feature
                self.mg_features_list_dict["feature_0.5_seconds"].append(pd.read_csv(f"{root_path}/0.5_seconds/0.5_seconds_{date}.csv").values)
                # add the date number
                self.date_num += 1

    def __len__(self):
        """ Get the length of dataset. """

        return self.date_num * self.total_minute_num

    def __getitem__(self, idx: int):
        """ Get the item based on idx, and lag the item.

        return: item_data (one lagged minute sample of one date)
            - `mg_features`: the multi-granularity (5 kinds of granularity) features of Future LOB dataset, the format is:
                {
                    "g1": , shape=(time_steps, 1, 1), # feature_1_minute
                    "g2": , shape=(time_steps, 2, 1), # feature_30_seconds
                    "g3": , shape=(time_steps, 6, 1), # feature_10_seconds
                    "g4": , shape=(time_steps, 60, 1), # feature_1_second
                    "g5": , shape=(time_steps, 120, 1) # feature_0.5_seconds
                } shape is (T, K^g, D), please make sure REMEMBER the true time period of each granularity !!!
            - `label`: the return label, shape=(1, )
            - `weight`: the weight, shape=(1, )

        """

        # ---- Compute the index pair [date_idx, minute_idx] to locate data ---- #
        date_idx = idx // self.total_minute_num  # get the date index to locate the date of data
        minute_idx = idx % self.total_minute_num  # get the minute index to locate the minute of daily data
        second_30_idx = (minute_idx + 1) * 2 - 1  # get the 30 seconds index
        second_10_idx = (minute_idx + 1) * 6 - 1  # get the 10 seconds index
        second_1_idx = (minute_idx + 1) * 60 - 1  # get the 1-second index
        second_05_idx = (minute_idx + 1) * 120 - 1  # get the 0.5 seconds index

        # ---- Get the multi-granularity features, label and weight ---- #
        # feature dict, each item is a list of ndarray with shape=(time_steps, feature_shape)
        mg_features_dict = {"g1": None, "g2": None, "g3": None, "g4": None, "g5": None}
        # meaningless data, features are made to all zeros, erasing the front and tail data
        if minute_idx < self.T - 1 or minute_idx >= self.total_minute_num - 1:
            # set features, all zeros, shape is different from granularity to granularity
            mg_features_dict["g1"] = np.zeros((self.T, 1, 20))  # 1_minute granularity
            mg_features_dict["g2"] = np.zeros((self.T, 2, 20))  # 30_seconds granularity
            mg_features_dict["g3"] = np.zeros((self.T, 6, 20))  # 10_seconds granularity
            mg_features_dict["g4"] = np.zeros((self.T, 24, 20))  # 1_second granularity
            mg_features_dict["g5"] = np.zeros((self.T, 96, 20))  # 0.5_seconds granularity
            # `label = 0.0` for loss computation, shape=(1)
            label = np.zeros(1)
            # `weight = 0.0` means data is meaningless, shape=(1)
            weight = np.zeros(1)
        # meaningful data, load the true feature and label
        else:
            # load features, shape is based on granularity, (T, K^g, D)
            mg_features_dict["g1"] = self.mg_features_list_dict[
                                         "feature_1_minute"][date_idx][minute_idx - self.T + 1:minute_idx + 1].reshape(self.T, 1, 20)
            mg_features_dict["g2"] = self.mg_features_list_dict[
                                         "feature_30_seconds"][date_idx][second_30_idx - self.T * 2 + 1:second_30_idx + 1].reshape(self.T, 2, 20)
            mg_features_dict["g3"] = self.mg_features_list_dict[
                                         "feature_10_seconds"][date_idx][second_10_idx - self.T * 6 + 1:second_10_idx + 1].reshape(self.T, 6, 20)
            mg_features_dict["g4"] = self.mg_features_list_dict[
                                         "feature_1_second"][date_idx][second_1_idx - self.T * 60 + 1:second_1_idx + 1].reshape(self.T, 60, 20)
            mg_features_dict["g5"] = self.mg_features_list_dict[
                                         "feature_0.5_seconds"][date_idx][second_05_idx - self.T * 120 + 1:second_05_idx + 1].reshape(self.T, 120, 20)
            # get the label, shape=(1)
            label = self.label_list[date_idx][minute_idx].reshape(1)
            # set `the weight = 1`, shape=(1)
            weight = np.ones(1)

        # ---- Construct item data ---- #
        item_data = {
            "mg_features": mg_features_dict,
            "label": label,
            "weight": weight
        }

        return item_data


if __name__ == "__main__":  # a demo using LOBDataset
    LOB_DATASET_PATH = "../../Data/Future_LOB_dataset/IF_M0"
    data_set = LOBDataset(LOB_DATASET_PATH, start_date="20220901", end_date="20221031", time_steps=2)
    for i in range(len(data_set)):
        g1_data = data_set[i]["mg_features"]["g1"]
        g2_data = data_set[i]["mg_features"]["g2"]
        g3_data = data_set[i]["mg_features"]["g3"]
        g4_data = data_set[i]["mg_features"]["g4"]
        g5_data = data_set[i]["mg_features"]["g5"]
        assert ((g1_data[:, :, 0].max(axis=1) - g5_data[:, :, 0].max(axis=1)) < 1e-3).all(), f"g1 error !! bid 1 price not max !!"
        assert ((g1_data[:, :, 2].min(axis=1) - g5_data[:, :, 2].min(axis=1)) < 1e-3).all(), f"g1 error !! ask 1 price not min !!"
        assert ((g2_data[:, :, 0].max(axis=1) - g5_data[:, :, 0].max(axis=1)) < 1e-3).all(), f"g2 error !! bid 1 price not max !!"
        assert ((g2_data[:, :, 2].min(axis=1) - g5_data[:, :, 2].min(axis=1)) < 1e-3).all(), f"g2 error !! ask 1 price not min !!"
        assert ((g3_data[:, :, 0].max(axis=1) - g5_data[:, :, 0].max(axis=1)) < 1e-3).all(), f"g3 error !! bid 1 price not max !!"
        assert ((g3_data[:, :, 2].min(axis=1) - g5_data[:, :, 2].min(axis=1)) < 1e-3).all(), f"g3 error !! ask 1 price not min !!"
        assert ((g4_data[:, :, 0].max(axis=1) - g5_data[:, :, 0].max(axis=1)) < 1e-3).all(), f"g4 error !! bid 1 price not max !!"
        assert ((g4_data[:, :, 2].min(axis=1) - g5_data[:, :, 2].min(axis=1)) < 1e-3).all(), f"g4 error !! ask 1 price not min !!"
        print(data_set[i]["label"].shape)
        break
    print(data_set[-3])
