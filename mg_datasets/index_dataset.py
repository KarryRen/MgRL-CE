# -*- coding: utf-8 -*-
# @Time    : 2024/4/17 11:11
# @Author  : Karry Ren

""" The torch.Dataset of CSI300 index dataset.

After the preprocessing raw CSI300 index dataset (download from web) by
    run `python index_preprocess.py` you will get the following CSI300 index dataset directory:
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
    - during `__getitem__()`, we will READ 1 item with multi-granularity data and lag it by `DAY` and do the Z-Score normalization.

"""

from torch.utils import data
import pandas as pd
import numpy as np


class INDEXDataset(data.Dataset):
    """ The torch.Dataset of CSI300 index dataset. """

    def __init__(self, root_path: str, data_type: str = "Train", time_steps: int = 2, need_norm: bool = True):
        """ The init function of INDEXDataset. Will READ all `.csv` files of multi-granularity data to memory.
        For this dataset, the task is predicting the next day index return, so let the daily data be core !!

        :param root_path: the root path of CSI300 index dataset
        :param data_type: the data_type of dataset, you have 3 choices now:
            - "Train" for train dataset
            - "Valid" for valid dataset
            - "Test" for test dataset
        :param time_steps: the time steps (lag steps)
        :param need_norm: whether to normalize the data

        """

        assert data_type in ["Train", "Valid", "Test"], "data_type ERROR !"

        # ---- Step 0. Set the params ---- #
        self.T = time_steps  # time steps (seq len)
        needed_features = ["OPEN", "HIGH", "LOW", "CLOSE", "VOLUME", "AMT"]
        self.need_norm = need_norm  # whether to normalize the lob data

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
        if day_idx < self.T - 1 or day_idx >= self.total_day_num - 2:
            # set features, all zeros, shape is different from granularity to granularity
            mg_features_dict["g1"] = np.zeros((self.T, 1, 6))  # 1_day granularity
            mg_features_dict["g2"] = np.zeros((self.T, 4, 6))  # 1_hour granularity
            mg_features_dict["g3"] = np.zeros((self.T, 16, 6))  # 15_minutes granularity
            mg_features_dict["g4"] = np.zeros((self.T, 48, 6))  # 5_minutes granularity
            mg_features_dict["g5"] = np.zeros((self.T, 240, 6))  # 1_minute granularity
            # `label = 0.0` for loss computation, shape=(1)
            label = np.zeros(1)
            # `weight = 0.0` means data is meaningless, shape=(1)
            weight = np.zeros(1)
        # meaningful data, load the true feature and label
        else:
            # load features, shape is based on granularity, (T, K^g, D)
            mg_features_dict["g1"] = self.mg_features_list_dict[
                                         "feature_1_day"][day_idx - self.T + 1:day_idx + 1].reshape(self.T, 1, 6)
            mg_features_dict["g2"] = self.mg_features_list_dict[
                                         "feature_1_hour"][hour_1_idx - self.T * 4 + 1:hour_1_idx + 1].reshape(self.T, 4, 6)
            mg_features_dict["g3"] = self.mg_features_list_dict[
                                         "feature_15_minutes"][minute_15_idx - self.T * 16 + 1:minute_15_idx + 1].reshape(self.T, 16, 6)
            mg_features_dict["g4"] = self.mg_features_list_dict[
                                         "feature_5_minute"][minute_5_idx - self.T * 48 + 1:minute_5_idx + 1].reshape(self.T, 48, 6)
            mg_features_dict["g5"] = self.mg_features_list_dict[
                                         "feature_1_minute"][minute_1_idx - self.T * 240 + 1:minute_1_idx + 1].reshape(self.T, 240, 6)
            # get the label, shape=(1, )
            label = self.label[day_idx]
            # set `the weight = 1`, shape=(1, )
            weight = np.ones(1)

        # ---- Do the Z-Score Normalization  ---- #
        if self.need_norm:
            for g in ["g1", "g2", "g3", "g4", "g5"]:
                mg_feature = mg_features_dict[g]  # get feature
                # norm the price
                price_mean = mg_feature[:, :, :4].mean()  # compute the mean, shape=(1)
                price_std = mg_feature[:, :, :4].std()  # compute the std, shape=(1)
                mg_feature[:, :, :4] = (mg_feature[:, :, :4] - price_mean) / (price_std + 1e-5)
                # norm the volume & amt
                va_mean = mg_feature[:, :, 4:].mean(axis=(0, 1), keepdims=True)  # compute the mean, shape=(1, 1, 2)
                va_std = mg_feature[:, :, 4:].std(axis=(0, 1), keepdims=True)  # compute the mean, shape=(1, 1, 2)
                mg_feature[:, :, 4:] = (mg_feature[:, :, 4:] - va_mean) / (va_std + 1e-5)
                # set back
                mg_features_dict[g] = mg_feature

        # ---- Construct item data ---- #
        item_data = {
            "mg_features": mg_features_dict,
            "label": label,
            "weight": weight
        }

        return item_data


if __name__ == "__main__":  # a demo using INDEXDataset
    INDEX_DATASET_PATH = "../../Data/CSI300_index_dataset/dataset"

    data_set = INDEXDataset(INDEX_DATASET_PATH, data_type="Train", time_steps=2, need_norm=False)
    for i in range(1, len(data_set) - 2):
        item_data = data_set[i]
        g1_data = item_data["mg_features"]["g1"]
        g2_data = item_data["mg_features"]["g2"]
        g3_data = item_data["mg_features"]["g3"]
        g4_data = item_data["mg_features"]["g4"]
        g5_data = item_data["mg_features"]["g5"]
        g_data_list = [g1_data, g2_data, g3_data, g4_data]
        for g_idx, g_data in enumerate(g_data_list):
            assert (g_data[:, 0, 0] == g5_data[:, 0, 0]).all(), f"g{(g_idx + 1)} error !! OPEN error !!"
            assert (g_data[:, :, 1].max(axis=1) == g5_data[:, :, 1].max(axis=1)).all(), f"g{(g_idx + 1)} error !! HIGH error !!"
            assert (g_data[:, :, 2].min(axis=1) == g5_data[:, :, 2].min(axis=1)).all(), f"g{g_idx + 1} error !! LOW error !!"
            assert (g_data[:, -1, 3] == g5_data[:, -1, 3]).all(), f"g{g_idx + 1} error !! CLOSE error !!"
            assert ((g_data[:, :, 4].sum(axis=1) - g5_data[:, :, 4].sum(axis=1)) / g5_data[:, :, 4].sum(
                axis=1) < 1e-3).all(), f"g{g_idx + 1} error !! VOLUME error !!"
            assert ((g_data[:, :, 5].sum(axis=1) - g5_data[:, :, 5].sum(axis=1)) / g5_data[:, :, 5].sum(
                axis=1) < 1e-3).all(), f"g{g_idx + 1} error !! AMT error !!"
        print(g1_data, g2_data, g3_data, g4_data, g5_data)
        print(item_data["label"])
        break

    data_set = INDEXDataset(INDEX_DATASET_PATH, data_type="Train", time_steps=2)
    for i in range(1, len(data_set) - 2):
        item_data = data_set[i]
        g1_data = item_data["mg_features"]["g1"]
        g2_data = item_data["mg_features"]["g2"]
        g3_data = item_data["mg_features"]["g3"]
        g4_data = item_data["mg_features"]["g4"]
        g5_data = item_data["mg_features"]["g5"]
        print(g1_data, g2_data, g3_data, g4_data, g5_data)
        print(item_data["label"])
        break
