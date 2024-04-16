# -*- coding: utf-8 -*-
# @Time    : 2024/4/16 16:41
# @Author  : Karry Ren

""" The torch.Dataset of Future LOB dataset.

After the preprocessing raw Future LOB dataset(download from Qlib following `README.md`) by
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
    - during `__init__()`, we will READ all `.csv` files of multi-granularity data to memory.
    - during `__getitem__()`, we will READ 1 item with multi-granularity data and lag it by `MINUTE`.

"""
