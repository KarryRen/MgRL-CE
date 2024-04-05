# MgRL-CE: A study of time series prediction based on Multi-granularity Residual Learning and Confidence Estimation.

> This repository provides the code for my paper "A study of time series prediction based on Multi-granularity Residual
> Learning and Confidence Estimation."

```python
MgRL-CE/
├── datasets
    ├── datasets_preprocess
        ├── elect_preprocess.py # The pre-process code of UCI electricity dataset (download from web).
    ├── elect_dataset # The torch.Dataset of UCI electricity dataset (after preprocessing).
├── images # All used images of this repo.
├── model # The MgRL-CE models.
    ├── MgRL.py # The BASIC Multi-Granularity Residual Learning Framework.
    ├── MgRL_CE.py # Multi-granularity Residual Learning Framework with Confidence Estimation.
    ├── modules.py # The modules for model.
├── exp_elect # The train&prediction code of UCI electricity dataset.
    ├── elect_train_pred_MgRL.py # Training and Prediction code of `MgRLNet` for UCI electricity dataset.
    ├── elect_config.py # Config file of UCI electricity dataset.
└── utils.py # Some util functions.
```



## Introduction



## Dataset Acquisition

This study extensively performs experiments on 3 real-world Datasets to verify the feasibility of the proposed MgRL-CE. You can **DOWNLOAD** the raw dataset from the following links. 

- **UCI electricity dataset**. Could be downloaded from [**HERE**](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014) ! 

  > The UCI electricity dataset collects the electricity consumption (kWh) every 15 minutes of 370 clients from 2011 to 2014. This study aims to **predict the daily consumption of each client**. I split Train, Valid and Test datasets with 36, 6 and 6 months separately. The granularity of input features is 1 day, 12 hours, 4 hours, 1 hour, and 15 minutes.

- **IF_M0 future Limit Order Book(LOB) dataset**. Updating 🔥.
- **CSI300 Stock dataset**. Updating 🔥.



## Data Pre-Process and `torch.Dataset`

After downloading the datasets following the **Dataset Acquisition**, data preprocessing is needed to get the structured dataset. I have released Pre-Process code for datasets, please read them carefully and **follow the guidelines in the comment rather than running the shell command directly !!!** I also released `torch.Dataset` code for datasets.

- **UCI electricity dataset**. 
  - The Pre-Process code is in `elect_preprocess.py`, [**HERE**](https://github.com/KarryRen/MgRL-CE/blob/main/datasets/datasets_preprocess/elect_preprocess.py) ! You can **RUN** it by `python3 elect_preprocess.py`
  - The  `torch.Dataset` code is in `elect_dataset.py`, [**HERE**](https://github.com/KarryRen/MgRL-CE/blob/main/datasets/elect_dataset.py) ! 
- **IF_M0 future dataset**. 
  - Updating 🔥.
- **CSI300 stock dataset**. 
  - Updating 🔥.



## Training & Prediction

There are many **differences** between the different datasets **during Training and Prediction**, so we built separate training and prediction code for each dataset to run:

- **UCI electricity dataset**. 
  - The Training and Prediction code of `MgRLNet` is in ` elect_train_pred_MgRL.py `, **HERE** ! You can **RUN** it by `python3 elect_train_pred_MgRL.py`
