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
    ├── MgRL.py # The Multi-Granularity Residual Learning Framework (includes two core models).
    ├── comparision_methods.py # All comparison methods.
    ├── loss.py # The loss function of MgRLNet and MgRL_CE_Net.
    ├── metric.py # The metrics of y_ture and y_pred.
    ├── modules.py # The modules of model.
├── data_visualization
├── configs # The train&prediction code of 3 datasets.
    ├── elect_config.py # Config file of UCI electricity dataset.
├── train_pred_MgRL.py # Training and Prediction code of `MgRLNet` for 3 datasets.
└── utils.py # Some util functions.
```



## Introduction

Here are two new Net : `MgRLNet` and `MgRL_CE_Net`.



## Dataset Acquisition

This study extensively performs experiments on 3 real-world Datasets to verify the feasibility of the proposed MgRL-CE. You can **DOWNLOAD** the raw dataset from the following links. 

- **UCI electricity dataset (ELECT)**. Could be downloaded from [**HERE**](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014) ! 

  > The UCI electricity dataset collects electricity consumption (kW of each 15mins) of **370 clients** over a 4-year period from 2011 to 2014, some of which were created after 2011, and all of the missing electricity consumption data for these customers are filled with **ZEROS**.
  >

- **IF_M0 future Limit Order Book dataset (LOB)**. Updating 🔥.
- **CSI300 Stock dataset (STOCK)**. Updating 🔥.



## Data Pre-Process and `torch.Dataset`

After downloading the datasets following the **Dataset Acquisition**, data preprocessing is needed to get the structured dataset. I have released Pre-Process code for datasets, please read them carefully and **follow the guidelines in the top comment rather than running the shell command directly !!!** I also released `torch.Dataset` code for datasets.

- **UCI electricity dataset**. 
  
  > In order to minimize the interference caused by missing data, this study intercepts the sample data from the original dataset for the three-year period from **2012 to 2014**, and excludes the samples with more than 10 days of missing data in the interval, and finally retains the electricity consumption data of **370 clients**. The target task of this paper is to **predict the daily electricity consumption of each clients**, and the dataset is divided into training set, validation set and test set according to the time sequence, which covers 24 months, 6 months and 6 months, respectively. The input network is characterized by **five granularity**: 1 day, 12 hours, 4 hours, 1 hour and 15 minutes.
  
  - The Pre-Process code is in `elect_preprocess.py`, [**HERE**](https://github.com/KarryRen/MgRL-CE/blob/main/datasets/datasets_preprocess/elect_preprocess.py) ! You can **RUN** it by `python3 elect_preprocess.py`
  - The  `torch.Dataset` code is in `elect_dataset.py`, [**HERE**](https://github.com/KarryRen/MgRL-CE/blob/main/datasets/elect_dataset.py) ! 
  
- **IF_M0 future dataset**. 
  - Updating 🔥.
  
- **CSI300 stock dataset**. 
  - Updating 🔥.



## Training & Prediction

There are many **differences** between the different datasets **during Training and Prediction**, so please carefully set the `configs` file of different datasets.

- **UCI electricity dataset**. 
  - You should set the config file firstly in `elect_config.py` , **HERE** !
  
  - The Training and Prediction code of `MgRLNet` is in ` train_pred_MgRL.py `, **HERE** ! 
  
    You can **RUN** it by `python3.8 train_pred_MgRL.py --dataset elect`
