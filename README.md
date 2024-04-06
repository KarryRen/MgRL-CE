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
    ├── comparision_methods # All comparison methods.
        ├── 
    ├── loss.py # The loss function of MgRLNet and MgRL_CE_Net.
    ├── metric.py # The metrics of y_ture and y_pred.
    ├── modules.py # The modules of model.
├── data_visualization # some data visulization functions
├── configs # The train&prediction code of 3 datasets.
    ├── elect_config.py # Config file of UCI electricity dataset.
├── train_pred_MgRL.py # Training and Prediction code of `MgRLNet` for 3 datasets.
├── train_pred_CM.py # Training and Prediction code of Comparison Methods for 3 datasets.
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

After downloading the datasets following the **Dataset Acquisition**, data preprocessing is needed to get the structured dataset. I have released Pre-Process code for datasets, please read them carefully and **follow the guidelines in the top comment rather than running the shell command directly !** I have also released `torch.Dataset` code for datasets.

- **UCI electricity dataset**. 
  
  > In order to minimize the interference caused by missing data, this study intercepts the sample data from the original dataset for the three-year period from **2012 to 2014**, and excludes the samples with more than 10 days of missing data in the interval, and finally retains the electricity consumption data of **370 clients**. The target task of this paper is to **predict the daily electricity consumption of each clients**, and the dataset is divided into training set, validation set and test set according to the time sequence, which covers 24 months, 6 months and 6 months, respectively. The input network is characterized by **five granularity**: 1 day (coarsest), 12 hours, 4 hours, 1 hour and 15 minutes (finest). During the pre-process i have also **change the unit of data from kW*15min to kWh**.
  
  - The Pre-Process code is in `elect_preprocess.py`, [**HERE**](https://github.com/KarryRen/MgRL-CE/blob/main/datasets/datasets_preprocess/elect_preprocess.py) ! You can **RUN** it by：
  
    ```shell
    python3 elect_preprocess.py
    ```
  
  - The  `torch.Dataset` code is in `elect_dataset.py`, [**HERE**](https://github.com/KarryRen/MgRL-CE/blob/main/datasets/elect_dataset.py) ! 
  
- **IF_M0 future dataset**. 
  
  - Updating 🔥.
  
- **CSI300 stock dataset**.
  
  - Updating 🔥.



## Training & Prediction

There are some **differences** between the different datasets **during Training and Prediction**, so please carefully set the `configs` file of different datasets. Also to facilitate the comparison of two different models `MgRLNet` and `MgRL_CE_Net`, i built two training and prediction frameworks: `train_pred_MgRL.py` and `train_pred_MgRL_CE.py (updating 🔥)`.

- **UCI electricity dataset**. 
  
  - You should set the config file firstly in `elect_config.py` , [**HERE**](https://github.com/KarryRen/MgRL-CE/blob/main/configs/elect_config.py) !
  
  - The Training and Prediction code of `MgRLNet` is in ` train_pred_MgRL.py `, [**HERE**](https://github.com/KarryRen/MgRL-CE/blob/main/train_pred_MgRL.py) !  You can **RUN** it by:
  
     ```shell
     python3 train_pred_MgRL.py --dataset elect
     ```



## Comparison Methods

This study compares the proposed method with numerous other methods. The competitive baselines i compared can be categorized into four groups:

**GROUP 1. General Time Series Forecasting Models (using single granularity)**

- GRU, [**Ref. Paper**](https://arxiv.org/pdf/1406.1078.pdf), 
- LSTM, [**Ref. Paper**](https://blog.xpgreat.com/file/lstm.pdf), 
- Transformer, [**Ref. Paper**](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf), 
- DeepAR, [**Ref. Paper**](http://162.14.120.130/机器学习-时间序列分析/deepAR.pdf), 
- Informer, [**Ref. Paper**](https://www.researchgate.net/publication/347125466_Informer_Beyond_Efficient_Transformer_for_Long_Sequence_Time-Series_Forecasting), 

**GROUP 2. Current TOP Models for Stock Trend Prediction (using single granularity)**





To facilitate the comparison of all Comparison Methods, i built the training and prediction framework: `train_pred_CM.py` of 3 datasets, **HERE** ! You can **RUN** it by:

```shell
python3 train_pred_CM.py --dataset dataset_name --method methond_name
```



