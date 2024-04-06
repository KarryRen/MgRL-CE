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
        ├── gru.py # The Comparison Methods 1. GRU.
    ├── loss.py # The loss function of MgRLNet and MgRL_CE_Net.
    ├── metric.py # The metrics of y_ture and y_pred.
    ├── modules.py # The modules of model.
├── configs # The train&prediction code of 3 datasets.
    ├── elect_config.py # Config file of UCI electricity dataset.
├── train_pred_MgRL.py # Training and Prediction code of `MgRLNet` for 3 datasets.
├── train_pred_CM.py # Training and Prediction code of Comparison Methods for 3 datasets.
└── utils.py # Some util functions.
```



## Introduction



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
  
  > In order to minimize the interference caused by missing data, this study intercepts the sample data from the original dataset for the three-year period from **2012 to 2014**, and excludes the samples with more than 1 days of missing data in the interval, and finally retains the electricity consumption data of **320 clients**. The target task of this paper is to **predict the daily electricity consumption of each clients**, and the dataset is divided into training set, validation set and test set according to the time sequence, which covers 24 months, 6 months and 6 months, respectively. The input network is characterized by **five granularity**: 1 day (coarsest), 12 hours, 4 hours, 1 hour and 15 minutes (finest). During the pre-process i have also **change the unit of data from kW*15min to kWh**.
  
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

### Comparison Methods List

This study compares the proposed method with numerous other methods. The competitive baselines i compared can be categorized into **4 Groups:**

**GROUP 1. General Time Series Forecasting Models (using single granularity)**

- GRU: [**HERE**](https://github.com/KarryRen/MgRL-CE/blob/main/model/comparison_methods/gru.py). [**Kyunghyun Cho, et al. 2014**](https://arxiv.org/pdf/1406.1078.pdf), [**Ref. Code**](https://github.com/microsoft/qlib/blob/main/qlib/contrib/model/pytorch_gru.py#L294).
- LSTM, [**Sepp Hochreiter, et al. Neural computation 1997**](https://blog.xpgreat.com/file/lstm.pdf), [**Ref. Code**](https://github.com/microsoft/qlib/blob/main/qlib/contrib/model/pytorch_lstm.py#L286).
- Transformer, [**Ashish Vaswani, et al. NeurIPS 2017**](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf), [**Ref. Code**](https://github.com/microsoft/qlib/blob/main/qlib/contrib/model/pytorch_transformer.py#L258).
- DeepAR, [**Salinas D, et al. Int. J. Forecasting 2020**](http://162.14.120.130/机器学习-时间序列分析/deepAR.pdf), [**Ref. Code**](https://github.com/husnejahan/DeepAR-pytorch/tree/master).
- Informer, [**Zhou H, et al. AAAI 2021**](https://www.researchgate.net/publication/347125466_Informer_Beyond_Efficient_Transformer_for_Long_Sequence_Time-Series_Forecasting), [**Ref. Code**](https://github.com/zhouhaoyi/Informer2020/tree/main).

**GROUP 2. Current TOP Models for Stock Trend Prediction (using single granularity)**

- SFM, [**Liheng Zhang, et al. KDD 2017**](https://userpages.umbc.edu/~nroy/courses/fall2018/cmisr/papers/stock_price.pdf), [**Ref. Code**](https://github.com/microsoft/qlib/blob/main/qlib/contrib/model/pytorch_sfm.py#L25).
- ALSTM, [**Yao Qin, et al. IJCAI 2017**](https://arxiv.org/pdf/1704.02971.pdf), [**Ref. Code**](https://github.com/microsoft/qlib/blob/main/qlib/contrib/model/pytorch_alstm.py#L294).
- ADV-ALSTM, [**Feng F, et al. IJCAI 2019**](https://www.ijcai.org/proceedings/2019/0810.pdf), [**Ref. Code**](https://zhuanlan.zhihu.com/p/566172868).

**GROUP 3. Model Variants (using different granularities of data)**

- Fine-Grained GRU, using only finest-grained data.
- Multi-Grained GRU, using the concatenation of two granularity data.
- Ensemble, ensembleing result for five independent training models with different granularity data.

**GROUP 4. Two Ablation Models for MgRL_CE  (using different granularity of data)**

- MgRL, not containing the confidence estimation mechanism in MgRL_CE.
- MgRL_Attention, replacing the confidence estimation mechanism in MgRL_CE with the classical [**soft attention mechanism**](https://arxiv.org/pdf/1409.0473.pdf?utm_source=ColumnsChannel).

### Run the comparison methods

To facilitate the comparison of all Comparison Methods, i built the training and prediction frameworks for each comparison methods of 3 datasets ! 

```shell
python3 train_pred_CM.py --dataset dataset_name --method method_name
```
