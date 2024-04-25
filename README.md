# MgRL-CE: A study of time series prediction based on Multi-granularity Residual Learning and Confidence Estimation.

> This repository provides the code for my paper "A study of time series prediction based on Multi-granularity Residual
> Learning and Confidence Estimation."

```python
MgRL-CE/
├── images # All used images of this repo.
    ├── MgRL_Framework.png # The MgRL FrameWork (without CE).
    ├── CE.png # The Confidence Estimation Module.
    ├── Elect_Data_Distribution.png # The distribution of uci electricity dataset.
    ├── MgRL_CE_Images.pptx # The raw images of MgRL_CE.
├── datasets
    ├── datasets_preprocess
        ├── elect_preprocess.py # The preprocess code of UCI electricity dataset (download from web).
        ├── lob_preprocess # The preprocess package of Future LOB dataset (downlaod from Qlib).
            ├── price_alignment_features # The paf algorithm.
            └── lob_preprocess.py # The preprocess code of Future LOB dataset.
        └── index_preprocess.py # The preprocess code of CSI300 index dataset.
    ├── elect_dataset.py # The torch.Dataset of UCI electricity dataset (after preprocessing).
    ├── lob_dataset.py # The torch.Dataset of Future LOB dataset (after preprocessing).
    └── index_dataset.py # The torch.Dataset of CSI300 index dataset (after preprocessing).
├── configs # The train&prediction config files of 3 datasets.
    ├── elect_config.py # Config file of UCI electricity dataset.
    ├── lob_config.py # Config file of UCI electricity dataset.
    └── index_config.py # Config file of CSI300 index dataset.
├── models # The MgRL-CE models and Comparison Methods.
    ├── MgRL.py # The Multi-Granularity Residual Learning Net: `MgRL_Net` and `MgRL_CE_Net` & `MgRL_Attention_Net`.
    ├── comparison_methods # All comparison methods.
        ├── gru.py # The Comparison Methods 1: GRU(when `use_g=g1`) & 9: Fine-Grained GRU(when `use_g=g5`) & 10 & 11
        ├── lstm.py # The Comparison Methods 2: LSTM.
        ├── transformer.py # The Comparison Methods 3: Transformer.
        ├── deepar.py # The Comparison Methods 4: DeepAR (Will be Updated 🔥 !).
        ├── informer.py # The Comparison Methods 5: Informer (Updating 🔥 !).
        ├── sfm.py # The Comparison Methods 6: SFM.
        ├── alstm.py # The Comparison Methods 7: ALSTM (Will be Updated 🔥 !).
        ├── adv_alstm.py # The Comparison Methods 8: ADV-ALSTM (Will be Updated 🔥 !).
    ├── ablation_methods # All ablation methods.
        ├── mg_add.py # The Ablation Method 1: Mg_Add.
        ├── mg_cat.py # The Ablation Method 2: Mg_Cat.
        └── modules.py # The modules of ablation models.
    ├── loss.py # The loss function of MgRLNet and MgRL_CE_Net.
    ├── metric.py # The metrics of y_ture and y_pred.
    ├── modules.py # The modules of model.
├── train_pred_MgRL.py # Training and Prediction code of `MgRLNet` and `MgRL_CE_Net` for 3 datasets.
├── train_pred_CM.py # Training and Prediction code of Comparison Methods for 3 datasets.
└── utils.py # Some util functions.
```



## Introduction

![MgRL_Framework](./images/MgRL_Framework.png)

<center>Framework of MgRL</center>

![CE](./images/CE.png)

<center>CE Module</center>



## Dataset Acquisition

This study extensively performs experiments on **3 Real-World Datasets** to verify the feasibility of the proposed MgRL-CE. You can **DOWNLOAD** the raw datasets from the following links, and here are also some description about the datasets.

- **UCI electricity dataset (ELECT)**. Could be downloaded from [**HERE**](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014) ! 

  > The UCI electricity dataset collects electricity consumption of each 15 minutes (unit: kW*15min) from a total of **370** clients over a **4-year period from 2011 to 2014**, some of which were created after 2011, and all missing data on electricity consumption for these clients are filled with **ZEROS** !

- **Future Limit Order Book dataset (LOB)**. Could be downloaded from the public `Qlib` platform, [**HERE**](https://github.com/microsoft/qlib) !

  > The Future Limit Order Book dataset collects high-frequency trading data of CSI 300 stock index future (**IF_M0**), including Limit Order Book (LOB) with 5 levels for both ask and bid direction. The **trading frequency is 0.5 seconds**. The dataset **range from Jan. 4, 2022 to Dec. 30, 2022**, covering all **242 trading days** in 2022, with **28,800** trading records for each trading day.

- **CSI300 index dataset (INDEX)**. Could be downloaded based on the public `AKShare` toolkit, [**HRER**](https://github.com/akfamily/akshare) !

  > The CSI300 index dataset collects 1 minute high-frequency trading data for the CSI 300 stock index (000300.SH) from publicly available data sources. The collection interval is from the **beginning of 2016** to the **end of 2023**, containing a total of **1,945 trading days** over an 8-year period, with **240** trading records for each trading day.



## Data Preprocess and `torch.Dataset`

After downloading the datasets following the **Dataset Acquisition**, data preprocessing is needed to get the structured dataset. I have released preprocess code for datasets, please read them carefully and **follow the guidelines in the top comment rather than running the shell command directly !** I have also released `torch.Dataset` code for datasets.

- **UCI electricity dataset (ELECT)**. 

  > In order to minimize the interference caused by missing data, this study intercepts the sample data from the original dataset for the **3-year period from 2012 to 2014**, and excludes the clients with **more than 1 day of missing data** in the interval, and finally retains the electricity consumption data of **320 clients**. The target task of this paper is to **predict the next day's electricity consumption of each client**, and the dataset is divided into training set, validation set and test set according to the time sequence, which covers 24 months, 6 months and 6 months, respectively. The feature data input to the network has **5 kind of granularity**: 1 day (coarsest), 12 hours, 4 hours, 1 hour and 15 minutes (finest).
  >
  > ATTENTION: During the preprocessing I have also **changed the unit of data from kW*15min to kWh** and **adjusted the scale of data distribution by dividing each client data by their daily electricity consumption on the first day**.

  - The preprocess code is in `elect_preprocess.py`, [**HERE**](mg_datasets/datasets_preprocess/elect_preprocess.py) ! You can **RUN** it by：

    ```shell
    python elect_preprocess.py
    ```

  - The  `torch.Dataset` code is in `elect_dataset.py`, [**HERE**](mg_datasets/elect_dataset.py) ! 

- **Future Limit Order Book dataset (LOB)**. 

  > Similarly, the LOB dataset is divided in chronological order: the training, validation, and test sets cover 8, 2, and 2 months, respectively. In this study, the original LOB data is modeled directly, i.e., only the **20 basic features of price and volume** from 1 to 5 ticks in both ask and bid directions are used, and no other factors are constructed manually. The objective is to **predict the minute frequency return of future**, i.e., $y=log(MidPrice_{T+1}/MidPrice_{T})*10^{4}$,  where $MidPrice_{t} = (Price_t^{ask} + Price_t^{bid}) / 2$ denotes the average of the 1 level ask price and bid price in the minute $t$. There are **5 types of input feature granularity**: 1 minute (coarsest), 30 seconds, 10 seconds, 1 second and 0.5 seconds (finest). All feature data were normalized by the Z-Score method.

  - The preprocess code is in `lob_preprocess.py`, [**HERE**](mg_datasets/datasets_preprocess/lob_preprocess) ! You can **RUN** it by：

    ```shell
    # ---- Step 1. Build up the Cython file ---- #
    sh build_cython.sh
    # ---- Step 2. Preprocess the LOB dataset ---- #
    python lob_preprocess.py
    ```

  - The `torch.Dataset` code is in `lob_dataset.py`, [**HERE**](mg_datasets/lob_dataset.py) !

- **CSI300 index dataset (INDEX)**.

  > The training, validation, and test sets span 6 years (2016 to 2021), 1 year (2022), and 1 year (2023), respectively, in chronological order.  **Six commonly used market factors** are extracted as feature inputs, including high price, opening price, low price, closing price, volume and turnover, and all features are normalized by Z-Score method before inputting into the model. This paper also **chooses the daily return of the stock as the prediction target**, i.e., $y=(P_{T+2}/P_{T+1}-1)\times100$ where $P_t$ stands for the average price of the CSI 300 stock index on the $t$-th day. This dataset also has **5 feature granularities**: 1 day (coarsest), 1 hour, 15 minutes, 5 minutes, and 1 minute (finest).

  - The preprocess code is in `index_preprocess.py`, [**HERE**](mg_datasets/datasets_preprocess/index_preprocess) ! You can **RUN** it by:

    ```python
    python3 index_preprocess.py
    ```

  - The `torch.Dataset` code is in `index_dataset.py`, [**HERE**](mg_datasets/index_dataset.py) !



## Training & Prediction

There are some **differences** between the different datasets **during Training and Prediction**. Please carefully set the config files of different datasets following my example.

- **UCI electricity dataset (ELECT)**. 

  - You should firstly set the config file of elect dataset in `elect_config.py`, [**HERE**](configs/elect_config.py) !

  - The Training and Prediction code is in ` train_pred_MgRL.py `, [**HERE**](main/train_pred_MgRL.py) !  You can **RUN** it by:

    ```shell 
    python3 train_pred_MgRL.py --model MgRL_CE_Net --dataset elect
    ```

- **Future Limit Order Book dataset (LOB)**. 

  - You should firstly set the config file of LOB dataset in `lob_config.py`, [**HERE**](configs/lob_config.py) !

  - The Training and Prediction code is in ` train_pred_MgRL.py `, [**HERE**](main/train_pred_MgRL.py) !  You can **RUN** it by:

    ```shell
    python3 train_pred_MgRL.py --model MgRL_CE_Net --dataset lob
    ```

- **CSI300 index dataset (INDEX)**.

  - You should firstly set the config file of index dataset in `index_config.py`, [**HERE**](configs/index_config.py) !

  - The Training and Prediction code is in ` train_pred_MgRL.py `, [**HERE**](main/train_pred_MgRL.py) !  You can **RUN** it by:

    ```shell
    python3 train_pred_MgRL.py --model MgRL_CE_Net --dataset index
    ```



## Comparison Methods

### Comparison Methods List

This study compares the proposed method with numerous other methods. The competitive baselines i compared can be categorized into **4 Groups:**

**GROUP 1. General Time Series Forecasting Models (using single granularity)**

- GRU, [**HERE**](models/comparison_methods/gru.py). [Kyunghyun Cho, et al. 2014](https://arxiv.org/pdf/1406.1078.pdf), [Ref. Code](https://github.com/microsoft/qlib/blob/main/qlib/contrib/model/pytorch_gru.py#L294).
- LSTM, [**HERE**](models/comparison_methods/lstm.py). [Sepp Hochreiter, et al. Neural computation 1997](https://blog.xpgreat.com/file/lstm.pdf), [Ref. Code](https://github.com/microsoft/qlib/blob/main/qlib/contrib/model/pytorch_lstm.py#L286).
- Transformer, [**HERE**](models/comparison_methods/transformer.py). [Ashish Vaswani, et al. NeurIPS 2017](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf), [Ref. Code](https://github.com/microsoft/qlib/blob/main/qlib/contrib/model/pytorch_transformer.py#L258).
- DeepAR, [**HERE** (Will be Updated :fire:)](models/comparison_methods/deep_ar.py). [Salinas D, et al. Int. J. Forecasting 2020](http://162.14.120.130/机器学习-时间序列分析/deepAR.pdf), [Ref. Code](https://github.com/jingw2/demand_forecast/blob/master/deepar.py), [Ref. Video](https://www.bilibili.com/video/BV1iS4y1j7Za/?spm_id_from=333.337.search-card.all.click&vd_source=66823c3216b82637e31f708a5e627a0b). 
- Informer, [**HERE** (Updating :fire:)](models/comparison_methods/informer). [Zhou H, et al. AAAI 2021](https://www.researchgate.net/publication/347125466_Informer_Beyond_Efficient_Transformer_for_Long_Sequence_Time-Series_Forecasting), [Ref. Code](https://github.com/zhouhaoyi/Informer2020/tree/main), [Ref. Video](https://www.bilibili.com/video/BV1AG4y1z7bW/?spm_id_from=333.337.search-card.all.click&vd_source=66823c3216b82637e31f708a5e627a0b)

**GROUP 2. Current TOP Models for Stock Trend Prediction (using single granularity)**

- SFM, [**HERE**](models/comparison_methods/sfm.py). [Liheng Zhang, et al. KDD 2017](https://userpages.umbc.edu/~nroy/courses/fall2018/cmisr/papers/stock_price.pdf), [Ref. Code](https://github.com/microsoft/qlib/blob/main/qlib/contrib/model/pytorch_sfm.py#L25).
- ALSTM, [**HERE** (Will be Updated :fire:)](models/comparison_methods/alstm.py). [Yao Qin, et al. IJCAI 2017](https://arxiv.org/pdf/1704.02971.pdf), [Ref. Code](https://github.com/microsoft/qlib/blob/main/qlib/contrib/model/pytorch_alstm.py#L294).
- ADV-ALSTM, [**HERE** (Will be Updated :fire:)](models/comparison_methods/adv_alstm.py). [Feng F, et al. IJCAI 2019](https://www.ijcai.org/proceedings/2019/0810.pdf), [Ref. Code](https://zhuanlan.zhihu.com/p/566172868).

**GROUP 3. Model Variants (using different granularity of data)**

- Fine-Grained GRU, [**HERE**](models/comparison_methods/gru.py). using only finest-grained data.
- Multi-Grained GRU, [**HERE**](models/comparison_methods/gru.py). using the concatenation of two granularity data.
- Ensemble, [**HERE** (Updating :fire:)](train_pred_Ensemble.py). ensemble result for five independent training models with different granularity data.

**GROUP 4. Two Ablation Models for MgRL_CE  (using different granularity of data)**

- MgRL, [**HERE**](models/MgRL.py), not containing the confidence estimation mechanism in MgRL_CE.
- MgRL_Attention, [**HERE**](models/MgRL.py), replacing the confidence estimation mechanism in MgRL_CE with the classical [**soft attention mechanism**](https://arxiv.org/pdf/1409.0473.pdf?utm_source=ColumnsChannel).

### Run the comparison methods training and prediction

To facilitate the comparison of all Comparison Methods, i built the training and prediction frameworks for each comparison methods of 3 datasets ! 

```shell
python3 train_pred_CM.py --dataset dataset_name --method method_name
```