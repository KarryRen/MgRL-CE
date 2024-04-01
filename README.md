# MgRL-CE: A study of time series prediction based on Multi-granularity Residual Learning and Confidence Estimation.

> This repository provides the code for my paper "A study of time series prediction based on Multi-granularity Residual
> Learning and Confidence Estimation."

```python
MgRL-CE/
├── datasets
└── 
```



## Introduction



## Dataset Acquisition

This study extensively performs experiments on 2 real-world Datasets to verify the feasibility of the proposed MgRL-CE. You can **DOWNLOAD** the raw dataset from the following links. 

- **UCI electricity dataset**. Could be downloaded from [**HERE**](https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014) ! 

  > The UCI electricity dataset collects the electricity consumption (kWh) every 15 minutes of 321 clients from 2012 to 2014. This study aims to **predict the daily consumption of each client**. This work splits train, val, test datasets with 24, 6, 6 months separatly. The granularity of input features is 1 day, 12 hours, 4 hours, 1 hour, and 15 minutes.

- **CSI300 stock dataset**. Updating 🔥.



## Data Pre-Process and `torch.Dataset`

After downloading the datasets following the **Dataset Acquisition**, data preprocessing is needed to get the structured dataset. We have released Pre-Process code for datasets, please read them carefully and follow the guidelines in the comment ! Also we released `torch.Dataset` code for datasets.

- **UCI electricity dataset**. 
  - The Pre-Process code is in `ispy_preprocess.py`, [**HERE**](https://github.com/KarryRen/UML/blob/main/dataset/dataset_preprocess/ispy_preprocess.py) ! You can **RUN** it using `python3 ispy_preprocess.py`
  - The  `torch.Dataset` code is in `uci_dataset.py`, [**HERE**](https://github.com/KarryRen/UML/blob/main/dataset/ispy_dataset.py) ! 
- **CSI300 stock dataset**. 
  - Updating 🔥.
