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

We use **3 Datasets** to test our UML network. You can **DOWNLOAD** the raw dataset from the following links. 

- **I-SPY1 Trail Dataset**. Could be downloaded from [**HERE**](https://www.kaggle.com/datasets/saarthakkapse/ispy1-trail-dataset) ! 
- **Refuge Glaucoma**. Could be downloaded from [**HERE**](https://pan.baidu.com/s/1DE8a3UgwGJY85bsr4U7tdw?pwd=2023) ! 



## Data Pre-Process and `torch.Dataset`

After downloading the datasets following the **Dataset Acquisition**, data preprocessing is needed which is to reformat the directory structure  of datasets. We have released Pre-Process code for datasets, please read them carefully and follow the guidelines in the comment ! Also we released `torch.Dataset` code for datasets,

- **I-SPY1 Trail Dataset**. 
  - The Pre-Process code is in `ispy_preprocess.py`, [**HERE**](https://github.com/KarryRen/UML/blob/main/dataset/dataset_preprocess/ispy_preprocess.py) ! You can **RUN** it using `python3 ispy_preprocess.py`
  - The  `torch.Dataset` code is in `ispy_dataset.py`, [**HERE**](https://github.com/KarryRen/UML/blob/main/dataset/ispy_dataset.py) ! 
- **Refuge Glaucoma**. 
  - The Pre-Process code is in `refuge_preprocess.py`, [**HERE**](https://github.com/KarryRen/UML/blob/main/dataset/dataset_preprocess/refuge_preprocess.py) ! You can **RUN** it using `python3 refuge_preprocess.py`
  - The  `torch.Dataset` code is in `refuge_dataset.py`, [**HERE**](https://github.com/KarryRen/UML/blob/main/dataset/refuge_dataset.py) !
