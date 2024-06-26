# -*- coding: utf-8 -*-
# @Time    : 2024/4/5 09:33
# @Author  : Karry Ren

""" Training and Prediction code of 3 Nets for 3 datasets:
    - MgRL_Net
    - MgRL_Attention_Net
    - MgRL_CE_Net

Training, Validation and Prediction be together !
    Here are two functions:
        - `train_valid_model()` => train and valid model, and save trained models of all epochs.
        - `pred_model()` => use the best model to do prediction (test model).

You can run this python script by: `python3 train_pred_MgRL.py --model model_name --dataset dataset_name`

"""

import os
import logging
import torch
import torch.utils.data as data
import numpy as np
from tqdm import tqdm
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from mg_datasets.elect_dataset import ELECTDataset
from mg_datasets.lob_dataset import LOBDataset
from mg_datasets.index_dataset import INDEXDataset
from utils import fix_random_seed, load_best_model
from models.metrics import r2_score, corr_score, rmse_score, mae_score

# ---- Init and parse the args parser ---- #
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset", type=str, default="elect",
    help="The dataset name. You have only 3 choices: `elect`, `lob`, `index`."
)
parser.add_argument(
    "--method", type=str, default="MgRL",
    help="The model name. You have 3 choices now: "
         "- `MgRL` for MgRL,\n"
         "- `MgRL_Attention` for MgRL_Attention, \n"
         "- `MgRL_CE` for MgRL_CE"
)
args = parser.parse_args()

# ---- Based on the args set the different params ---- #
# dataset name
if args.dataset == "elect":  # the UCI electricity dataset.
    import configs.elect_config as config
elif args.dataset == "lob":  # the Future LOB dataset
    import configs.lob_config as config
elif args.dataset == "index":  # the CSI300 index dataset
    import configs.index_config as config
else:
    raise TypeError(args.dataset)
# method name
METHOD_NAME = args.method
if METHOD_NAME == "MgRL":  # The MgRL_Net
    from models.MgRL import MgRL_Net
    from models.loss import MgRL_Loss
elif METHOD_NAME == "MgRL_Attention":  # The MgRL_Net
    from models.MgRL import MgRL_Attention_Net
    from models.loss import MgRL_Loss
elif METHOD_NAME == "MgRL_CE":  # The MgRL_CE_Net
    from models.MgRL import MgRL_CE_Net
    from models.loss import MgRL_CE_Loss
else:
    raise f"{METHOD_NAME} is ERROR ! Please check your command of `--method` !"


def train_valid_model() -> None:
    """ Train & Valid Model. """

    logging.info(f"***************** RUN TRAIN&VALID MODEL: `{METHOD_NAME}`, DATASET: `{args.dataset}` ! *****************")

    # ---- Get the train and valid device ---- #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"***************** In device {device}   *****************")

    # ---- Make the dataset and dataloader ---- #
    logging.info(f"***************** BEGIN MAKE DATASET & DATALOADER of `{args.dataset}` ! *****************")
    logging.info(f"||| time_steps = {config.TIME_STEPS}, batch size = {config.BATCH_SIZE} |||")
    logging.info(f"**** TRAINING DATASET & DATALOADER ****")
    # make the dataset and dataloader of training
    logging.info(f"**** TRAINING DATASET & DATALOADER ****")
    if args.dataset == "elect":  # the UCI electricity dataset.
        train_dataset = ELECTDataset(root_path=config.UCI_ELECT_DATASET_PATH, data_type="Train", time_steps=config.TIME_STEPS)
    elif args.dataset == "lob":  # the Future LOB dataset
        train_dataset = LOBDataset(
            root_path=config.LOB_DATASET_PATH,
            start_date=config.TRAIN_START_DATE, end_date=config.TRAIN_END_DATE, need_norm=config.NEED_NORM
        )
    elif args.dataset == "index":  # the CSI300 index dataset
        train_dataset = INDEXDataset(root_path=config.INDEX_DATASET_PATH, data_type="Train", need_norm=config.NEED_NORM)
    else:
        raise TypeError(args.dataset)
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)  # the train dataloader
    logging.info(f"**** VALID DATASET & DATALOADER ****")
    if args.dataset == "elect":  # the UCI electricity dataset.
        valid_dataset = ELECTDataset(root_path=config.UCI_ELECT_DATASET_PATH, data_type="Valid", time_steps=config.TIME_STEPS)
    elif args.dataset == "lob":  # the Future LOB dataset
        valid_dataset = LOBDataset(
            root_path=config.LOB_DATASET_PATH,
            start_date=config.VALID_START_DATE, end_date=config.VALID_END_DATE, need_norm=config.NEED_NORM
        )
    elif args.dataset == "index":  # the CSI300 index dataset
        valid_dataset = INDEXDataset(root_path=config.INDEX_DATASET_PATH, data_type="Valid", need_norm=config.NEED_NORM)
    else:
        raise TypeError(args.dataset)
    valid_loader = data.DataLoader(dataset=valid_dataset, batch_size=config.BATCH_SIZE, shuffle=False)  # the valid dataloader
    logging.info("***************** DATASET MAKE OVER ! *****************")
    logging.info(f"Train dataset: length = {len(train_dataset)}")
    logging.info(f"Valid dataset: length = {len(valid_dataset)}")

    # ---- Construct the model and transfer device, while making loss and optimizer ---- #
    logging.info(f"***************** BEGIN BUILD UP THE MODEL `{METHOD_NAME}`! *****************")
    if METHOD_NAME == "MgRL":  # The MgRL_Net and loss
        model = MgRL_Net(
            granularity_dict=config.GRANULARITY_DICT, ga_K=config.GA_K,
            encoding_input_size=config.ENCODING_INPUT_SIZE, encoding_hidden_size=config.ENCODING_HIDDEN_SIZE,
            device=device
        )
        criterion = MgRL_Loss(reduction=config.LOSS_REDUCTION, lambda_1=config.LAMBDA_1)
    elif METHOD_NAME == "MgRL_Attention":  # The MgRL_Attention_Net and loss
        model = MgRL_Attention_Net(
            granularity_dict=config.GRANULARITY_DICT, ga_K=config.GA_K,
            encoding_input_size=config.ENCODING_INPUT_SIZE, encoding_hidden_size=config.ENCODING_HIDDEN_SIZE,
            device=device
        )
        criterion = MgRL_Loss(reduction=config.LOSS_REDUCTION, lambda_1=config.LAMBDA_1)
    elif METHOD_NAME == "MgRL_CE":  # The MgRL_CE_Net and loss
        model = MgRL_CE_Net(
            granularity_dict=config.GRANULARITY_DICT, ga_K=config.GA_K,
            encoding_input_size=config.ENCODING_INPUT_SIZE, encoding_hidden_size=config.ENCODING_HIDDEN_SIZE,
            negative_sample_num=config.NEGATIVE_SAMPLE_NUM,
            device=device
        )
        criterion = MgRL_CE_Loss(reduction=config.LOSS_REDUCTION, lambda_1=config.LAMBDA_1)
    else:
        raise TypeError(METHOD_NAME)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR, weight_decay=config.LAMBDA_THETA)  # the optimizer

    # ---- Start Train and Valid ---- #
    # init the metric dict of all epochs
    epoch_metric = {
        # train & valid loss
        "train_loss": np.zeros(config.EPOCHS), "valid_loss": np.zeros(config.EPOCHS),
        # train & valid r2
        "train_R2": np.zeros(config.EPOCHS), "valid_R2": np.zeros(config.EPOCHS),
        # train & valid CORR
        "train_CORR": np.zeros(config.EPOCHS), "valid_CORR": np.zeros(config.EPOCHS),
        # train & valid RMSE
        "train_RMSE": np.zeros(config.EPOCHS), "valid_RMSE": np.zeros(config.EPOCHS),
        # train & valid MAE
        "train_MAE": np.zeros(config.EPOCHS), "valid_MAE": np.zeros(config.EPOCHS)
    }

    # train model epoch by epoch
    logging.info(f"***************** BEGIN TRAINING AND VALID THE MODEL `{METHOD_NAME}` ! *****************")
    # start train and valid during train
    for epoch in tqdm(range(config.EPOCHS)):
        # start timer for one epoch
        t_start = datetime.now()
        # set the array for one epoch to store (all empty)
        train_loss_one_epoch, valid_loss_one_epoch = [], []
        train_dataset_len, valid_dataset_len = len(train_dataset), len(valid_dataset)
        train_preds_one_epoch = torch.zeros(train_dataset_len).to(device=device)
        train_labels_one_epoch = torch.zeros(train_dataset_len).to(device=device)
        train_weights_one_epoch = torch.zeros(train_dataset_len).to(device=device)
        valid_preds_one_epoch = torch.zeros(valid_dataset_len).to(device=device)
        valid_labels_one_epoch = torch.zeros(valid_dataset_len).to(device=device)
        valid_weights_one_epoch = torch.zeros(valid_dataset_len).to(device=device)
        # - TRAIN model
        last_step = 0
        model.train()
        for batch_data in tqdm(train_loader):
            # move data to device, and change the dtype from double to float32
            mg_features = batch_data["mg_features"]
            labels = batch_data["label"].to(device=device, dtype=torch.float32)
            weights = batch_data["weight"].to(device=device, dtype=torch.float32)
            # zero_grad, forward, compute loss, backward and optimize
            optimizer.zero_grad()
            outputs = model(mg_features)
            if METHOD_NAME in ("MgRL", "MgRL_Attention"):
                preds, rec_residuals = outputs["pred"], outputs["rec_residuals"]
                loss = criterion(y_true=labels, y_pred=preds, rec_residuals=rec_residuals, weight=weights)
            elif METHOD_NAME == "MgRL_CE":
                preds, rec_residuals, contrastive_loss = outputs["pred"], outputs["rec_residuals"], outputs["contrastive_loss"]
                loss = criterion(y_true=labels, y_pred=preds, rec_residuals=rec_residuals, contrastive_loss=contrastive_loss, weight=weights)
            else:
                raise TypeError(METHOD_NAME)
            loss.backward()
            optimizer.step()
            # note the loss of training in one iter
            train_loss_one_epoch.append(loss.item())
            # note the result in one iter
            now_step = last_step + preds.shape[0]
            train_preds_one_epoch[last_step:now_step] = preds[:, 0].detach()
            train_labels_one_epoch[last_step:now_step] = labels[:, 0].detach()
            train_weights_one_epoch[last_step:now_step] = weights[:, 0].detach()
            last_step = now_step
        # -- note the loss and metrics for one epoch of TRAINING
        epoch_metric["train_loss"][epoch] = np.mean(train_loss_one_epoch)
        epoch_metric["train_R2"][epoch] = r2_score(
            y_true=train_labels_one_epoch.cpu().numpy(), y_pred=train_preds_one_epoch.cpu().numpy(), weight=train_weights_one_epoch.cpu().numpy()
        )
        epoch_metric["train_CORR"][epoch] = corr_score(
            y_true=train_labels_one_epoch.cpu().numpy(), y_pred=train_preds_one_epoch.cpu().numpy(), weight=train_weights_one_epoch.cpu().numpy()
        )
        epoch_metric["train_RMSE"][epoch] = rmse_score(
            y_true=train_labels_one_epoch.cpu().numpy(), y_pred=train_preds_one_epoch.cpu().numpy(), weight=train_weights_one_epoch.cpu().numpy()
        )
        epoch_metric["train_MAE"][epoch] = mae_score(
            y_true=train_labels_one_epoch.cpu().numpy(), y_pred=train_preds_one_epoch.cpu().numpy(), weight=train_weights_one_epoch.cpu().numpy()
        )

        # - VALID model
        last_step = 0
        model.eval()
        with torch.no_grad():
            for batch_data in tqdm(valid_loader):
                # move data to device, and change the dtype from double to float32
                mg_features = batch_data["mg_features"]
                labels = batch_data["label"].to(device=device, dtype=torch.float32)
                weights = batch_data["weight"].to(device=device, dtype=torch.float32)
                # forward to compute outputs
                outputs = model(mg_features)
                if METHOD_NAME in ("MgRL", "MgRL_Attention"):
                    preds, rec_residuals = outputs["pred"], outputs["rec_residuals"]
                    loss = criterion(y_true=labels, y_pred=preds, rec_residuals=rec_residuals, weight=weights)
                elif METHOD_NAME == "MgRL_CE":
                    preds, rec_residuals, contrastive_loss = outputs["pred"], outputs["rec_residuals"], outputs["contrastive_loss"]
                    loss = criterion(y_true=labels, y_pred=preds, rec_residuals=rec_residuals, contrastive_loss=contrastive_loss, weight=weights)
                else:
                    raise TypeError(METHOD_NAME)
                # note the loss of valid in one iter
                valid_loss_one_epoch.append(loss.item())
                # doc the result in one iter, no matter what label_len is, just get the last one
                now_step = last_step + preds.shape[0]
                valid_preds_one_epoch[last_step:now_step] = preds[:, 0].detach()
                valid_labels_one_epoch[last_step:now_step] = labels[:, 0].detach()
                valid_weights_one_epoch[last_step:now_step] = weights[:, 0].detach()
                last_step = now_step
        # -- note the loss and all metrics for one epoch of VALID
        epoch_metric["valid_loss"][epoch] = np.mean(valid_loss_one_epoch)
        epoch_metric["valid_R2"][epoch] = r2_score(
            y_true=valid_labels_one_epoch.cpu().numpy(), y_pred=valid_preds_one_epoch.cpu().numpy(), weight=valid_weights_one_epoch.cpu().numpy()
        )
        epoch_metric["valid_CORR"][epoch] = corr_score(
            y_true=valid_labels_one_epoch.cpu().numpy(), y_pred=valid_preds_one_epoch.cpu().numpy(), weight=valid_weights_one_epoch.cpu().numpy()
        )
        epoch_metric["valid_RMSE"][epoch] = rmse_score(
            y_true=valid_labels_one_epoch.cpu().numpy(), y_pred=valid_preds_one_epoch.cpu().numpy(), weight=valid_weights_one_epoch.cpu().numpy()
        )
        epoch_metric["valid_MAE"][epoch] = mae_score(
            y_true=valid_labels_one_epoch.cpu().numpy(), y_pred=valid_preds_one_epoch.cpu().numpy(), weight=valid_weights_one_epoch.cpu().numpy()
        )

        # - save model&model_config and metrics
        torch.save(model, config.MODEL_SAVE_PATH + f"{METHOD_NAME}_model_pytorch_epoch_{epoch}")
        pd.DataFrame(epoch_metric).to_csv(config.MODEL_SAVE_PATH + "model_metric.csv")

        # write metric log
        dt = datetime.now() - t_start
        logging.info(
            f"Epoch {epoch + 1}/{config.EPOCHS}, Duration: {dt}, {['%s:%.4f ' % (key, value[epoch]) for key, value in epoch_metric.items()]}"
        )

    # draw figure of train and valid metrics
    plt.figure(figsize=(15, 6))
    plt.subplot(3, 1, 1)
    plt.plot(epoch_metric["train_loss"], label="train loss", color="g")
    plt.plot(epoch_metric["valid_loss"], label="valid loss", color="b")
    plt.legend()
    plt.subplot(3, 2, 3)
    plt.plot(epoch_metric["train_R2"], label="train R2", color="g")
    plt.plot(epoch_metric["valid_R2"], label="valid R2", color="b")
    plt.legend()
    plt.subplot(3, 2, 4)
    plt.plot(epoch_metric["train_CORR"], label="train CORR", color="g")
    plt.plot(epoch_metric["valid_CORR"], label="valid CORR", color="b")
    plt.legend()
    plt.subplot(3, 2, 5)
    plt.plot(epoch_metric["train_RMSE"], label="train RMSE", color="g")
    plt.plot(epoch_metric["valid_RMSE"], label="valid RMSE", color="b")
    plt.legend()
    plt.subplot(3, 2, 6)
    plt.plot(epoch_metric["train_MAE"], label="train MAE", color="g")
    plt.plot(epoch_metric["valid_MAE"], label="valid MAE", color="b")
    plt.legend()
    plt.savefig(config.SAVE_PATH + "training_steps.png", dpi=200, bbox_inches="tight")
    logging.info("***************** TRAINING OVER ! *****************\n")


def pred_model() -> None:
    """ Test Model. """

    logging.info(f"***************** RUN PRED MODEL: `{METHOD_NAME}`, DATASET: `{args.dataset}` ! *****************")

    # ---- Get the computing device ---- #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"***************** In device {device}   *****************")

    # ---- Make the dataset and dataloader ---- #
    logging.info(f"***************** BEGIN MAKE DATASET & DATALOADER of `{args.dataset}` !! *****************")
    logging.info(f"||| time_steps = {config.TIME_STEPS}, batch size = {config.BATCH_SIZE} |||")
    # make the dataset and dataloader of test
    logging.info(f"**** TEST DATASET & DATALOADER ****")
    if args.dataset == "elect":  # the UCI electricity dataset.
        test_dataset = ELECTDataset(root_path=config.UCI_ELECT_DATASET_PATH, data_type="Test", time_steps=config.TIME_STEPS)
    elif args.dataset == "lob":  # the Future LOB dataset
        test_dataset = LOBDataset(
            root_path=config.LOB_DATASET_PATH, start_date=config.TEST_START_DATE, end_date=config.TEST_END_DATE, need_norm=config.NEED_NORM
        )
    elif args.dataset == "index":  # the CSI300 index dataset
        test_dataset = INDEXDataset(root_path=config.INDEX_DATASET_PATH, data_type="Test", need_norm=config.NEED_NORM)
    else:
        raise TypeError(args.dataset)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    logging.info("***************** DATASET MAKE OVER ! *****************")
    logging.info(f"Test dataset: length = {len(test_dataset)}")

    # ---- Load model and test ---- #
    model, model_path = load_best_model(config.MODEL_SAVE_PATH, METHOD_NAME, config.MAIN_METRIC)
    logging.info(f"***************** LOAD BEST MODEL {model_path} *****************")

    # ---- Test Model ---- #
    labels_array = torch.zeros(len(test_dataset)).to(device=device)
    predictions_array = torch.zeros(len(test_dataset)).to(device=device)
    weight_array = torch.zeros(len(test_dataset)).to(device=device)
    last_step = 0
    model.eval()  # start test
    with torch.no_grad():
        for batch_data in tqdm(test_loader):
            # move data to device, and change the dtype from double to float32
            mg_features = batch_data["mg_features"]
            labels = batch_data["label"].to(device=device, dtype=torch.float32)
            weights = batch_data["weight"].to(device=device, dtype=torch.float32)
            # forward (no mater which model, just get the preds)
            outputs = model(mg_features)
            preds = outputs["pred"]
            # doc the result
            now_step = last_step + preds.shape[0]
            labels_array[last_step:now_step] = labels[:, 0].detach()
            predictions_array[last_step:now_step] = preds[:, 0].detach()
            weight_array[last_step:now_step] = weights[:, 0].detach()
            last_step = now_step

    # ---- Logging the result ---- #
    test_R2 = r2_score(y_true=labels_array.cpu().numpy(), y_pred=predictions_array.cpu().numpy(), weight=weight_array.cpu().numpy())
    test_CORR = corr_score(y_true=labels_array.cpu().numpy(), y_pred=predictions_array.cpu().numpy(), weight=weight_array.cpu().numpy())
    test_RMSE = rmse_score(y_true=labels_array.cpu().numpy(), y_pred=predictions_array.cpu().numpy(), weight=weight_array.cpu().numpy())
    test_MAE = mae_score(y_true=labels_array.cpu().numpy(), y_pred=predictions_array.cpu().numpy(), weight=weight_array.cpu().numpy())
    logging.info(f"******** test_R2 : {test_R2} **********")
    logging.info(f"******** test_CORR : {test_CORR} **********")
    logging.info(f"******** test_RMSE : {test_RMSE} **********")
    logging.info(f"******** test_MAE : {test_MAE} **********")
    logging.info("***************** TEST OVER ! *****************\n")


if __name__ == "__main__":
    # ---- Step 0. Prepare some environments for training and prediction ---- #
    # fix the random seed
    fix_random_seed(seed=config.RANDOM_SEED)
    # build up the save directory
    if not os.path.exists(config.SAVE_PATH):
        os.makedirs(config.SAVE_PATH)
    if not os.path.exists(config.MODEL_SAVE_PATH):
        os.makedirs(config.MODEL_SAVE_PATH)
    # construct the train&valid log file
    logging.basicConfig(filename=config.LOG_FILE, format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

    # ---- Step 1. Train & Valid model ---- #
    train_valid_model()

    # ---- Step 2. Pred Model ---- #
    pred_model()
