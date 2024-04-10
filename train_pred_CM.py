# -*- coding: utf-8 -*-
# @Time    : 2024/4/6 14:50
# @Author  : Karry Ren

""" Training and Prediction code of the following comparison methods for 3 datasets:
    - 2 Layer GRU

Training, Validation and Prediction be together !
    Here are two functions:
        - `train_valid_model()` => train and valid model, and save trained models of all epochs.
        - `pred_model()` => use the best model to do prediction (test model).

You can run this python script by `python3 train_pred_CM.py --dataset dataset_name --method method_name`

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

from utils import fix_random_seed, load_best_model
from mg_datasets.elect_dataset import ELECTDataset
from models.comparison_methods.gru import GRUNet
from models.loss import MSE_Loss
from models.metrics import r2_score, corr_score, rmse_score, mae_score

# ---- Init the args parser ---- #
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="elect", help="the dataset name")
parser.add_argument("--method", type=str, default="gru", help="the comparison method name")
args = parser.parse_args()

# ---- Based on the args adjust the settings ---- #
# dataset name
if args.dataset == "elect":  # The UCI electricity dataset.
    import configs.elect_config as config
else:
    raise TypeError(args.dataset)
# method name
METHOD_NAME = args.method


def train_valid_model() -> None:
    """ Train & Valid Model. """

    # ---- Build up the save directory ---- #
    if not os.path.exists(config.SAVE_PATH):
        os.makedirs(config.SAVE_PATH)
    if not os.path.exists(config.MODEL_SAVE_PATH):
        os.makedirs(config.MODEL_SAVE_PATH)

    # ---- Construct the train&valid log file (might be same as prediction) ---- #
    logging.basicConfig(filename=config.LOG_FILE, format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
    logging.info(f"***************** RUN TRAIN&VALID `{METHOD_NAME}` ! *****************")

    # ---- Get the device ---- #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"***************** In device {device}   *****************")

    # ---- Make the dataset and dataloader ---- #
    logging.info(f"***************** BEGIN MAKE DATASET & DATALOADER of `{args.dataset}` ! *****************")
    logging.info(f"||| time_steps = {config.TIME_STEPS}, batch size = {config.BATCH_SIZE} |||")
    # make the dataset and dataloader of training
    logging.info(f"**** TRAINING DATASET & DATALOADER ****")
    if args.dataset == "elect":  # The UCI electricity dataset.
        train_dataset = ELECTDataset(root_path=config.UCI_ELECT_DATASET_PATH, data_type="Train", time_steps=config.TIME_STEPS)
    else:
        raise TypeError(args.dataset)
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)  # the train dataloader
    logging.info(f"**** VALID DATASET & DATALOADER ****")
    if args.dataset == "elect":  # The UCI electricity dataset.
        valid_dataset = ELECTDataset(root_path=config.UCI_ELECT_DATASET_PATH, data_type="Valid", time_steps=config.TIME_STEPS)
    else:
        raise TypeError(args.dataset)
    valid_loader = data.DataLoader(dataset=valid_dataset, batch_size=config.BATCH_SIZE, shuffle=False)  # the valid dataloader
    logging.info("***************** DATASET MAKE OVER ! *****************")
    logging.info(f"Train dataset: length = {len(train_dataset)}")
    logging.info(f"Valid dataset: length = {len(valid_dataset)}")

    # ---- Construct the model and transfer device, while making loss and optimizer ---- #
    # the model
    if METHOD_NAME == "gru":
        model = GRUNet(input_size=config.INPUT_SIZE, device=device)
    else:
        raise TypeError(METHOD_NAME)
    # the loss function
    criterion = MSE_Loss(reduction=config.LOSS_REDUCTION)
    # the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR, weight_decay=config.LAMBDA_THETA)

    # ---- Start Train and Valid ---- #
    # init the metric dict of all epochs
    epoch_metric = {
        # train & valid loss
        "train_loss": np.zeros(config.EPOCHS),
        "valid_loss": np.zeros(config.EPOCHS),
        # train & valid r2
        "train_R2": np.zeros(config.EPOCHS),
        "valid_R2": np.zeros(config.EPOCHS),
        # train & valid CORR
        "train_CORR": np.zeros(config.EPOCHS),
        "valid_CORR": np.zeros(config.EPOCHS),
        # train & valid RMSE
        "train_RMSE": np.zeros(config.EPOCHS),
        "valid_RMSE": np.zeros(config.EPOCHS),
        # train & valid MAE
        "train_MAE": np.zeros(config.EPOCHS),
        "valid_MAE": np.zeros(config.EPOCHS)
    }
    # train model epoch by epoch
    logging.info(f"***************** BEGIN TRAINING `{METHOD_NAME}` ! *****************")
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
        # - train model
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
            preds = outputs["pred"]
            loss = criterion(y_true=labels, y_pred=preds, weight=weights)
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
        # note the loss and metrics for one epoch of TRAINING
        epoch_metric["train_loss"][epoch] = np.mean(train_loss_one_epoch)
        epoch_metric["train_R2"][epoch] = r2_score(y_true=train_labels_one_epoch.cpu().numpy(),
                                                   y_pred=train_preds_one_epoch.cpu().numpy(),
                                                   weight=train_weights_one_epoch.cpu().numpy())
        epoch_metric["train_CORR"][epoch] = corr_score(y_true=train_labels_one_epoch.cpu().numpy(),
                                                       y_pred=train_preds_one_epoch.cpu().numpy(),
                                                       weight=train_weights_one_epoch.cpu().numpy())
        epoch_metric["train_RMSE"][epoch] = rmse_score(y_true=train_labels_one_epoch.cpu().numpy(),
                                                       y_pred=train_preds_one_epoch.cpu().numpy(),
                                                       weight=train_weights_one_epoch.cpu().numpy())
        epoch_metric["train_MAE"][epoch] = mae_score(y_true=train_labels_one_epoch.cpu().numpy(),
                                                     y_pred=train_preds_one_epoch.cpu().numpy(),
                                                     weight=train_weights_one_epoch.cpu().numpy())
        # - valid model
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
                preds = outputs["pred"]
                loss = criterion(y_true=labels, y_pred=preds, weight=weights)
                # note the loss of valid in one iter
                valid_loss_one_epoch.append(loss.item())
                # doc the result in one iter, no matter what label_len is, just get the last one
                now_step = last_step + preds.shape[0]
                valid_preds_one_epoch[last_step:now_step] = preds[:, 0].detach()
                valid_labels_one_epoch[last_step:now_step] = labels[:, 0].detach()
                valid_weights_one_epoch[last_step:now_step] = weights[:, 0].detach()
                last_step = now_step
        # note the loss and all metrics for one epoch of VALID
        epoch_metric["valid_loss"][epoch] = np.mean(valid_loss_one_epoch)
        epoch_metric["valid_R2"][epoch] = r2_score(y_true=valid_labels_one_epoch.cpu().numpy(),
                                                   y_pred=valid_preds_one_epoch.cpu().numpy(),
                                                   weight=valid_weights_one_epoch.cpu().numpy())
        epoch_metric["valid_CORR"][epoch] = corr_score(y_true=valid_labels_one_epoch.cpu().numpy(),
                                                       y_pred=valid_preds_one_epoch.cpu().numpy(),
                                                       weight=valid_weights_one_epoch.cpu().numpy())
        epoch_metric["valid_RMSE"][epoch] = rmse_score(y_true=valid_labels_one_epoch.cpu().numpy(),
                                                       y_pred=valid_preds_one_epoch.cpu().numpy(),
                                                       weight=valid_weights_one_epoch.cpu().numpy())
        epoch_metric["valid_MAE"][epoch] = mae_score(y_true=valid_labels_one_epoch.cpu().numpy(),
                                                     y_pred=valid_preds_one_epoch.cpu().numpy(),
                                                     weight=valid_weights_one_epoch.cpu().numpy())

        # save model&model_config and metrics
        torch.save(model, config.MODEL_SAVE_PATH + f"{METHOD_NAME}_model_pytorch_epoch_{epoch}")
        pd.DataFrame(epoch_metric).to_csv(config.MODEL_SAVE_PATH + "model_metric.csv")

        # write metric log
        dt = datetime.now() - t_start
        logging.info(f"Epoch {epoch + 1}/{config.EPOCHS}, Duration: {dt}, "
                     f"{['%s:%.4f ' % (key, value[epoch]) for key, value in epoch_metric.items()]}")
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
    plt.savefig(f"{config.SAVE_PATH}training_steps.png", dpi=200, bbox_inches="tight")
    logging.info("***************** TRAINING OVER ! *****************")


def pred_model(verbose: bool = False) -> None:
    """ Test Model.

    :param verbose: show image or not

    """

    # ---- Construct the test log file (might be same with train&valid) ---- #
    logging.basicConfig(filename=config.LOG_FILE, format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)

    # ---- Get the computing device ---- #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"***************** RUN PRED `{METHOD_NAME}` ! *****************")
    logging.info(f"***************** In device {device}   *****************")

    # ---- Make the dataset and dataloader ---- #
    logging.info(f"***************** BEGIN MAKE DATASET & DATALOADER of `{args.dataset}` !! *****************")
    logging.info(f"||| time_steps = {config.TIME_STEPS}, batch size = {config.BATCH_SIZE} |||")
    # make the dataset and dataloader of test
    logging.info(f"**** TEST DATASET & DATALOADER ****")
    if args.dataset == "elect":  # The UCI electricity dataset.
        test_dataset = ELECTDataset(root_path=config.UCI_ELECT_DATASET_PATH, data_type="Test", time_steps=config.TIME_STEPS)
    else:
        raise TypeError(args.dataset)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    logging.info("***************** DATASET MAKE OVER ! *****************")
    logging.info(f"Test dataset: length = {len(test_dataset)}")

    # ---- Load model and test ---- #
    model, model_path = load_best_model(config.MODEL_SAVE_PATH, METHOD_NAME, config.MAIN_METRIC)
    logging.info(f"***************** LOAD Best Model {model_path} *****************")

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
            # forward
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

    # ---- Plt the pred ---- #
    if verbose:
        # build up the images save directory
        if not os.path.exists(config.IMAGE_SAVE_PATH):
            os.makedirs(config.IMAGE_SAVE_PATH)
        client_num, day_num = test_dataset.total_client_nums, test_dataset.total_day_nums  # get the client num & day_num
        scale_adj_df = test_dataset.elect_data_scale_adj_df  # get the scale adjustment dataframe
        client_labels_array = labels_array.cpu().numpy().reshape(client_num, day_num)  # shape=(320, day_num)
        client_predictions_array = predictions_array.cpu().numpy().reshape(client_num, day_num)  # shape=(320, day_num)
        client_weights_array = weight_array.cpu().numpy().reshape(client_num, day_num)  # shape=(320, day_num)
        for client_idx, client in enumerate(test_dataset.client_list):
            scale_adj_array = np.repeat(scale_adj_df[client].values, np.sum(client_weights_array[client_idx]))  # (day_num)
            plt.figure(figsize=(15, 6))
            plt.plot(client_labels_array[client_idx][client_weights_array[client_idx] == 1] * scale_adj_array, label="label", color="g")
            plt.plot(client_predictions_array[client_idx][client_weights_array[client_idx] == 1] * scale_adj_array, label="pred", color="b")
            plt.legend()
            plt.savefig(f"{config.IMAGE_SAVE_PATH}{client}.png", dpi=200, bbox_inches="tight")
            print(f"|| Plot Prediction {client_idx}: {client} !! ||")
            break

    # ---- Finish Pred ---- #
    logging.info("***************** TEST OVER ! *****************")
    logging.info("")


if __name__ == "__main__":
    # ---- Step 0. Fix the random seed ---- #
    fix_random_seed(seed=config.RANDOM_SEED)

    # ---- Step 1. Train & Valid model ---- #
    train_valid_model()

    # ---- Step 2. Pred Model ---- #
    pred_model(verbose=False)
