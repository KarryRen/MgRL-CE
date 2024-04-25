# -*- coding: utf-8 -*-
# @Time    : 2024/4/6 14:50
# @Author  : Karry Ren

""" Training and Prediction code of the following comparison methods for 3 datasets:
        - GRU
        - LSTM
        - Transformer
        - DeepAR
        - SFM
        - ALSTM
        - ADV-ALSTM
        - Fine-Grained GRU
        - Multi-Grained GRU
    Also for some ablation methods for 3 dataset:
        - Mg_Add
        - Mg_Cat

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
from mg_datasets.lob_dataset import LOBDataset
from mg_datasets.index_dataset import INDEXDataset
from models.comparison_methods.gru import GRU_Net, Multi_Grained_GRU_Net
from models.comparison_methods.lstm import LSTM_Net
from models.comparison_methods.transformer import Transformer_Net
from models.comparison_methods.deepar import DeepAR_Net
from models.comparison_methods.sfm import SFM_Net
from models.comparison_methods.alstm import ALSTM_Net
from models.comparison_methods.adv_alstm import ALSTM_Net as ADV_ALSTM_Net
from models.ablation_methods.mg_add import Mg_Add_Net
from models.ablation_methods.mg_cat import Mg_Cat_Net
from models.loss import MSE_Loss, DeepAR_Loss
from models.metrics import r2_score, corr_score, rmse_score, mae_score

# ---- Init the args parser ---- #
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset", type=str, default="elect",
    help="The dataset name. You have only 3 choices: `elect`, `lob`, `index`."
)
parser.add_argument(
    "--method", type=str, default="gru",
    help="The dataset name. You have 8 choices now: \n"
         "- `gru` for the Comparison Methods 1&9: GRU, \n"
         "- `lstm` for the Comparison Methods 2: LSTM, \n"
         "- `transformer` for the Comparison Methods 3: Transformer.\n"
         "- `deepar` for the Comparison Methods 4: DeepAR.\n"
         "- `sfm` for the Comparison Methods 6: SFM. \n"
         "- `alstm` for the Comparison Methods 7: ALSTM.\n"
         "- `adv-alstm` for the Comparison Methods 8: ADV-ALSTM.\n"
         "- `mg-gru` for the Comparison Methods 10: Multi-Grained GRU.\n"
         "- `Mg_Add` for the Ablation Method 1: Mg_Add. \n"
         "- `Mg_Cat` for the Ablation Method 2: Mg_Cat."
)
args = parser.parse_args()

# ---- Based on the args adjust the settings ---- #
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


def train_valid_model() -> None:
    """ Train & Valid Model. """

    logging.info(f"***************** RUN TRAIN&VALID MODEL: `{METHOD_NAME}`, DATASET: `{args.dataset}` ! *****************")

    # ---- Get the device ---- #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"***************** In device {device}   *****************")

    # ---- Make the dataset and dataloader ---- #
    logging.info(f"***************** BEGIN MAKE DATASET & DATALOADER of `{args.dataset}` ! *****************")
    logging.info(f"||| time_steps = {config.TIME_STEPS}, batch size = {config.BATCH_SIZE} |||")
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
    # the model
    if METHOD_NAME == "gru":
        model = GRU_Net(
            input_size=config.INPUT_SIZE, hidden_size=config.ENCODING_HIDDEN_SIZE,
            dropout=config.DROPOUT_RATIO, device=device, use_g=config.USE_G
        )
    elif METHOD_NAME == "lstm":
        model = LSTM_Net(input_size=config.INPUT_SIZE, hidden_size=config.ENCODING_HIDDEN_SIZE, dropout=config.DROPOUT_RATIO, device=device)
    elif METHOD_NAME == "transformer":
        model = Transformer_Net(d_feat=config.INPUT_SIZE, d_model=config.ENCODING_HIDDEN_SIZE, dropout=config.DROPOUT_RATIO, device=device)
    elif METHOD_NAME == "deepar":
        model = DeepAR_Net(input_size=config.INPUT_SIZE, hidden_size=config.ENCODING_HIDDEN_SIZE, dropout=config.DROPOUT_RATIO, device=device)
    elif METHOD_NAME == "sfm":
        model = SFM_Net(
            input_dim=config.INPUT_SIZE, hidden_dim=config.ENCODING_HIDDEN_SIZE,
            dropout_U=config.DROPOUT_RATIO, dropout_W=config.DROPOUT_RATIO, device=device
        )
    elif METHOD_NAME in "alstm":
        model = ALSTM_Net(input_size=config.INPUT_SIZE, hidden_size=config.ENCODING_HIDDEN_SIZE, dropout=config.DROPOUT_RATIO, device=device)
    elif METHOD_NAME in "adv-alstm":
        model = ADV_ALSTM_Net(input_size=config.INPUT_SIZE, hidden_size=config.ENCODING_HIDDEN_SIZE, dropout=config.DROPOUT_RATIO, device=device)
    elif METHOD_NAME in "mg-gru":
        model = Multi_Grained_GRU_Net(
            input_size=config.GRANULARITY_DICT["g1"] * config.FEATURE_DIM + config.GRANULARITY_DICT["g5"] * config.FEATURE_DIM,
            hidden_size=config.ENCODING_HIDDEN_SIZE, dropout=config.DROPOUT_RATIO, device=device
        )
    elif METHOD_NAME in "Mg_Add":
        model = Mg_Add_Net(
            granularity_dict=config.GRANULARITY_DICT, ga_K=config.GA_K,
            encoding_input_size=config.ENCODING_INPUT_SIZE, encoding_hidden_size=config.ENCODING_HIDDEN_SIZE,
            device=device
        )
    elif METHOD_NAME in "Mg_Cat":
        model = Mg_Cat_Net(
            granularity_dict=config.GRANULARITY_DICT, ga_K=config.GA_K,
            encoding_input_size=config.ENCODING_INPUT_SIZE, encoding_hidden_size=config.ENCODING_HIDDEN_SIZE,
            device=device
        )
    else:
        raise TypeError(METHOD_NAME)
    # the loss function
    if METHOD_NAME == "deepar":
        criterion = DeepAR_Loss(reduction=config.LOSS_REDUCTION)
    else:
        criterion = MSE_Loss(reduction=config.LOSS_REDUCTION)
    # the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR, weight_decay=config.LAMBDA_THETA)

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
            # zero_grad, forward, compute loss, backward and optimize, different model have different loss
            optimizer.zero_grad()
            outputs = model(mg_features)
            if METHOD_NAME == "deepar":
                preds = outputs["mu"]  # for deepar the `mu` is pred
                loss = criterion(y_true=labels, mu=outputs["mu"], sigma=outputs["sigma"], weight=weights)
            elif METHOD_NAME == "adv-alstm":
                preds, e = outputs["pred"], outputs["e"]  # for adv-alstm should get e
                preds_adv = model.get_adv(e=e, y_true=labels, weight=weights, criterion=criterion)  # get adv
                loss_raw = criterion(y_true=labels, y_pred=preds, weight=weights)  # loss for raw prediction
                loss_adv = criterion(y_true=labels, y_pred=preds_adv, weight=weights)  # loss for adv prediction
                loss = loss_raw + loss_adv  # final loss
            else:
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
        # - valid model
        last_step = 0
        model.eval()
        with torch.no_grad():
            for batch_data in tqdm(valid_loader):
                # move data to device, and change the dtype from double to float32
                mg_features = batch_data["mg_features"]
                labels = batch_data["label"].to(device=device, dtype=torch.float32)
                weights = batch_data["weight"].to(device=device, dtype=torch.float32)
                # forward to compute outputs, different model have different loss
                outputs = model(mg_features)
                if METHOD_NAME == "deepar":
                    preds = outputs["mu"]
                    loss = criterion(y_true=labels, mu=outputs["mu"], sigma=outputs["sigma"], weight=weights)
                else:  # during valid and test the adv-lstm has no differences
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


def pred_model() -> None:
    """ Test Model. """

    logging.info(f"***************** RUN PRED MODEL: `{METHOD_NAME}`, DATASET: `{args.dataset}` ! *****************")

    # ---- Get the computing device ---- #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"***************** RUN PRED `{METHOD_NAME}` ! *****************")
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
            # forward to compute outputs, different model have different loss
            outputs = model(mg_features)
            if METHOD_NAME == "deepar":
                preds = outputs["mu"]
            else:
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

    # ---- Finish Pred ---- #
    logging.info("***************** TEST OVER ! *****************")
    logging.info("")


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
