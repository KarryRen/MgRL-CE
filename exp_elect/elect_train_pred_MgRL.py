# -*- coding: utf-8 -*-
# @Time    : 2024/4/5 09:33
# @Author  : Karry Ren

""" Training and Prediction code of `MgRLNet` for UCI electricity dataset.

Training, Validation and Prediction be together !
    Here are two functions:
        - `train_valid_model()` => train and valid model, and save trained models of all epochs.
        - `pred_model()` => use the best model to do prediction.

"""

import os
import logging
import torch
import torch.utils.data as data

from utils import fix_random_seed
import elect_config as config
from datasets.elect_dataset import ELECTDataset


def train_valid_model() -> None:
    # ---- Build up the save directory ---- #
    if not os.path.exists(config.SAVE_PATH):
        os.makedirs(config.SAVE_PATH)
    if not os.path.exists(config.MODEL_SAVE_PATH):
        os.makedirs(config.MODEL_SAVE_PATH)

    # ---- Construct the train&valid log file (might be same as prediction) ---- #
    logging.basicConfig(filename=config.LOG_FILE, format="%(asctime)s - %(levelname)s - %(message)s", level=logging.INFO)
    logging.info(f"***************** RUN TRAIN DEEP-LOB ! *****************")

    # ---- Get the device ---- #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"***************** In device {device}   *****************")

    # ---- Make the dataset and dataloader ---- #
    logging.info(f"***************** BEGIN MAKE DATASET & DATALOADER ! *****************")
    logging.info(f"||| time_steps = {config.TIME_STEPS}, batch size = {config.BATCH_SIZE} |||")
    # make the dataset and dataloader of training
    logging.info(f"**** TRAINING DATASET & DATALOADER ****")
    train_dataset = ELECTDataset(root_path=config.UCI_ELECT_DATASET_PATH, data_type="Train", time_steps=config.TIME_STEPS)
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    logging.info(f"**** TRAINING DATASET & DATALOADER ****")
valid_dataset = DeepLOBDataset(data_root_dict=data_root_dict,
                               future_types=config.FUTURE_TYPES,
                               label_len=config.LABEL_LEN,
                               dates=config.VALID_DATES,
                               feature_len=config.FEATURE_LEN,
                               tick_num=config.TICK_NUM,
                               start_tick=config.START_TICK,
                               end_tick=config.END_TICK,
                               label_k=config.LABEL_K,
                               label_offset=config.LABEL_OFFSET,
                               label_way=config.LABEL_WAY,
                               needed_features=config.NEEDED_FEATURES,
                               paf_shift_k=config.PAF_SHIFT_K,
                               use_sample_std_weight=config.DICT.get("valid_use_sample_std_weight", False))
valid_loader = data.DataLoader(dataset=valid_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
logging.info("***************** DATASET MAKE OVER ! *****************")
logging.info(f"Train dataset: length = {len(train_dataset)}")
logging.info(f"Valid dataset: length = {len(valid_dataset)}")
# ---- Construct the model and transfer device, while making loss and optimizer ---- #
# the model
model = DeepLOBNet(feature_encoder_params=config.FEATURE_ENCODER_PARAMS,
                   fusion_way_list=config.FUSION_WAY_LIST,
                   seq_encoder_params=config.SEQ_ENCODER_PARAMS,
                   label_len=config.LABEL_LEN,
                   device=device)
# the loss function
criterion = get_loss(config.DICT.get("loss", {"MTL": {}}))
# the optimizer
if config.OPTIM_NAME == "AdamW":
    optimizer = torch.optim.AdamW(model.parameters(), **config.OPTIM_PARAMS)
elif config.OPTIM_NAME == "Adadelta":
    optimizer = torch.optim.Adadelta(model.parameters(), **config.OPTIM_PARAMS)
else:
    raise ValueError(f"The Type of Optimize {config.OPTIM_NAME} is not allowed now.")

# ---- Train and Valid ---- #
# init the metric dict
df_metric = {
    # train & valid loss
    "train_loss": np.zeros(config.EPOCHS),
    "valid_loss": np.zeros(config.EPOCHS),
    # train & valid r2
    "train_global_r2": np.zeros(config.EPOCHS),
    "valid_global_r2": np.zeros(config.EPOCHS),
    "valid_daily_mid_r2": np.zeros(config.EPOCHS),
    # valid corr
    "valid_global_corr": np.zeros(config.EPOCHS),
    # adding
    "valid_beta": np.zeros(config.EPOCHS),
    "valid_rescaled_global_r2": np.zeros(config.EPOCHS),
    "valid_rescaled_daily_mean_r2": np.zeros(config.EPOCHS),
    "valid_rescaled_global_corr": np.zeros(config.EPOCHS)
}

# train model epoch by epoch
logging.info("***************** BEGIN TRAINING ! *****************")
logging.info(f"||| feature_encoder_params = {config.FEATURE_ENCODER_PARAMS} |||")
logging.info(f"||| fusion_way_list = {config.FUSION_WAY_LIST} |||")
logging.info(f"||| seq_encoder_params = {config.SEQ_ENCODER_PARAMS} |||")
logging.info(f"||| optimizer_type = {config.OPTIM_NAME} |||")
logging.info(f"||| optimizer_params = {config.OPTIM_PARAMS} |||")

# start train and valid during train
for epoch in tqdm(range(config.EPOCHS)):
    # start timer for one epoch
    t_start = datetime.now()

    # set the array for one epoch to store
    train_loss_one_epoch = []
    valid_loss_one_epoch = []
    if config.TRAIN_DATASET_TYPE == "DeepLOBDataset_DailySample":
        real_sample_num = len(train_dataset) * config.TICK_NUM_IN_ONE_SAMPLE * len(config.TRAIN_DATES)
    else:
        real_sample_num = len(train_dataset)
    train_preds_one_epoch = torch.zeros(real_sample_num).to(device=device)
    train_labels_one_epoch = torch.zeros(real_sample_num).to(device=device)
    train_weights_one_epoch = torch.zeros(real_sample_num).to(device=device)
    valid_preds_one_epoch = torch.zeros(len(valid_dataset)).to(device=device)
    valid_labels_one_epoch = torch.zeros(len(valid_dataset)).to(device=device)
    valid_weights_one_epoch = torch.zeros(len(valid_dataset)).to(device=device)

    # train model
    last_step = 0
    model.train()
    for batch_data in tqdm(train_loader):
        # move data to device, and change the dtype from double to float32
        lob_features = batch_data["features"]
        lob_labels = batch_data["label"].to(device=device, dtype=torch.float32)
        lob_weights = batch_data["weight"].to(device=device, dtype=torch.float32)
        # zero_grad, forward, compute loss, backward and optimize
        optimizer.zero_grad()
        outputs = model(lob_features)["label"]
        loss = criterion(outputs, lob_labels, lob_weights)
        loss.backward()
        optimizer.step()
        # note the loss of training in one iter
        train_loss_one_epoch.append(loss.item())
        # doc the result in one iter, no matter what label_len is, just get the last one
        now_step = last_step + outputs.shape[0]
        train_preds_one_epoch[last_step:now_step] = outputs[:, -1, 0].detach()
        train_labels_one_epoch[last_step:now_step] = lob_labels[:, -1, 0].detach()
        train_weights_one_epoch[last_step:now_step] = lob_weights[:, -1, 0].detach()
        last_step = now_step
    # note the loss and r2 for one epoch of TRAINING
    df_metric["train_loss"][epoch] = np.mean(train_loss_one_epoch)
    df_metric["train_global_r2"][epoch] = r2_score(y_true=train_labels_one_epoch.cpu().numpy(),
                                                   y_pred=train_preds_one_epoch.cpu().numpy(),
                                                   weight=train_weights_one_epoch.cpu().numpy())

    # valid model
    last_step = 0
    model.eval()
    with torch.no_grad():
        for batch_data in tqdm(valid_loader):
            # move data to device, and change the dtype from double to float32
            lob_features = batch_data["features"]
            lob_labels = batch_data["label"].to(device=device, dtype=torch.float32)
            lob_weights = batch_data["weight"].to(device=device, dtype=torch.float32)
            # forward to compute outputs
            outputs = model(lob_features)["label"]
            # note the loss of valid in one iter
            loss = criterion(outputs, lob_labels, lob_weights)
            valid_loss_one_epoch.append(loss.item())
            # doc the result in one iter, no matter what label_len is, just get the last one
            now_step = last_step + outputs.shape[0]
            valid_preds_one_epoch[last_step:now_step] = outputs[:, -1, 0].detach()
            valid_labels_one_epoch[last_step:now_step] = lob_labels[:, -1, 0].detach()
            valid_weights_one_epoch[last_step:now_step] = lob_weights[:, -1, 0].detach()
            last_step = now_step

    # note the loss and all metrics for one epoch
    df_metric["valid_loss"][epoch] = np.mean(valid_loss_one_epoch)
    x_dot_x = torch.sum(
        valid_preds_one_epoch.view(-1) * valid_preds_one_epoch.view(-1) * valid_weights_one_epoch.view(-1))
    x_dot_y = torch.sum(
        valid_preds_one_epoch.view(-1) * valid_labels_one_epoch.view(-1) * valid_weights_one_epoch.view(-1))
    valid_beta = (x_dot_y / x_dot_x).cpu().numpy()  # the beta of OLS from y_pred to label
    df_metric["valid_beta"][epoch] = float(valid_beta)
    df_metric["valid_global_r2"][epoch] = r2_score(y_true=valid_labels_one_epoch.cpu().numpy(),
                                                   y_pred=valid_preds_one_epoch.cpu().numpy(),
                                                   weight=valid_weights_one_epoch.cpu().numpy())
    df_metric["valid_global_corr"][epoch] = corr_score(y_true=valid_labels_one_epoch.cpu().numpy(),
                                                       y_pred=valid_preds_one_epoch.cpu().numpy(),
                                                       weight=valid_weights_one_epoch.cpu().numpy())
    df_metric["valid_daily_mid_r2"][epoch] = r2_score(
        y_true=valid_labels_one_epoch.cpu().numpy().reshape([len(config.VALID_DATES), -1]),
        y_pred=valid_preds_one_epoch.cpu().numpy().reshape([len(config.VALID_DATES), -1]),
        weight=valid_weights_one_epoch.cpu().numpy().reshape([len(config.VALID_DATES), -1]),
        mode="daily_mid"
    )
    df_metric["valid_rescaled_global_r2"][epoch] = r2_score(
        y_true=valid_labels_one_epoch.cpu().numpy(),
        y_pred=valid_preds_one_epoch.cpu().numpy() * valid_beta,
        weight=valid_weights_one_epoch.cpu().numpy()
    )
    df_metric["valid_rescaled_global_corr"][epoch] = corr_score(
        y_true=valid_labels_one_epoch.cpu().numpy(),
        y_pred=valid_preds_one_epoch.cpu().numpy() * valid_beta,
        weight=valid_weights_one_epoch.cpu().numpy()
    )
    df_metric["valid_rescaled_daily_mean_r2"][epoch] = r2_score(
        y_true=valid_labels_one_epoch.cpu().numpy().reshape([len(config.VALID_DATES), -1]),
        y_pred=valid_preds_one_epoch.cpu().numpy().reshape([len(config.VALID_DATES), -1]) * valid_beta,
        weight=valid_weights_one_epoch.cpu().numpy().reshape([len(config.VALID_DATES), -1]),
        mode="daily_mean"
    )

    # save model&model_config and metrics
    torch.save(model, config.MODEL_SAVE_PATH + f"model_pytorch_epoch_{epoch}")
    with open(config.MODEL_SAVE_PATH + f"model_config_epoch_{epoch}", "w") as file:
        model_config = {"valid_beta": float(valid_beta)}
        json.dump(model_config, file)  # save the model_config to file
    pd.DataFrame(df_metric).to_csv(config.MODEL_SAVE_PATH + "model_pytorch_metric.csv")

    # write log
    dt = datetime.now() - t_start
    logging.info(f"Epoch {epoch + 1}/{config.EPOCHS}, Duration: {dt}, "
                 f"{['%s:%.4f ' % (key, value[epoch]) for key, value in df_metric.items()]}")

# draw figure of train and valid metrics
plt.figure(figsize=(15, 6))
plt.subplot(3, 1, 1)
plt.plot(df_metric["train_loss"], label="train loss", color="g")
plt.legend()
plt.subplot(3, 1, 2)
plt.plot(df_metric["valid_loss"], label="valid loss", color="b")
plt.legend()
plt.subplot(3, 1, 3)
plt_r2 = [key for key in df_metric.keys() if "_r2" in key]
for k, key in enumerate(plt_r2):
    plt.plot(df_metric[key], label=key, color=config.DICT["color"][k], linestyle="--")
plt.legend()
plt.savefig(config.SAVE_PATH + "training_steps.png", dpi=200, bbox_inches="tight")
logging.info("***************** TRAINING OVER ! *****************")

if __name__ == "__main__":
    # ---- Step 0. Fix the random seed ---- #
    fix_random_seed(seed=config.RANDOM_SEED)

    # ---- Step 1. Train & Valid model ---- #
    train_valid_model()
