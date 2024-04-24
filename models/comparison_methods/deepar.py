# -*- coding: utf-8 -*-
# @author : RenKai (intern in HIGGS ASSET)
# @time   : 4/24/24 4:00 PM
#
# pylint: disable=no-member

""" The Comparison Methods 4: DeepAR.

Ref. https://arxiv.org/abs/1704.04110

TODO: Ref ERROR, should change

"""

from typing import Dict
import logging
import torch
import torch.nn as nn


class DeepAR_Net(nn.Module):
    """ The 2 Layer DeepAR. hidden_size=64. """

    def __init__(
            self, input_size: int, hidden_size: int = 64, num_layers: int = 2,
            dropout: float = 0.0, device: torch.device = torch.device("cpu")
    ):
        """ The init function of DeepAR Net.

        :param input_size: input size for each time step
        :param hidden_size: hidden size of gru
        :param num_layers: the num of gru layers
        :param dropout: the dropout ratio
        :param device: the computing device

        """

        super(DeepAR_Net, self).__init__()
        self.device = device

        # ---- Log the info of DeepAR ---- #
        logging.info(f"|||| Using DeepAR Now ! input_size={input_size}, hidden_size={hidden_size}, num_layers={num_layers}, dropout_ratio={dropout}||||")

        # ---- Part 1. The LSTM module ---- #
        # define the LSTM
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout, bias=True
        ).to(device=device)
        # initialize LSTM forget gate bias to be 1
        # as recommended by http://proceedings.mlr.press/v37/jozefowicz15.pdf
        for param_name_list in self.lstm._all_weights:  # for loop all weight
            for param_name in filter(lambda p_n: "bias" in p_n, param_name_list):
                bias = getattr(self.lstm, param_name)  # get the bias
                bias_len = bias.shape[0]  # get the bias
                start, end = bias_len // 4, bias_len // 2
                bias.data[start:end].fill_(1.)  # set to 1

        # ---- Part 2. MU computation module ---- #
        self.distribution_mu = nn.Linear(hidden_size, 1).to(device=device)

        # ---- Part 3. SIGMA computation module ---- #
        self.distribution_sigma = nn.Linear(hidden_size, 1).to(device=device)
        self.softplus = nn.Softplus().to(device=device)

    def forward(self, mul_granularity_input: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """ The forward of DeepAR. Predict mu and sigma of the distribution for x_T (last step).

        :param mul_granularity_input: the input multi granularity, a dict with the format:
            {
                "g1": feature_g1,
                "g2": feature_g2,
                ...,
                "gG": feature_gG
            }

        returns: output, a dict with format:
            {
                "mu" : the prediction mu, shape=(bs, 1),
                "sigma" : the prediction sigma, shape=(bs, 1)
            }

        """

        # ---- Step 1. Get the feature ---- #
        # g1 feature (coarsest), shape=(bs, T, K^g1, D)
        feature_g1 = mul_granularity_input["g1"].to(dtype=torch.float32, device=self.device)
        # get the feature shape
        bs, T, K_g1, d = feature_g1.shape[0], feature_g1.shape[1], feature_g1.shape[2], feature_g1.shape[3]

        # ---- Step 2. Preprocess the input for encoding ---- #
        feature_g1 = feature_g1.reshape(bs, T, K_g1 * d)  # reshape, shape=(bs, T, K^g1*D)

        # ---- Step 3. LSTM Encoding and get the hidden_feature ---- #
        # lstm forward
        hidden_g1, _ = self.lstm(feature_g1)  # shape=(bs, num_layers, hidden_size)
        # get the last step hidden g1
        last_step_hidden_g1 = hidden_g1[:, -1, :]  # shape=(bs, hidden_size)

        # ---- Step 4. Compute mu ---- #
        mu = self.distribution_mu(last_step_hidden_g1)

        # ---- Step 5. Compute sigma ---- #
        sigma = self.distribution_sigma(last_step_hidden_g1)
        sigma = self.softplus(sigma)  # softplus to make sure standard deviation is positive

        # ---- Step 6. Return the output ---- #
        output = {"mu": mu, "sigma": sigma}
        return output


if __name__ == "__main__":  # A demo of DeepAR
    bath_size, time_steps, D = 16, 4, 1
    mg_input = {
        "g1": torch.ones((bath_size, time_steps, 1, D)),
        "g2": torch.ones((bath_size, time_steps, 2, D)),
        "g3": torch.ones((bath_size, time_steps, 6, D)),
        "g4": torch.ones((bath_size, time_steps, 24, D)),
        "g5": torch.ones((bath_size, time_steps, 96, D))
    }
    g_dict = {"g1": 1, "g2": 2, "g3": 6, "g4": 24, "g5": 96}
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DeepAR_Net(input_size=1, device=dev)
    out = model(mg_input)
    print(out["mu"].shape)
