# -*- coding: utf-8 -*-
# @time   : 4/21/24 11:35 AM
# @Author  : Karry Ren

""" The Comparison Methods 6: SFM.

Ref. https://github.com/microsoft/qlib/blob/main/qlib/contrib/model/pytorch_sfm.py#L25

"""

from typing import Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import logging


class SFM_Net(nn.Module):
    """ The SFM. hidden_size=64. """

    def __init__(
            self, input_dim: int, output_dim=16, freq_dim: int = 10, hidden_dim: int = 64,
            dropout_W: float = 0.0, dropout_U: float = 0.0, device: torch.device = torch.device("cpu"),
    ):
        """ The init function of SFM_Net.

        :param input_dim: the input dim of each time step feature
        :param output_dim: the final output dim
        :param freq_dim: the frequency dim
        :param hidden_dim: the hidden dim
        :param dropout_W: the dropout ratio of W
        :param dropout_U: the dropout ratio of U
        :param device: the computing device

        """

        super(SFM_Net, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.freq_dim = freq_dim
        self.hidden_dim = hidden_dim

        self.device = device
        # ---- Log the info of Transformer ---- #
        logging.info(
            f"|||| Using SFM Now ! input_dim={input_dim}, output_dim={output_dim}, hidden_dim={hidden_dim},"
            f"freq_dim={freq_dim}, dropout_W={dropout_W}, dropout_U={dropout_U} ||||"
        )

        # ---- States to maintain ---- #
        # states list have 8 items:
        # first 5 are init states of time step 0,
        # end 3 are constants
        self.states = []

        # ---- Part 1. The general settings ---- #
        self.activation = nn.Tanh().to(device=device)
        self.inner_activation = nn.Hardsigmoid().to(device=device)
        self.dropout_W, self.dropout_U = (dropout_W, dropout_U)

        # ---- Part 2. The time operation Blocks of SFM ---- #
        # - part 2.1 the `i` block
        self.W_i = nn.Parameter(init.xavier_uniform_(torch.empty((input_dim, hidden_dim)))).to(device=device)
        self.U_i = nn.Parameter(init.orthogonal_(torch.empty(hidden_dim, hidden_dim))).to(device=device)
        self.b_i = nn.Parameter(torch.zeros(hidden_dim)).to(device=device)
        # - part 2.2 the `ste` block
        self.W_ste = nn.Parameter(init.xavier_uniform_(torch.empty(input_dim, hidden_dim))).to(device=device)
        self.U_ste = nn.Parameter(init.orthogonal_(torch.empty(hidden_dim, hidden_dim))).to(device=device)
        self.b_ste = nn.Parameter(torch.ones(hidden_dim)).to(device=device)
        # - part 2.3 the `fre` block
        self.W_fre = nn.Parameter(init.xavier_uniform_(torch.empty(input_dim, freq_dim))).to(device=device)
        self.U_fre = nn.Parameter(init.orthogonal_(torch.empty(hidden_dim, freq_dim))).to(device=device)
        self.b_fre = nn.Parameter(torch.ones(freq_dim)).to(device=device)
        # - part 2.4 the `c` block
        self.W_c = nn.Parameter(init.xavier_uniform_(torch.empty(input_dim, hidden_dim))).to(device=device)
        self.U_c = nn.Parameter(init.orthogonal_(torch.empty(hidden_dim, hidden_dim))).to(device=device)
        self.b_c = nn.Parameter(torch.zeros(hidden_dim)).to(device=device)
        # - part 2.5 the `0` block
        self.W_o = nn.Parameter(init.xavier_uniform_(torch.empty(input_dim, hidden_dim))).to(device=device)
        self.U_o = nn.Parameter(init.orthogonal_(torch.empty(hidden_dim, hidden_dim))).to(device=device)
        self.b_o = nn.Parameter(torch.zeros(hidden_dim)).to(device=device)
        # - part 2.6 the `a` block
        self.U_a = nn.Parameter(init.orthogonal_(torch.empty(freq_dim, 1))).to(device=device)
        self.b_a = nn.Parameter(torch.zeros(hidden_dim)).to(device=device)
        # - part 2.7 the `p` block
        self.W_p = nn.Parameter(init.xavier_uniform_(torch.empty(hidden_dim, output_dim))).to(device=device)
        self.b_p = nn.Parameter(torch.zeros(output_dim)).to(device=device)

        # ---- Part 4. The output fully connect layer ---- #
        self.fc_out = nn.Linear(self.output_dim, 1).to(device=device)

    def forward(self, mul_granularity_input: Dict[str, torch.Tensor]):
        """ The forward function of Transformer Net.

        :param mul_granularity_input: the input multi granularity, a dict with the format:
            {
                "g1": feature_g1,
                "g2": feature_g2,
                ...,
                "gG": feature_gG
            }

        returns: output, a dict with format:
            {"pred" : the prediction result, shape=(bs, 1)}

        """

        # ---- Step 1. Get the feature ---- #
        # g1 feature (coarsest), shape=(bs, T, K^g1, D)
        feature_g1 = mul_granularity_input["g1"].to(dtype=torch.float32, device=self.device)
        # get the feature shape
        bs, T, K_g1, d = feature_g1.shape[0], feature_g1.shape[1], feature_g1.shape[2], feature_g1.shape[3]

        # ---- Step 2. Preprocess the input for encoding ---- #
        feature_g1 = feature_g1.reshape(bs, T, K_g1 * d)  # reshape, shape=(bs, T, K^g1*D)

        # ---- Step 3. For loop the time_steps to do the SFM encoding ---- #
        for t in range(T):  # for loop the time_steps `T`
            # step 3.1 get the feature at time_step `t`
            x = feature_g1[:, t, :]
            # step 3.2 init the states
            if len(self.states) == 0:  # hasn't initialized the states of sfm
                self.init_states()
            # step 3.3 get the constants
            self.get_constants()
            # step 3.4 parse the states, first 5 are states, end 3 are constants
            p_tm1, h_tm1 = self.states[0], self.states[1]
            S_re_tm1, S_im_tm1, time_tm1 = self.states[2], self.states[3], self.states[4]
            B_U, B_W, frequency = self.states[5], self.states[6], self.states[7]
            # TODO: read the operations carefully
            x_i = torch.matmul(x * B_W[0], self.W_i) + self.b_i
            x_ste = torch.matmul(x * B_W[0], self.W_ste) + self.b_ste
            x_fre = torch.matmul(x * B_W[0], self.W_fre) + self.b_fre
            x_c = torch.matmul(x * B_W[0], self.W_c) + self.b_c
            x_o = torch.matmul(x * B_W[0], self.W_o) + self.b_o
            i = self.inner_activation(x_i + torch.matmul(h_tm1 * B_U[0], self.U_i))
            ste = self.inner_activation(x_ste + torch.matmul(h_tm1 * B_U[0], self.U_ste))
            fre = self.inner_activation(x_fre + torch.matmul(h_tm1 * B_U[0], self.U_fre))
            ste = torch.reshape(ste, (-1, self.hidden_dim, 1))
            fre = torch.reshape(fre, (-1, 1, self.freq_dim))
            f = ste * fre
            c = i * self.activation(x_c + torch.matmul(h_tm1 * B_U[0], self.U_c))
            time = time_tm1 + 1
            omega = torch.tensor(2 * np.pi) * time * frequency
            re = torch.cos(omega)
            im = torch.sin(omega)
            c = torch.reshape(c, (-1, self.hidden_dim, 1))
            S_re = f * S_re_tm1 + c * re
            S_im = f * S_im_tm1 + c * im
            A = torch.square(S_re) + torch.square(S_im)
            A = torch.reshape(A, (-1, self.freq_dim)).float()
            A_a = torch.matmul(A * B_U[0], self.U_a)
            A_a = torch.reshape(A_a, (-1, self.hidden_dim))
            a = self.activation(A_a + self.b_a)
            o = self.inner_activation(x_o + torch.matmul(h_tm1 * B_U[0], self.U_o))
            h = o * a
            p = torch.matmul(h, self.W_p) + self.b_p
            # update the states, make the constants to None
            self.states = [p, h, S_re, S_im, time, None, None, None]

        # ---- Step 4. FC to get the prediction ---- #
        # get the last step hidden states
        last_step_p_g1 = self.states[0]  # shape=(bs, hidden_size)
        # use the last step to predict
        y = self.fc_out(last_step_p_g1)  # shape=(bs, 1)

        # ---- Step 5. Refresh the states ---- #
        self.states = []

        # ---- Step 6. Return the output ---- #
        output = {"pred": y}
        return output

    def init_states(self) -> None:
        """ Init the states of SFM (5 kinds of init states).

        Return None, just update the self.states.

        """

        # ---- Make up the init states ---- #
        reducer_f = torch.zeros((self.hidden_dim, self.freq_dim)).to(self.device)
        reducer_p = torch.zeros((self.hidden_dim, self.output_dim)).to(self.device)

        init_state_h = torch.zeros(self.hidden_dim).to(self.device)
        init_state_p = torch.matmul(init_state_h, reducer_p)

        init_state = torch.zeros_like(init_state_h).to(self.device)
        init_freq = torch.matmul(init_state_h, reducer_f)

        init_state = torch.reshape(init_state, (-1, self.hidden_dim, 1))
        init_freq = torch.reshape(init_freq, (-1, 1, self.freq_dim))

        init_state_S_re = init_state * init_freq
        init_state_S_im = init_state * init_freq

        init_state_time = torch.tensor(0).to(self.device)

        # ---- Summary them to the self.states list ---- #
        self.states = [
            init_state_p, init_state_h, init_state_S_re, init_state_S_im, init_state_time,
            None, None, None,
        ]

    def get_constants(self):
        """ Get 3 constants to states.

        Return None, just update the self.states.

        """

        # ---- Part 1. First 2 constants (states 5, 6) ---- #
        constants = [
            [torch.tensor(1.0).to(self.device) for _ in range(6)],
            [torch.tensor(1.0).to(self.device) for _ in range(7)]
        ]

        # ---- Part 2. Frequency constants (states 7) ---- #
        array = np.array([float(ii) / self.freq_dim for ii in range(self.freq_dim)])
        constants.append(torch.tensor(array).to(self.device))

        # ---- Put the constants to states ---- #
        self.states[5:] = constants


if __name__ == "__main__":  # A demo of SFM
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

    model = SFM_Net(input_dim=1, device=dev)
    out = model(mg_input)
    print(out["pred"].shape)
