# -*- coding: utf-8 -*-
# @author : RenKai (intern in HIGGS ASSET)
# @time   : 4/24/24 7:15 PM
#
# pylint: disable=no-member

""""""


class ALSTM(nn.Module):

    def __init__(self, T, D, E, U, Q, num_lstmlayer):
        super(ALSTM, self).__init__()

        self.D = D
        self.T = T
        self.E = E
        self.U = U
        self.Q = Q
        self.num_lstmlayer = num_lstmlayer
        self.mapping_layer = nn.Linear(D, E, bias=True)
        self.LSTM_layer = nn.LSTM(E, U, num_lstmlayer)
        self.layer = nn.Linear(U, Q, bias=True)
        self.layer_2 = nn.Linear(Q, 1, bias=False)
        self.tanh = nn.tanh()

    def forward(self, origin_input):
        batch_size = origin_input.shape[0]
        origin_input = origin_input.flatten(start_dim=0, end_dim=1)
        mapping_feature = self.tanh(mapping_layer(origin_input))
        mapping_feature = mapping_feature.reshape(batch_size, T, -1)
        mapping_feature = mapping_feature.transpose(0, 1)
        # LSTM Embedding
        h_0 = torch.rand(self.num_lstmlayer, batch_size, self.U)
        c_0 = torch.randn(self.num_lstmlayer, batch_size, self.U)
        output, (hn, cn) = self.LSTM_layer(mapping_feature, (h_0, c_0))
        output = output.transpose(0, 1)
        hn = hn.squeeze(0)  # now turn to batchsize * U
        # Temporal attention layer
        h_s = output.flatten(start_dim=0, end_dim=1)
        x = self.layer(h_s)
        x = self.tanh(x)
        x = self.layer_2(x).squeeze(1)
        x = x.reshape(batch_size, self.T)
        weighted_vector = torch.exp(x) / torch.sum(torch.exp(x), dim=1).unsqueeze(1)  # weight of each time stamp
        # feature aggreagation and final output
        out = output.transpose(1, 2)
        weighted_vector = weight_vector.unsqueeze(2)
        a_s = torch.bmm(out, weighted_vector).squeeze(2)
        h_T = output[:, -1, :]  # last output of the hidden layers of the LSTM block
        es = torch.cat([a_s, h_T], dim=1)
        return es
