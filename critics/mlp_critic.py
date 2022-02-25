import torch as th
import torch.nn as nn
import torch.nn.functional as F
MONTH_EMB = 8
# DAY_EMB = 4
WEEKDAY_EMB = 8
# HOUR_EMB = 4
# MINUTE_EMB = 4


class MLPCritic(nn.Module):
    def __init__(self, input_shape, output_shape, args, date_emb=False):
        super(MLPCritic, self).__init__()
        self.args = args
        self.date_emb = date_emb
        # Easiest to reuse hid_size variable
        if self.date_emb:
            self.fc1 = nn.Linear(
                input_shape - 2 + MONTH_EMB + WEEKDAY_EMB, args.hid_size)
        else:
            self.fc1 = nn.Linear(input_shape, args.hid_size)
        if args.layernorm:
            self.layernorm = nn.LayerNorm(args.hid_size)

        self.fc2 = nn.Linear(args.hid_size, args.hid_size)
        self.fc3 = nn.Linear(args.hid_size, output_shape)
        if self.date_emb:
            self.month_embed_layer = nn.Embedding(12+1, MONTH_EMB)
            # self.day_embed_layer = nn.Embedding(31+1, DAY_EMB)
            self.weekday_embed_layer = nn.Embedding(7, WEEKDAY_EMB)
            # self.hour_embed_layer = nn.Embedding(24, HOUR_EMB)
            # self.minute_embed_layer = nn.Embedding(60, MINUTE_EMB)

        if self.args.hid_activation == 'relu':
            self.hid_activation = nn.ReLU()
        elif self.args.hid_activation == 'tanh':
            self.hid_activation = nn.Tanh()

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.hid_size).zero_()

    def forward(self, inputs, hidden_state=None):
        if self.date_emb:
            month_embedding = self.month_embed_layer(inputs[:, 0].long())
            # day_embedding = self.day_embed_layer(inputs[:,1].long())
            weekday_embedding = self.weekday_embed_layer(inputs[:, 1].long())
            # hour_embedding = self.hour_embed_layer(inputs[:,3].long())
            # minute_embedding = self.minute_embed_layer(inputs[:,4].long())
            dense_input = inputs[:, self.args.date_dim:]
            x = th.cat([dense_input, month_embedding,
                       weekday_embedding], dim=-1)
        else:
            x = inputs
        x = self.fc1(x)
        if self.args.layernorm:
            x = self.layernorm(x)
        x = self.hid_activation(x)
        h = self.hid_activation(self.fc2(x))
        v = self.fc3(h)
        return v, h
