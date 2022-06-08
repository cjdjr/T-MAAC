import torch as th
import torch.nn as nn
import torch.nn.functional as F
class TransformerCritic(nn.Module):
    def __init__(self, input_shape, output_shape, args, date_emb=False):
        super(TransformerCritic, self).__init__()
        self.args = args
        self.date_emb = date_emb
        # Easiest to reuse hid_size variable
        self.fc1 = nn.Linear(input_shape, args.hid_size)
        if args.layernorm:
            self.layernorm = nn.LayerNorm(args.hid_size)
        self.fc2 = nn.Linear(args.hid_size, output_shape)

        if self.args.hid_activation == 'relu':
            self.hid_activation = nn.ReLU()
        elif self.args.hid_activation == 'tanh':
            self.hid_activation = nn.Tanh()

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.hid_size).zero_()

    def forward(self, inputs, hidden_state):
        x = inputs
        x = self.fc1(x)
        if self.args.layernorm:
            x = self.layernorm(x)
        x = self.hid_activation(x)
        v = self.fc2(x)
        return v, None