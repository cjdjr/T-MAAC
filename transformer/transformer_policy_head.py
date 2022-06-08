import torch.nn as nn

# MLP and GRU after transformer encoder
class TransformerAgent(nn.Module):
    # sa_sizes, hidden_dim=32, norm_in=True, attend_heads=1
    def __init__(self, args):
        super(TransformerAgent, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(args.hid_size, args.hid_size)
        if args.layernorm:
            self.layernorm = nn.LayerNorm(args.hid_size)
        self.rnn = nn.GRUCell(args.hid_size, args.hid_size)
        self.final_output_layer = nn.Linear(args.hid_size, args.action_dim)
        if self.args.hid_activation == 'relu':
            self.hid_activation = nn.ReLU()
        elif self.args.hid_activation == 'tanh':
            self.hid_activation = nn.Tanh()

    def init_hidden(self):
        # make hidden states on same device as model
        return self.fc1.weight.new(1, self.args.agent_num, self.args.hid_size).zero_()
        # return th.zeros(1, self.args.agent_num, self.hidden_dim).cuda()

    def forward(self, inputs, hidden_state):
        x = self.fc1(inputs)
        if self.args.layernorm:
            x = self.layernorm(x)
        x = self.hid_activation(x)
        h_in = hidden_state.reshape(-1, self.args.hid_size)
        h = self.rnn(x, h_in)
        a = self.final_output_layer(h)
        return a, None, h
