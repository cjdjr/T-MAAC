from pickle import NONE
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def multi_head_attention(q, k, v, mask=None):
    # q shape = (B, n_heads, n, key_dim)   : n can be either 1 or N
    # k,v shape = (B, n_heads, N, key_dim)
    # mask.shape = (B, group, N)

    B, n_heads, n, key_dim = q.shape

    # score.shape = (B, n_heads, n, N)
    score = th.matmul(q, k.transpose(2, 3)) / np.sqrt(q.size(-1))

    if mask is not None:
        score += mask[:, None, :, :].expand_as(score)

    shp = [q.size(0), q.size(-2), q.size(1) * q.size(-1)]
    attn = th.matmul(F.softmax(score, dim=3), v).transpose(1, 2)
    return attn.reshape(*shp)


def make_heads(qkv, n_heads):
    shp = (qkv.size(0), qkv.size(1), n_heads, -1)
    return qkv.reshape(*shp).transpose(1, 2)


class EncoderLayer(nn.Module):
    def __init__(self, embedding_dim, n_heads=8):
        super(EncoderLayer, self).__init__()

        self.n_heads = n_heads

        self.Wq = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wk = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.Wv = nn.Linear(embedding_dim, embedding_dim, bias=False)
        self.multi_head_combine = nn.Linear(embedding_dim, embedding_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4), nn.ReLU(),
            nn.Linear(embedding_dim * 4, embedding_dim))
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x, mask=None):
        q = make_heads(self.Wq(x), self.n_heads)
        k = make_heads(self.Wk(x), self.n_heads)
        v = make_heads(self.Wv(x), self.n_heads)
        x = x + self.multi_head_combine(multi_head_attention(q, k, v, mask))
        x = self.norm1(x.view(-1, x.size(-1))).view(*x.size())
        x = x + self.feed_forward(x)
        x = self.norm2(x.view(-1, x.size(-1))).view(*x.size())
        return x


class TransformerCritic(nn.Module):
    # sa_sizes, hidden_dim=32, norm_in=True, attend_heads=1
    def __init__(self, input_shape, args):
        super(TransformerCritic, self).__init__()
        self.hidden_dim = args.hid_size
        # self.out_hidden_dim = args.out_hid_size
        self.attend_heads = args.attend_heads
        assert (self.hidden_dim % self.attend_heads) == 0
        self.n_ = args.agent_num
        self.n_layers = args.n_layers
        self.attend_heads = args.attend_heads
        self.args = args
        self.init_projection_layer = nn.Linear(input_shape, self.hidden_dim)
        self.attn_layers = nn.ModuleList([
            EncoderLayer(embedding_dim=self.hidden_dim,
                         n_heads=self.attend_heads)
            for _ in range(self.n_layers)
        ])
        self.reward_head = nn.Sequential(
            nn.Linear(self.hidden_dim * self.n_, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )
        self.cost_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )
        if args.layernorm:
            self.layernorm = nn.LayerNorm(self.hidden_dim)
        if args.hid_activation == 'relu':
            self.hid_activation = nn.ReLU()
        elif args.hid_activation == 'tanh':
            self.hid_activation = nn.Tanh()

    def forward(self, obs):
        # obs : (b, n, d)
        x = self.init_projection_layer(obs)
        for layer in self.attn_layers:
            x = layer(x)
        pred_r = self.reward_head(
            x.view(-1, self.n_ * self.hidden_dim))    # (b, 1)
        pred_c = self.cost_head(x)  # (b,n,1)
        return pred_r, pred_c
