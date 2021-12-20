import torch as th
import torch.nn as nn
import numpy as np
from models.maddpg import MADDPG

from qpsolvers import solve_qp
from utilities.define import * 

class CSMADDPG(MADDPG):

    def __init__(self, args, target_net=None):
        super(CSMADDPG, self).__init__(args, target_net)
        self.multiplier = th.nn.Parameter(th.tensor(args.init_lambda,device=self.device))
        self.upper_bound = args.upper_bound
    
    def get_loss(self, batch):
        batch_size = len(batch.state)
        state, actions, old_log_prob_a, old_values, old_next_values, rewards, cost, next_state, done, last_step, actions_avail, last_hids, hids = self.unpack_data(batch)
        _, actions_pol, log_prob_a, action_out, _ = self.get_actions(state, status='train', exploration=False, actions_avail=actions_avail, target=False, last_hid=last_hids)
        if self.args.double_q:
            _, next_actions, _, _, _ = self.get_actions(next_state, status='train', exploration=False, actions_avail=actions_avail, target=False, last_hid=hids)
        else:
            _, next_actions, _, _, _ = self.get_actions(next_state, status='train', exploration=False, actions_avail=actions_avail, target=True, last_hid=hids)
        values_pol = self.value(state, actions_pol).contiguous().view(-1, self.n_)
        values = self.value(state, actions).contiguous().view(-1, self.n_)
        next_values = self.target_net.value(next_state, next_actions.detach()).contiguous().view(-1, self.n_)
        returns = th.zeros((batch_size, self.n_), dtype=th.float).to(self.device)
        assert values_pol.size() == next_values.size()
        assert returns.size() == values.size()
        done = done.to(self.device)
        returns = rewards + self.args.gamma * (1 - done) * next_values.detach() - self.multiplier.detach() * (cost - self.upper_bound)
        deltas = returns - values
        advantages = values_pol
        if self.args.normalize_advantages:
            advantages = self.batchnorm(advantages)
        policy_loss = - advantages
        policy_loss = policy_loss.mean()
        value_loss = deltas.pow(2).mean()
        lambda_loss = - ((cost - self.upper_bound) * self.multiplier).mean()
        return policy_loss, value_loss, action_out, lambda_loss

    def reset_multiplier(self):
        self.multiplier = th.nn.Parameter(th.tensor(0.,device=self.device))
