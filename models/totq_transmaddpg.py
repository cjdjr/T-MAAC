import torch as th
import torch.nn as nn
import numpy as np
from utilities.util import select_action
from models.trans_maddpg import TransMADDPG
from critics.transformer_critic import TransformerCritic


class TotQTransMADDPG(TransMADDPG):
    def __init__(self, args, target_net=None):
        # self.predict_dim = self.obs_position_list[:,1]-self.obs_position_list[:,0]
        super(TotQTransMADDPG, self).__init__(args, target_net)

    def construct_value_net(self):
        input_shape = (self.args.hid_size) * self.n_
        output_shape = 1
        if self.args.predict_loss:
            self.value_dicts = nn.ModuleList( [ TransformerCritic(self.obs_dim, self.act_dim, input_shape, output_shape, self.args, self.predict_dim ) ] )
        else:
            self.value_dicts = nn.ModuleList( [ TransformerCritic(self.obs_dim, self.act_dim, input_shape, output_shape, self.args) ] )

    def value(self, obs, act):
        # obs_shape = (b, n, o)
        # act_shape = (b, n, a)
        batch_size = obs.size(0)
        obs_reshape = obs
        act_reshape = act
        inputs = th.cat( (obs_reshape, act_reshape), dim=-1 )   # (b, n, o+a)

        if self.args.shared_params:
            agent_value = self.value_dicts[0]
            values, _ = agent_value((inputs,None))
            values = values.contiguous().view(batch_size, 1)

        return values

    def get_loss(self, batch):
        batch_size = len(batch.state)
        state, actions, old_log_prob_a, old_values, old_next_values, rewards, cost, next_state, done, last_step, actions_avail, last_hids, hids = self.unpack_data(batch)
        rewards = rewards.mean(dim=-1,keepdim=True)
        _, actions_pol, log_prob_a, action_out, _ = self.get_actions(state, status='train', exploration=False, actions_avail=actions_avail, target=False, last_hid=last_hids)
        if self.args.double_q:
            _, next_actions, _, _, _ = self.get_actions(next_state, status='train', exploration=False, actions_avail=actions_avail, target=False, last_hid=hids)
        else:
            _, next_actions, _, _, _ = self.get_actions(next_state, status='train', exploration=False, actions_avail=actions_avail, target=True, last_hid=hids)
        values_pol = self.value(state, actions_pol).contiguous().view(-1, 1)
        values = self.value(state, actions).contiguous().view(-1, 1)
        next_values = self.target_net.value(next_state, next_actions.detach()).contiguous().view(-1, 1)
        returns = th.zeros((batch_size, 1), dtype=th.float).to(self.device)
        assert values_pol.size() == next_values.size()
        assert returns.size() == values.size()
        done = done.to(self.device)
        returns = rewards + self.args.gamma * (1 - done) * next_values.detach()
        deltas = returns - values
        advantages = values_pol
        if self.args.normalize_advantages:
            advantages = self.batchnorm(advantages)
        policy_loss = - advantages
        policy_loss = policy_loss.mean()
        value_loss = deltas.pow(2).mean()
        if self.args.predict_loss:
            value_loss = (value_loss, self.get_predict_loss(state, actions, next_state))
        return policy_loss, value_loss, action_out, None

        