import torch as th
import torch.nn as nn
import numpy as np
from utilities.util import select_action
from models.maddpg import MADDPG
# from critics.transformer_critic import TransformerCritic
from critics.transformer_critic_ex import TransformerCritic


class TransMADDPG(MADDPG):
    def __init__(self, args, target_net=None):
        # self.obs_position_list = args.obs_position_list
        # self.predict_dim = self.obs_position_list[:,1]-self.obs_position_list[:,0]
        super(TransMADDPG, self).__init__(args, target_net)

    def construct_value_net(self):
        input_shape = self.obs_dim + self.act_dim
        self.value_dicts = nn.ModuleList( [ TransformerCritic(input_shape, self.args) ] )

    def value(self, obs, act):
        # obs_shape = (b, n, o)
        # act_shape = (b, n, a)
        batch_size = obs.size(0)
        obs_reshape = obs.contiguous()
        act_reshape = act.contiguous()

        inputs = th.cat( (obs_reshape, act_reshape), dim=-1 )   # (b, n, o+a)

        if self.args.shared_params:
            agent_value = self.value_dicts[0]
            values, costs = agent_value(inputs)
            values = values.contiguous().unsqueeze(dim=-1).repeat(1, self.n_, 1).view(batch_size, self.n_, 1)
            costs = costs.contiguous().view(batch_size, self.n_, 1)

        else:
            NotImplementedError()

        return values, costs

    def get_loss(self, batch):
        batch_size = len(batch.state)
        state, actions, old_log_prob_a, old_values, old_next_values, rewards, cost, next_state, done, last_step, actions_avail, last_hids, hids = self.unpack_data(batch)
        _, actions_pol, log_prob_a, action_out, _ = self.get_actions(state, status='train', exploration=False, actions_avail=actions_avail, target=False, last_hid=last_hids)
        if self.args.double_q:
            _, next_actions, _, _, _ = self.get_actions(next_state, status='train', exploration=False, actions_avail=actions_avail, target=False, last_hid=hids)
        else:
            _, next_actions, _, _, _ = self.get_actions(next_state, status='train', exploration=False, actions_avail=actions_avail, target=True, last_hid=hids)
        values_pol = self.value(state, actions_pol)[0].contiguous().view(-1, self.n_)
        values, costs = self.value(state, actions)
        values = values.contiguous().view(-1, self.n_)
        costs = costs.contiguous().view(-1, self.n_)
        next_values = self.target_net.value(next_state, next_actions.detach())[0].contiguous().view(-1, self.n_)
        returns = th.zeros((batch_size, self.n_), dtype=th.float).to(self.device)
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
        if self.args.auxiliary_loss:
            value_loss += (costs - cost).pow(2).mean()
        return policy_loss, value_loss, action_out, None

    # def get_predict_loss(self, obs, actions, next_obs):
    #     embedding = self.value_dicts[0].encoder(obs) # (B,N,H)
    #     pred = []
    #     label = []
    #     for i in range(self.n_):
    #         pred.append(self.value_dicts[0].predict_voltage(embedding[:,i,:].squeeze(dim=1), actions, i))
    #         label.append(next_obs[:,i,self.obs_position_list[i][0]:self.obs_position_list[i][1]].squeeze(dim=1))
    #     pred = th.cat(pred,dim=-1)
    #     label = th.cat(label,dim=-1)
    #     pred_loss = nn.MSELoss()(pred,label)
    #     return pred_loss
        