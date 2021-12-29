import torch as th
import torch.nn as nn
import numpy as np
from utilities.util import select_action
from models.maddpg import MADDPG
from critics.transformer_critic import TransformerCritic


class TransMADDPG(MADDPG):
    def __init__(self, args, target_net=None):
        self.obs_position_list = args.obs_position_list
        # self.predict_dim = self.obs_position_list[:,1]-self.obs_position_list[:,0]
        super(TransMADDPG, self).__init__(args, target_net)

    def construct_value_net(self):
        if self.args.agent_id:
            input_shape = (self.args.hid_size) * self.n_ + self.n_
        else:
            input_shape = (self.args.hid_size) * self.n_
        if self.args.use_date:
            input_shape -= self.args.date_dim * (self.n_ - 1 )
        output_shape = 1
        if self.args.predict_loss:
            self.value_dicts = nn.ModuleList( [ TransformerCritic(self.obs_dim, self.act_dim, input_shape, output_shape, self.args, self.predict_dim ) ] )
        else:
            self.value_dicts = nn.ModuleList( [ TransformerCritic(self.obs_dim, self.act_dim, input_shape, output_shape, self.args) ] )

    def value(self, obs, act):
        # obs_shape = (b, n, o)
        # act_shape = (b, n, a)
        batch_size = obs.size(0)
        if self.args.use_date:
            date = obs[:,:,:self.args.date_dim]
            obs = obs[:,:,self.args.date_dim:]
        obs_repeat = obs.unsqueeze(1).repeat(1, self.n_, 1, 1) # shape = (b, n, n, o)
        obs_reshape = obs_repeat.contiguous()
        if self.args.use_date:
            obs_reshape = th.cat((date, obs_reshape), dim=-1)

        # add agent id
        agent_ids = th.eye(self.n_).unsqueeze(0).repeat(batch_size, 1, 1).to(self.device) # shape = (b, n, n)

        # make up inputs
        act_repeat = act.unsqueeze(1).repeat(1, self.n_, 1, 1) # shape = (b, n, n, a)
        act_mask_others = agent_ids.unsqueeze(-1) # shape = (b, n, n, 1)
        act_mask_i = 1. - act_mask_others
        act_i = act_repeat * act_mask_others
        act_others = act_repeat * act_mask_i

        # detach other agents' actions
        act_repeat = act_others.detach() + act_i # shape = (b, n, n, a)

        if self.args.shared_params:
            obs_reshape = obs_reshape.contiguous().view( batch_size*self.n_, self.n_, -1 ) # shape = (b*n, n, o)
            act_reshape = act_repeat.contiguous().view( batch_size*self.n_, self.n_, -1) # shape = (b*n, n, a)
        else:
            obs_reshape = obs_reshape.contiguous().view( batch_size, self.n_, -1 ) # shape = (b, n, n*o+n)
            act_reshape = act_repeat.contiguous().view( batch_size, self.n_, -1 ) # shape = (b, n, n*a)

        inputs = th.cat( (obs_reshape, act_reshape), dim=-1 )   # (b*n, n, o+a)

        if self.args.shared_params:
            agent_value = self.value_dicts[0]
            values, _ = agent_value((inputs,agent_ids.view(batch_size*self.n_,-1)))
            values = values.contiguous().view(batch_size, self.n_, 1)
        else:
            values = []
            for i, agent_value in enumerate(self.value_dicts):
                value, _ = agent_value(inputs[:, i, :], None)
                values.append(value)
            values = th.stack(values, dim=1)

        return values

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

    def get_predict_loss(self, obs, actions, next_obs):
        embedding = self.value_dicts[0].encoder(obs) # (B,N,H)
        pred = []
        label = []
        for i in range(self.n_):
            pred.append(self.value_dicts[0].predict_voltage(embedding[:,i,:].squeeze(dim=1), actions, i))
            label.append(next_obs[:,i,self.obs_position_list[i][0]:self.obs_position_list[i][1]].squeeze(dim=1))
        pred = th.cat(pred,dim=-1)
        label = th.cat(label,dim=-1)
        pred_loss = nn.MSELoss()(pred,label)
        return pred_loss
        