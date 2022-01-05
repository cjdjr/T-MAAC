import torch as th
import torch.nn as nn
import numpy as np
from models.maddpg import MADDPG

from qpsolvers import solve_qp
from utilities.define import * 

from critics.mlp_critic_two_head import MLPTWOHEADCritic
from critics.mlp_critic import MLPCritic

class CSMADDPG(MADDPG):

    def __init__(self, args, target_net=None):
        super(CSMADDPG, self).__init__(args, target_net)
        self.multiplier = th.nn.Parameter(th.tensor(args.init_lambda,device=self.device))
        self.upper_bound = args.upper_bound

    def construct_value_net(self):
        if self.args.agent_id:
            input_shape = (self.obs_dim + self.act_dim) * self.n_ + self.n_
        else:
            input_shape = (self.obs_dim + self.act_dim) * self.n_
        if self.args.use_date:
            input_shape -= self.args.date_dim * (self.n_ - 1 )

        output_shape = 1
        if self.args.shared_params:
            self.value_dicts = nn.ModuleList( [ MLPCritic(input_shape, output_shape, self.args, self.args.use_date) ] )
            self.cost_dicts = nn.ModuleList( [ MLPCritic(input_shape, output_shape, self.args, self.args.use_date) ] )
        else:
            self.value_dicts = nn.ModuleList( [ MLPCritic(input_shape, output_shape, self.args, self.args.use_date) for _ in range(self.n_) ] )
            self.cost_dicts = nn.ModuleList( [ MLPCritic(input_shape, output_shape, self.args, self.args.use_date) for _ in range(self.n_) ] )

    def value(self, obs, act):
        # obs_shape = (b, n, o)
        # act_shape = (b, n, a)
        batch_size = obs.size(0)
        if self.args.use_date:
            date = obs[:,:,:self.args.date_dim]
            obs = obs[:,:,self.args.date_dim:]
        obs_repeat = obs.unsqueeze(1).repeat(1, self.n_, 1, 1) # shape = (b, n, n, o)
        obs_reshape = obs_repeat.contiguous().view(batch_size, self.n_, -1) # shape = (b, n, n*o)
        if self.args.use_date:
            obs_reshape = th.cat((date, obs_reshape), dim=-1)

        # add agent id
        agent_ids = th.eye(self.n_).unsqueeze(0).repeat(batch_size, 1, 1).to(self.device) # shape = (b, n, n)
        if self.args.agent_id:
            obs_reshape = th.cat( (obs_reshape, agent_ids), dim=-1 ) # shape = (b, n, n*o+n)
        
        # make up inputs
        act_repeat = act.unsqueeze(1).repeat(1, self.n_, 1, 1) # shape = (b, n, n, a)
        act_mask_others = agent_ids.unsqueeze(-1) # shape = (b, n, n, 1)
        act_mask_i = 1. - act_mask_others
        act_i = act_repeat * act_mask_others
        act_others = act_repeat * act_mask_i

        # detach other agents' actions
        act_repeat = act_others.detach() + act_i # shape = (b, n, n, a)

        if self.args.shared_params:
            obs_reshape = obs_reshape.contiguous().view( batch_size*self.n_, -1 ) # shape = (b*n, n*o+n)
            act_reshape = act_repeat.contiguous().view( batch_size*self.n_, -1 ) # shape = (b*n, n*a)
        else:
            obs_reshape = obs_reshape.contiguous().view( batch_size, self.n_, -1 ) # shape = (b, n, n*o+n)
            act_reshape = act_repeat.contiguous().view( batch_size, self.n_, -1 ) # shape = (b, n, n*a)

        inputs = th.cat( (obs_reshape, act_reshape), dim=-1 )

        if self.args.shared_params:
            agent_value = self.value_dicts[0]
            agent_cost = self.cost_dicts[0]
            values, _ = agent_value(inputs, None)
            costs, _ = agent_cost(inputs, None)
            values = values.contiguous().view(batch_size, self.n_, 1)
            costs = costs.contiguous().view(batch_size, self.n_, 1)
        else:
            values = []
            costs = []
            for i, agent_value,agent_cost in enumerate(self.value_dicts, self.cost_dicts):
                value, _ = agent_value(inputs[:, i, :], None)
                cost, _ = agent_cost(inputs[:, i, :], None)
                values.append(value)
                costs.append(cost)
            values = th.stack(values, dim=1)
            costs = th.stack(costs, dim=1)

        return values, costs

    def get_loss(self, batch):
        batch_size = len(batch.state)
        state, actions, old_log_prob_a, old_values, old_next_values, rewards, cost, next_state, done, last_step, actions_avail, last_hids, hids = self.unpack_data(batch)
        _, actions_pol, log_prob_a, action_out, _ = self.get_actions(state, status='train', exploration=False, actions_avail=actions_avail, target=False, last_hid=last_hids)
        if self.args.double_q:
            _, next_actions, _, _, _ = self.get_actions(next_state, status='train', exploration=False, actions_avail=actions_avail, target=False, last_hid=hids)
        else:
            _, next_actions, _, _, _ = self.get_actions(next_state, status='train', exploration=False, actions_avail=actions_avail, target=True, last_hid=hids)
        compose = self.value(state, actions_pol)
        values_pol, costs_pol = compose[0].contiguous().view(-1, self.n_), compose[1].contiguous().view(-1, self.n_)
        compose = self.value(state, actions)
        values, costs = compose[0].contiguous().view(-1, self.n_), compose[1].contiguous().view(-1, self.n_)
        compose = self.target_net.value(next_state, next_actions.detach())
        next_values, next_costs = compose[0].contiguous().view(-1, self.n_), compose[1].contiguous().view(-1, self.n_)
        returns, cost_returns = th.zeros((batch_size, self.n_), dtype=th.float).to(self.device), th.zeros((batch_size, self.n_), dtype=th.float).to(self.device)
        assert values_pol.size() == next_values.size()
        assert returns.size() == values.size()
        done = done.to(self.device)
        returns = rewards - self.multiplier.detach() * costs.detach() + self.args.gamma * (1 - done) * next_values.detach() 
        cost_returns = cost + self.args.cost_gamma * (1-done) * next_costs.detach()
        deltas, cost_deltas = returns - values, cost_returns - costs
        advantages = values_pol
        if self.args.normalize_advantages:
            advantages = self.batchnorm(advantages)
        policy_loss = - advantages
        policy_loss = policy_loss.mean()
        value_loss = deltas.pow(2).mean()
        lambda_loss = - ((cost_returns.detach() - self.upper_bound) * self.multiplier).mean() + cost_deltas.pow(2).mean()
        return policy_loss, value_loss, action_out, lambda_loss

    def reset_multiplier(self):
        self.multiplier = th.nn.Parameter(th.tensor(0.,device=self.device))
