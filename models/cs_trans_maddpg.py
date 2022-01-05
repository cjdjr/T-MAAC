from numpy.core.fromnumeric import repeat
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utilities.util import select_action
from models.model import Model
from critics.mlp_critic import MLPCritic
from critics.transformer_critic import TransformerCritic
from critics.transformer_encoder_ex import TransformerEncoder


class CSTRANSMADDPG(Model):
    def __init__(self, args, target_net=None):
        super(CSTRANSMADDPG, self).__init__(args)
        self.cs_num = 1
        self.multiplier = th.nn.Parameter(th.tensor([args.init_lambda for _ in range(self.cs_num)],device=self.device))
        self.upper_bound = args.upper_bound
        # self.cs_mask = th.tensor(args.constraint_mask).to(self.device) # (n,s)

        self.obs_bus_dim = args.obs_bus_dim
        self.obs_bus_num = np.max(args.obs_bus_num)
        self.agent_index_in_obs = args.agent_index_in_obs
        self.obs_mask = th.zeros(self.n_, self.obs_bus_num).to(self.device)
        self.q_index = -1
        for i in range(self.n_):
            self.obs_mask[i,args.obs_bus_num[i]:] = -np.inf
        self.agent2region = args.agent2region
        self.region_num = np.max(self.agent2region) + 1
        self.encoder = TransformerEncoder(self.obs_bus_num, self.obs_bus_dim + self.region_num, args)
        self.construct_model()
        self.apply(self.init_weights)
        if target_net != None:
            self.target_net = target_net
            self.reload_params_to_target()
        self.batchnorm = nn.BatchNorm1d(self.args.agent_num).to(self.device)
        
    def construct_policy_net(self):
        if self.args.agent_type == 'transformer':
            from agents.transformer_agent import TransformerAgent
            Agent = TransformerAgent
        else:
            NotImplementedError()
        if self.args.shared_params:
            self.policy_dicts = nn.ModuleList([ Agent(self.args) ])
        else:
            self.policy_dicts = nn.ModuleList([ Agent(self.args) for _ in range(self.n_) ])

    def construct_value_net(self):
        if self.args.encoder:
            # input_shape = (self.obs_bus_num * self.args.out_hid_size + self.act_dim) * self.n_ + self.n_
            if self.args.merge_act:
                input_shape = self.args.hid_size * self.n_ + self.n_
            else:
                input_shape = (self.args.hid_size + self.act_dim) * self.n_ + self.n_
            # input_shape = (self.obs_dim + self.act_dim) * self.n_ + self.n_
            output_shape = 1
            self.value_dicts = nn.ModuleList( [ TransformerCritic(input_shape, output_shape, self.args, self.args.use_date) ] )
            self.cost_dicts = nn.ModuleList( [ TransformerCritic(input_shape, output_shape, self.args, self.args.use_date) for _ in range(self.cs_num)] )
        else:
            input_shape = (self.obs_dim + self.act_dim) * self.n_ + self.n_
            output_shape = 1
            self.value_dicts = nn.ModuleList( [ MLPCritic(input_shape, output_shape, self.args, self.args.use_date) ] )
            self.cost_dicts = nn.ModuleList( [ MLPCritic(input_shape, output_shape, self.args, self.args.use_date) for _ in range(self.cs_num) ] )


    def construct_model(self):
        self.construct_value_net()
        self.construct_policy_net()

    def update_target(self):
        for name, param in self.target_net.policy_dicts.state_dict().items():
            update_params = (1 - self.args.target_lr) * param + self.args.target_lr * self.policy_dicts.state_dict()[name]
            self.target_net.policy_dicts.state_dict()[name].copy_(update_params)
        for name, param in self.target_net.value_dicts.state_dict().items():
            update_params = (1 - self.args.target_lr) * param + self.args.target_lr * self.value_dicts.state_dict()[name]
            self.target_net.value_dicts.state_dict()[name].copy_(update_params)
        if self.args.mixer:
            for name, param in self.target_net.mixer.state_dict().items():
                update_params = (1 - self.args.target_lr) * param + self.args.target_lr * self.mixer.state_dict()[name]
                self.target_net.mixer.state_dict()[name].copy_(update_params)
        if self.args.encoder:
            for name, param in self.target_net.encoder.state_dict().items():
                update_params = (1 - self.args.target_lr) * param + self.args.target_lr * self.encoder.state_dict()[name]
                self.target_net.encoder.state_dict()[name].copy_(update_params)
        if self.args.multiplier:
            for name, param in self.target_net.cost_dicts.state_dict().items():
                update_params = (1 - self.args.target_lr) * param + self.args.target_lr * self.cost_dicts.state_dict()[name]
                self.target_net.cost_dicts.state_dict()[name].copy_(update_params)

    def encode(self, raw_obs, raw_act, merge_act = False):
        # obs_shape = (b, n, o)
        # act_shape = (b, n, a)
        assert not merge_act or raw_act.shape[-1] == 1
        batch_size = raw_obs.size(0)
        obs = raw_obs.view(batch_size*self.n_, self.obs_bus_num, self.obs_bus_dim).contiguous() # (b*n, self.obs_bus_num, self.obs_bus_dim)
        if merge_act:
            act = raw_act.view(batch_size*self.n_, -1).contiguous()
            agent_index = F.one_hot(th.tensor(self.agent_index_in_obs)[None,:].repeat(batch_size, 1).view(batch_size*self.n_)).to(self.device)
            # for i in range(batch_size*self.n_):
            #     obs[i][agent_index[i]][self.q_index] = act[i]
            obs[:,:,self.q_index] = (1 - agent_index) * obs[:,:,self.q_index] + agent_index * act

        zone_id = F.one_hot(th.tensor(self.agent2region)).to(self.device).float()
        zone_id = zone_id[None,:,None,:].contiguous().repeat(batch_size, 1, self.obs_bus_num, 1).view(batch_size*self.n_, self.obs_bus_num, self.region_num)
        mask = self.obs_mask[None,:,None,:].repeat(batch_size,1,1,1).view(batch_size*self.n_,1,-1).contiguous() # (b*n, 1, obs_bus_num)
        agent_index = th.tensor(self.agent_index_in_obs)[None,:,None].repeat(batch_size, 1, 1).view(batch_size*self.n_, 1).contiguous().to(self.device)
        obs, _ = self.encoder(th.cat((obs,zone_id),dim=-1), None, agent_index, mask)
        obs = obs.view(batch_size, self.n_, -1).contiguous()
        return obs

    def policy(self, raw_obs, schedule=None, last_act=None, last_hid=None, info={}, stat={}):
        # obs_shape = (b, n, o)
        batch_size = raw_obs.size(0)

        if self.args.shared_params:
            obs = raw_obs.view(batch_size*self.n_, self.obs_bus_num, self.obs_bus_dim).contiguous() # (b*n, self.obs_bus_num, self.obs_bus_dim)
            zone_id = F.one_hot(th.tensor(self.agent2region)).to(self.device).float()
            zone_id = zone_id[None,:,None,:].contiguous().repeat(batch_size, 1, self.obs_bus_num, 1).view(batch_size*self.n_, self.obs_bus_num, self.region_num)
            mask = self.obs_mask[None,:,None,:].repeat(batch_size,1,1,1).view(batch_size*self.n_,1,-1).contiguous() # (b*n, 1, obs_bus_num)
            # mask = th.cat((mask, th.zeros(batch_size*self.n_, 1, 1).float().to(self.device)),dim = -1) # (b*n, 1, obs_bus_num +1 ) for hidden_state
            agent_index = th.tensor(self.agent_index_in_obs)[None,:,None].repeat(batch_size, 1, 1).view(batch_size*self.n_, 1).contiguous().to(self.device)
            obs = th.cat((obs,zone_id),dim=-1)
            enc_obs, _ = self.encoder(obs, None, agent_index, mask)
            agent_policy = self.policy_dicts[0]
            means, log_stds, hiddens = agent_policy(enc_obs, last_hid)
            # hiddens = th.stack(hiddens, dim=1)
            means = means.contiguous().view(batch_size, self.n_, -1)
            hiddens = hiddens.contiguous().view(batch_size, self.n_, -1)
            if self.args.gaussian_policy:
                log_stds = log_stds.contiguous().view(batch_size, self.n_, -1)
            else:
                stds = th.ones_like(means).to(self.device) * self.args.fixed_policy_std
                log_stds = th.log(stds)
        else:
            NotImplementedError()

        return means, log_stds, hiddens

    def value(self, obs, act):
        # obs_shape = (b, n, o)
        # act_shape = (b, n, a)
        batch_size = obs.size(0)
        if self.args.encoder:
            obs = self.encode(obs, act, self.args.merge_act)

            obs_repeat = obs.unsqueeze(1).repeat(1, self.n_, 1, 1) # shape = (b, n, n, h)
            obs_reshape = obs_repeat.contiguous().view(batch_size, self.n_, -1) # shape = (b, n, n*h)

            # add agent id
            agent_ids = th.eye(self.n_).unsqueeze(0).repeat(batch_size, 1, 1).to(self.device) # shape = (b, n, n)
            if self.args.agent_id:
                obs_reshape = th.cat( (obs_reshape, agent_ids), dim=-1 ) # shape = (b, n, n*h+n)
            
            if self.args.shared_params:
                obs_reshape = obs_reshape.contiguous().view( batch_size*self.n_, -1 ) # shape = (b*n, n*h+n)
            else:
                NotImplementedError()
            if self.args.merge_act:
                inputs = obs_reshape
            else:
                act_repeat = act.unsqueeze(1).repeat(1, self.n_, 1, 1).contiguous().view( batch_size*self.n_, -1 ) # shape = (b*n, n*a)
                inputs = th.cat( (obs_reshape, act_repeat), dim=-1 )
        else:
            obs_repeat = obs.unsqueeze(1).repeat(1, self.n_, 1, 1) # shape = (b, n, n, o)
            obs_reshape = obs_repeat.contiguous().view(batch_size, self.n_, -1) # shape = (b, n, n*o)

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
            values, _ = agent_value(inputs, None)
            values = values.contiguous().view(batch_size, self.n_, 1)
            costs = []
            for agent_cost in self.cost_dicts:
                cost, _ = agent_cost(inputs, None)
                costs.append(cost)
            costs = th.cat(costs, dim=-1).view(batch_size, self.n_, self.cs_num) # (B, N, S)
        else:
            NotImplementedError()

        return values, costs

    def get_actions(self, state, status, exploration, actions_avail, target=False, last_hid=None):
        target_policy = self.target_net.policy if self.args.target else self.policy
        if self.args.continuous:
            means, log_stds, hiddens = self.policy(state, last_hid=last_hid) if not target else target_policy(state, last_hid=last_hid)
            if means.size(-1) > 1:
                means_ = means.sum(dim=1, keepdim=True)
                log_stds_ = log_stds.sum(dim=1, keepdim=True)
            else:
                means_ = means
                log_stds_ = log_stds
            actions, log_prob_a = select_action(self.args, means_, status=status, exploration=exploration, info={'log_std': log_stds_})
            restore_mask = 1. - (actions_avail == 0).to(self.device).float()
            restore_actions = restore_mask * actions
            action_out = (means, log_stds)
        else:
            logits, _, hiddens = self.policy(state, last_hid=last_hid) if not target else target_policy(state, last_hid=last_hid)
            logits[actions_avail == 0] = -9999999
            actions, log_prob_a = select_action(self.args, logits, status=status, exploration=exploration)
            restore_actions = actions
            action_out = logits
        return actions, restore_actions, log_prob_a, action_out, hiddens

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
        returns = rewards - self.multiplier.detach() * cost  + self.args.gamma * (1 - done) * next_values.detach() 
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
        for i in range(self.cs_num):
            if self.multiplier[i] < 0:
                with th.no_grad():
                    self.multiplier[i] = 0.