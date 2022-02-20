from numpy.core.fromnumeric import repeat
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utilities.util import select_action
from models.model import Model
from critics.mlp_critic import MLPCritic
from critics.transformer_encoder import TransformerEncoder


class ENCMADDPG(Model):
    def __init__(self, args, target_net=None):
        super(ENCMADDPG, self).__init__(args)
        self.obs_bus_dim = args.obs_bus_dim
        self.obs_bus_num = np.max(args.obs_bus_num)
        self.agent2region = args.agent2region
        self.region_num = np.max(self.agent2region) + 1
        self.actor_encoder = nn.ModuleList()
        self.critic_encoder = nn.ModuleList()
        self.actor_encoder.append(TransformerEncoder(self.obs_bus_num, self.obs_bus_dim + self.region_num, args))
        self.critic_encoder.append(TransformerEncoder(self.obs_bus_num, self.obs_bus_dim + self.region_num, args))
        self.construct_model()
        self.apply(self.init_weights)
        if target_net != None:
            self.target_net = target_net
            self.reload_params_to_target()
        self.batchnorm = nn.BatchNorm1d(self.args.agent_num).to(self.device)
        

    def construct_policy_net(self):
        if self.args.agent_id:
            # input_shape = self.obs_bus_num * self.args.out_hid_size + self.n_
            input_shape = self.obs_dim + self.n_
        else:
            input_shape = self.obs_bus_num * self.args.out_hid_size

        if self.args.agent_type == 'mlp':
            if self.args.gaussian_policy:
                from agents.mlp_agent_gaussian import MLPAgent
            else:
                from agents.mlp_agent import MLPAgent
            Agent = MLPAgent
        elif self.args.agent_type == 'rnn':
            if self.args.gaussian_policy:
                from agents.rnn_agent_gaussian import RNNAgent
            else:
                from agents.rnn_agent import RNNAgent
            Agent = RNNAgent
        elif self.args.agent_type == 'rnn_with_date':
            if self.args.gaussian_policy:
                NotImplementedError()
            else:
                from agents.rnn_agent_dateemb import RNNAgent
            Agent = RNNAgent
        else:
            NotImplementedError()
            
        if self.args.shared_params:
            self.policy_dicts = nn.ModuleList([ Agent(input_shape, self.args) ])
        else:
            self.policy_dicts = nn.ModuleList([ Agent(input_shape, self.args) for _ in range(self.n_) ])

    def construct_value_net(self):
        if self.args.agent_id:
            input_shape = (self.obs_bus_num * self.args.out_hid_size + self.act_dim) * self.n_ + self.n_
            # input_shape = (self.obs_dim + self.act_dim) * self.n_ + self.n_
        else:
            input_shape = (self.obs_bus_num * self.args.out_hid_size + self.act_dim) * self.n_
        if self.args.use_date:
            input_shape -= self.args.date_dim * (self.n_ - 1 )

        output_shape = 1
        if self.args.shared_params:
            self.value_dicts = nn.ModuleList( [ MLPCritic(input_shape, output_shape, self.args, self.args.use_date) ] )
        else:
            self.value_dicts = nn.ModuleList( [ MLPCritic(input_shape, output_shape, self.args, self.args.use_date) for _ in range(self.n_) ] )
        # if self.args.agent_id:
        #     input_shape = (self.args.hid_size + self.act_dim) * self.n_ + self.n_
        # else:
        #     input_shape = (self.args.hid_size + self.act_dim) * self.n_
        # self.value_dicts = nn.ModuleList( [ TransformerCritic(self.obs_dim, self.act_dim, input_shape, output_shape, self.args) ] )

    def construct_model(self):
        self.construct_value_net()
        self.construct_policy_net()

    def encode(self, encoder, raw_obs):
        # trans_obs = raw_obs.transpose(0,1).contiguous()
        # obs = []
        # for i in range(self.n_):
        #     obs.append(self.actor_encoder[i](trans_obs[i]))
        # obs = th.stack(obs)
        # obs = obs.contiguous().transpose(0,1)
        batch_size = raw_obs.size(0)
        obs = raw_obs.view(batch_size*self.n_, self.obs_bus_dim, self.obs_bus_num).transpose(1,2).contiguous() # (b*n, self.obs_bus_num, self.obs_bus_dim)
        zone_id = F.one_hot(th.tensor(self.agent2region)).to(self.device).float()
        zone_id = zone_id[None,:,None,:].contiguous().repeat(batch_size, 1, self.obs_bus_num, 1).view(batch_size*self.n_, self.obs_bus_num, self.region_num)
        obs = encoder[0](th.cat((obs,zone_id),dim=-1)).view(batch_size, self.n_, -1).contiguous()
        return obs

    def policy(self, raw_obs, schedule=None, last_act=None, last_hid=None, info={}, stat={}):
        # obs_shape = (b, n, o)
        batch_size = raw_obs.size(0)
        # obs = self.encode(self.actor_encoder, raw_obs)
        obs = raw_obs

        # add agent id
        if self.args.agent_id:
            agent_ids = th.eye(self.n_).unsqueeze(0).repeat(batch_size, 1, 1).to(self.device) # shape = (b, n, n)
            obs = th.cat( (obs, agent_ids), dim=-1 ) # shape = (b, n, n+o)

        if self.args.shared_params:
            # print (f"This is the shape of last_hids: {last_hid.size()}")
            obs = obs.contiguous().view(batch_size*self.n_, -1) # shape = (b*n, n+o/o)
            agent_policy = self.policy_dicts[0]
            means, log_stds, hiddens = agent_policy(obs, last_hid)
            # hiddens = th.stack(hiddens, dim=1)
            means = means.contiguous().view(batch_size, self.n_, -1)
            hiddens = hiddens.contiguous().view(batch_size, self.n_, -1)
            if self.args.gaussian_policy:
                log_stds = log_stds.contiguous().view(batch_size, self.n_, -1)
            else:
                stds = th.ones_like(means).to(self.device) * self.args.fixed_policy_std
                log_stds = th.log(stds)
        else:
            means = []
            hiddens = []
            log_stds = []
            for i, agent_policy in enumerate(self.policy_dicts):
                mean, log_std, hidden = agent_policy(obs[:, i, :], last_hid[:, i, :])
                means.append(mean)
                hiddens.append(hidden)
                log_stds.append(log_std)
            means = th.stack(means, dim=1)
            hiddens = th.stack(hiddens, dim=1)
            if self.args.gaussian_policy:
                log_stds = th.stack(log_stds, dim=1)
            else:
                log_stds = th.zeros_like(means).to(self.device)

        return means, log_stds, hiddens

    def value(self, raw_obs, act):
        # obs_shape = (b, n, o)
        # act_shape = (b, n, a)
        batch_size = raw_obs.size(0)
        obs = self.encode(self.critic_encoder, raw_obs)
        # obs = raw_obs
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
        else:
            values = []
            for i, agent_value in enumerate(self.value_dicts):
                value, _ = agent_value(inputs[:, i, :], None)
                values.append(value)
            values = th.stack(values, dim=1)

        return values

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
        return policy_loss, value_loss, action_out, None
