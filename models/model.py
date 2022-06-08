import torch as th
import torch.nn as nn
import numpy as np
from collections import namedtuple
from utilities.util import prep_obs, translate_action, rev_translate_action
from utilities.define import *


class Model(nn.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.device = th.device(
            "cuda" if th.cuda.is_available() and self.args.cuda else "cpu")
        self.n_ = self.args.agent_num
        self.hid_dim = self.args.hid_size
        self.obs_dim = self.args.obs_size
        self.act_dim = self.args.action_dim
        self.Transition = namedtuple('Transition', ('state', 'action', 'log_prob_a', 'value', 'next_value',
                                     'reward', 'cost', 'next_state', 'done', 'last_step', 'action_avail', 'last_hid', 'hid'))
        self.batchnorm = nn.BatchNorm1d(self.n_)
        self.costbatchnorm = nn.BatchNorm1d(self.n_)

    def reload_params_to_target(self):
        self.target_net.policy_dicts.load_state_dict(
            self.policy_dicts.state_dict())
        self.target_net.value_dicts.load_state_dict(
            self.value_dicts.state_dict())
        if self.args.mixer:
            self.target_net.mixer.load_state_dict(self.mixer.state_dict())

    def update_target(self):
        for name, param in self.target_net.policy_dicts.state_dict().items():
            update_params = (1 - self.args.target_lr) * param + \
                self.args.target_lr * self.policy_dicts.state_dict()[name]
            self.target_net.policy_dicts.state_dict()[
                name].copy_(update_params)
        for name, param in self.target_net.value_dicts.state_dict().items():
            update_params = (1 - self.args.target_lr) * param + \
                self.args.target_lr * self.value_dicts.state_dict()[name]
            self.target_net.value_dicts.state_dict()[name].copy_(update_params)
        if self.args.mixer:
            for name, param in self.target_net.mixer.state_dict().items():
                update_params = (1 - self.args.target_lr) * param + \
                    self.args.target_lr * self.mixer.state_dict()[name]
                self.target_net.mixer.state_dict()[name].copy_(update_params)

    def transition_update(self, trainer, trans, stat, destroy):
        if self.args.replay:
            trainer.replay_buffer.add_experience(trans)
            replay_cond = trainer.steps > self.args.replay_warmup\
                and len(trainer.replay_buffer.buffer) >= self.args.batch_size\
                and (trainer.steps % self.args.behaviour_update_freq == 0 or destroy == 1.)
            if replay_cond:
                if self.args.auxiliary:
                    for _ in range(self.args.auxiliary_update_epochs):
                        trainer.auxiliary_replay_process(stat)
                for _ in range(self.args.value_update_epochs):
                    trainer.value_replay_process(stat)
                for _ in range(self.args.policy_update_epochs):
                    trainer.policy_replay_process(stat)
                if self.args.mixer:
                    for _ in range(self.args.mixer_update_epochs):
                        trainer.mixer_replay_process(stat)
                if self.args.multiplier:
                    trainer.lambda_replay_process(stat)
                # TODO: hard code
                # clear replay buffer for on-policy algorithm
                if self.__class__.__name__ in ["COMA", "IAC", "IPPO", "MAPPO"]:
                    trainer.replay_buffer.clear()
        else:
            trans_cond = trainer.steps % self.args.behaviour_update_freq == 0
            if trans_cond:
                for _ in range(self.args.value_update_epochs):
                    trainer.value_replay_process(stat)
                for _ in range(self.args.policy_update_epochs):
                    trainer.policy_replay_process(stat, trans)
                if self.args.mixer:
                    for _ in range(self.args.mixer_update_epochs):
                        trainer.mixer_replay_process(stat)
                if self.args.multiplier:
                    trainer.lambda_replay_process(stat)
        if self.args.target:
            target_cond = trainer.steps % self.args.target_update_freq == 0
            if target_cond:
                self.update_target()

    def episode_update(self, trainer, episode, stat):
        if self.args.replay:
            trainer.replay_buffer.add_experience(episode)
            replay_cond = trainer.episodes > self.args.replay_warmup\
                and len(trainer.replay_buffer.buffer) >= self.args.batch_size\
                and trainer.episodes % self.args.behaviour_update_freq == 0
            if replay_cond:
                for _ in range(self.args.value_update_epochs):
                    trainer.value_replay_process(stat)
                for _ in range(self.args.policy_update_epochs):
                    trainer.policy_replay_process(stat)
                if self.args.mixer:
                    for _ in range(self.args.mixer_update_epochs):
                        trainer.mixer_replay_process(stat)
        else:
            episode = self.Transition(*zip(*episode))
            episode_cond = trainer.episodes % self.args.behaviour_update_freq == 0
            if episode_cond:
                for _ in range(self.args.value_update_epochs):
                    trainer.value_replay_process(stat)
                for _ in range(self.args.policy_update_epochs):
                    trainer.policy_replay_process(stat)
                if self.args.mixer:
                    for _ in range(self.args.mixer_update_epochs):
                        trainer.mixer_replay_process(stat)

    def construct_model(self):
        raise NotImplementedError()

    def policy(self, obs, schedule=None, last_act=None, last_hid=None, info={}, stat={}):
        # obs_shape = (b, n, o)
        batch_size = obs.size(0)

        # add agent id
        if self.args.agent_id:
            agent_ids = th.eye(self.n_).unsqueeze(0).repeat(
                batch_size, 1, 1).to(self.device)  # shape = (b, n, n)
            obs = th.cat((obs, agent_ids), dim=-1)  # shape = (b, n, n+o)

        if self.args.shared_params:
            # print (f"This is the shape of last_hids: {last_hid.size()}")
            obs = obs.contiguous().view(batch_size*self.n_, -1)  # shape = (b*n, n+o/o)
            agent_policy = self.policy_dicts[0]
            means, log_stds, hiddens = agent_policy(obs, last_hid)
            # hiddens = th.stack(hiddens, dim=1)
            means = means.contiguous().view(batch_size, self.n_, -1)
            hiddens = hiddens.contiguous().view(batch_size, self.n_, -1)
            if self.args.gaussian_policy:
                log_stds = log_stds.contiguous().view(batch_size, self.n_, -1)
            else:
                stds = th.ones_like(means).to(self.device) * \
                    self.args.fixed_policy_std
                log_stds = th.log(stds)
        else:
            means = []
            hiddens = []
            log_stds = []
            for i, agent_policy in enumerate(self.policy_dicts):
                mean, log_std, hidden = agent_policy(
                    obs[:, i, :], last_hid[:, i, :])
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

    def value(self, obs, act, last_act=None, last_hid=None):
        raise NotImplementedError()

    def construct_policy_net(self):
        if self.args.agent_id:
            input_shape = self.obs_dim + self.n_
        else:
            input_shape = self.obs_dim

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
        elif self.args.agent_type == 'rnn_aux':
            if self.args.gaussian_policy:
                NotImplementedError()
            else:
                from agents.rnn_aux_agent import RNNAgent
            Agent = RNNAgent
        elif self.args.agent_type == "rnn_ex":
            if self.args.gaussian_policy:
                NotImplementedError()
            else:
                from agents.rnn_agent_ex import RNNAgent
            Agent = RNNAgent
        else:
            NotImplementedError()

        if self.args.shared_params:
            self.policy_dicts = nn.ModuleList([Agent(input_shape, self.args)])
        else:
            self.policy_dicts = nn.ModuleList(
                [Agent(input_shape, self.args) for _ in range(self.n_)])

    def construct_value_net(self):
        raise NotImplementedError()

    def init_weights(self, m):
        '''
        initialize the weights of parameters
        '''
        if type(m) == nn.Linear:
            if self.args.init_type == "normal":
                nn.init.normal_(m.weight, 0.0, self.args.init_std)
            elif self.args.init_type == "orthogonal":
                nn.init.orthogonal_(
                    m.weight, gain=nn.init.calculate_gain(self.args.hid_activation))

    def get_actions(self):
        raise NotImplementedError()

    def get_loss(self):
        raise NotImplementedError()

    def credit_assignment_demo(self, obs, act):
        assert isinstance(obs, np.ndarray)
        assert isinstance(act, np.ndarray)
        obs = th.tensor(obs).to(self.device).float()
        act = th.tensor(act).to(self.device).float()
        values = self.value(obs, act)
        if isinstance(values, tuple):
            values, costs = values
        return values

    def train_process(self, stat, trainer):
        stat_train = {'mean_train_reward': 0, 'mean_train_solver_infeasible': 0,
                      'mean_train_solver_interventions': 0}

        if self.args.episodic:
            episode = []

        # reset env
        state, global_state = trainer.env.reset()
        # state, global_state = trainer.env.manual_reset(199, 23, 2)

        # init hidden states
        last_hid = self.policy_dicts[0].init_hidden()

        for t in range(self.args.max_steps):
            # current state, action, value
            state_ = prep_obs(state).to(self.device).contiguous().view(
                1, self.n_, self.obs_dim)
            with th.no_grad():
                action, action_pol, log_prob_a, _, hid = self.get_actions(state_, status='train', exploration=True, actions_avail=th.tensor(
                    trainer.env.get_avail_actions()), target=False, last_hid=last_hid)
                value = self.value(state_, action_pol)
            if isinstance(value, tuple):
                value, cost = value
            _, actual = translate_action(self.args, action, trainer.env)

            # safe filter
            # if self.args.safe_filter != 'none':
            #     global_state_ = th.tensor(global_state).to(
            #         th.float32).to(self.device).contiguous().view(1, -1)
            #     actual_ = th.tensor(actual).to(th.float32).to(
            #         self.device).contiguous().view(1, -1)
            #     k = np.array(trainer.env.get_q_divide_a_coff())
            #     lb = trainer.env.action_space.low
            #     ub = trainer.env.action_space.high
            #     if self.args.safe_filter == 'hard':
            #         actual, flag = self.correct_actions_hard(
            #             global_state_, actual_, k, lb, ub)
            #     elif self.args.safe_filter == 'soft':
            #         actual, flag = self.correct_actions_soft(
            #             global_state_, actual_, k, lb, ub, trainer.env.pv_index)
            #     safe_action_pol = rev_translate_action(
            #         self.args, actual, trainer.env)
            #     # actual = actual.detach().squeeze().cpu().numpy()
            #     if flag == INFEASIBLE:
            #         stat_train['mean_train_solver_infeasible'] += 1
            #     if flag == INTERVENTIONS:
            #         stat_train['mean_train_solver_interventions'] += 1

            # reward
            reward, done, info = trainer.env.step(actual)
            if trainer.env.independ_reward:
                reward_repeat = reward
            else:
                reward_repeat = [reward]*trainer.env.get_num_of_agents()
            # if self.args.split_constraint:
            #     if self.args.cost_type == "region":
            #         out_of_control = [
            #             info['percentage_of_v_out_of_control_region'] for _ in range(self.n_)]
            #     elif self.args.cost_type == 'agent':
            #         out_of_control = info['percentage_of_v_out_of_control_agent']
            #     else:
            #         NotImplementedError()
            # else:
            #     out_of_control = [
            #         info['percentage_of_v_out_of_control']] * trainer.env.get_num_of_agents()
            out_of_control = [info['percentage_of_v_out_of_control']] * trainer.env.get_num_of_agents()
            # next state, action, value
            next_state = trainer.env.get_obs()
            next_state_ = prep_obs(next_state).to(
                self.device).contiguous().view(1, self.n_, self.obs_dim)
            with th.no_grad():
                _, next_action_pol, _, _, _ = self.get_actions(next_state_, status='train', exploration=True, actions_avail=th.tensor(
                    trainer.env.get_avail_actions()), target=False, last_hid=hid)
                next_value = self.value(next_state_, next_action_pol)
            if isinstance(next_value, tuple):
                next_value, next_cost = next_value
            # store trajectory
            if isinstance(done, list):
                done = np.sum(done)
            done_ = done or t == self.args.max_steps-1
            # if not self.args.safe_trans or info["totally_controllable_ratio"] == 1.:
            trans = self.Transition(state,
                                    # action_pol.detach().cpu().numpy() if self.args.safe_filter == 'none' else safe_action_pol,
                                    action_pol.detach().cpu().numpy(),
                                    log_prob_a,
                                    value.detach().cpu().numpy(),
                                    next_value.detach().cpu().numpy(),
                                    np.array(reward_repeat),
                                    np.array(out_of_control),
                                    next_state,
                                    done,
                                    done_,
                                    trainer.env.get_avail_actions(),
                                    last_hid.detach().cpu().numpy(),
                                    hid.detach().cpu().numpy()
                                    )
            if not self.args.episodic:
                self.transition_update(
                    trainer, trans, stat, info["destroy"])
            else:
                episode.append(trans)
            for k, v in info.items():
                if type(v) is not np.ndarray:
                    if 'mean_train_'+k not in stat_train.keys():
                        stat_train['mean_train_' + k] = v
                    else:
                        stat_train['mean_train_' + k] += v
            stat_train['mean_train_reward'] += np.mean(reward)
            # if not self.args.safe_trans or info["totally_controllable_ratio"] == 1.:
            trainer.steps += 1
            if done_:
                break
            # set the next state
            state = next_state
            # set the next last_hid
            last_hid = hid
        trainer.episodes += 1
        for k, v in stat_train.items():
            key_name = k.split('_')
            if key_name[0] == 'mean':
                stat_train[k] = v / float(t+1)
        stat.update(stat_train)
        if self.args.episodic:
            self.episode_update(trainer, episode, stat)

    def evaluation(self, stat, trainer, test_season='June'):
        num_eval_episodes = self.args.num_eval_episodes
        stat_test = {}
        constraint_model = trainer.constraint_model
        if test_season == "June":
            test_data = [
                529,
                893,
                152,
                160,
                530,
                902,
                903,
                905,
                520,
                526,
            ]
        elif test_season == "All":
            test_data = [
                46,
                454,
                836,
                868,
                903,
                931,
                948,
                621,
                646,
                332,
            ]
        else:
            NotImplementedError()
        trainer.env.set_episode_limit(self.args.max_eval_steps)
        with th.no_grad():
            for _ in range(num_eval_episodes):
                stat_test_epi = {'mean_test_reward': 0}
                # state, global_state = trainer.env.reset()
                state, global_state = trainer.env.manual_reset(
                    test_data[_], 23, 2)
                # init hidden states
                last_hid = self.policy_dicts[0].init_hidden()
                for t in range(self.args.max_eval_steps):
                    state_ = prep_obs(state).to(self.device).contiguous().view(
                        1, self.n_, self.obs_dim)
                    action, _, _, _, hid = self.get_actions(state_, status='test', exploration=False, actions_avail=th.tensor(
                        trainer.env.get_avail_actions()), target=False, last_hid=last_hid)
                    _, actual = translate_action(
                        self.args, action, trainer.env)
                    reward, done, info = trainer.env.step(actual)
                    done_ = done or t == self.args.max_eval_steps-1
                    next_state = trainer.env.get_obs()
                    # if constraint_model is not None:
                    #     with th.no_grad():
                    #         q = trainer.env.now_q
                    #         state = trainer.env.get_state()
                    #         label_v = state[-2*len(trainer.env.base_powergrid.bus):-len(trainer.env.base_powergrid.bus)]
                    #         state = th.tensor(state).to(th.float32).to(self.device)[None,:]
                    #         q = th.tensor(q).to(th.float32).to(self.device)[None,:]
                    #         pred_v = constraint_model(th.cat((state,q),dim=1)).detach().squeeze().cpu().numpy()
                    #         stat_test_epi['mean_test_constraint_error'] += np.mean(np.abs(pred_v - label_v))
                    #         stat_test_min_max['max_test_constraint_error'] = max(stat_test_min_max['max_test_constraint_error'], np.max(pred_v - label_v))
                    #         stat_test_min_max['min_test_constraint_error'] = min(stat_test_min_max['min_test_constraint_error'], np.min(pred_v - label_v))

                    if isinstance(done, list):
                        done = np.sum(done)
                    for k, v in info.items():
                        if type(v) is not np.ndarray:
                            if 'mean_test_' + k not in stat_test_epi.keys():
                                stat_test_epi['mean_test_' + k] = v
                            else:
                                stat_test_epi['mean_test_' + k] += v
                    stat_test_epi['mean_test_reward'] += np.mean(reward)
                    if done_:
                        break
                    # set the next state
                    state = next_state
                    # set the next last_hid
                    last_hid = hid
                for k, v in stat_test_epi.items():
                    stat_test_epi[k] = v / float(t+1)
                for k, v in stat_test_epi.items():
                    k = test_season + "_" + k
                    if k not in stat_test.keys():
                        stat_test[k] = v
                    else:
                        stat_test[k] += v
        for k, v in stat_test.items():
            stat_test[k] = v / float(num_eval_episodes)
        stat.update(stat_test)
        trainer.env.set_episode_limit(self.args.max_steps)

    def unpack_data(self, batch):
        reward = th.tensor(batch.reward, dtype=th.float).to(self.device)
        cost = th.tensor(batch.cost, dtype=th.float).to(self.device)
        last_step = th.tensor(batch.last_step, dtype=th.float).to(
            self.device).contiguous().view(-1, 1)
        done = th.tensor(batch.done, dtype=th.float).to(
            self.device).contiguous().view(-1, 1)
        action = th.tensor(np.concatenate(batch.action, axis=0),
                           dtype=th.float).to(self.device)
        log_prob_a = th.tensor(np.concatenate(
            batch.action, axis=0), dtype=th.float).to(self.device)
        value = th.tensor(np.concatenate(batch.value, axis=0),
                          dtype=th.float).to(self.device)
        next_value = th.tensor(np.concatenate(
            batch.next_value, axis=0), dtype=th.float).to(self.device)
        state = prep_obs(list(zip(batch.state))).to(self.device)
        next_state = prep_obs(list(zip(batch.next_state))).to(self.device)
        action_avail = th.tensor(np.concatenate(
            batch.action_avail, axis=0)).to(self.device)
        last_hid = th.tensor(np.concatenate(
            batch.last_hid, axis=0), dtype=th.float).to(self.device)
        hid = th.tensor(np.concatenate(batch.hid, axis=0),
                        dtype=th.float).to(self.device)
        if self.args.reward_normalisation:
            reward = self.batchnorm(reward).to(self.device)
            # cost = self.costbatchnorm(cost).to(self.device)
        return (state, action, log_prob_a, value, next_value, reward, cost, next_state, done, last_step, action_avail, last_hid, hid)
