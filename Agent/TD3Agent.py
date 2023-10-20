import os
import sys
import django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "power.power.settings")
django.setup()
from django.conf import settings
from grid.models import PowerGrid
import copy
import torch
import pickle
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from tqdm import tqdm
from Agent.model.Actor_Critic import *
from utilize.form_action import *
from Agent.BaseAgent import BaseAgent
from Environment.base_env import Environment
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class DDPG:
    def __init__(self, settings, config):
        # set basic info
        self.settings = settings
        self.config = config
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        self.device = config.device

        # set actor model, optimizer and scheduler
        self.actor = Actor(settings, config).to(self.device)
        self.actor_target = Actor(settings, config).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.lr_actor)
        self.actor_scheduler = torch.optim.lr_scheduler.StepLR(self.actor_optimizer,
                                                               step_size=config.lr_decay_step_size,
                                                               gamma=config.lr_decay_gamma)
        # set critic model, optimizer and scheduler
        self.critic1 = Critic(settings, config).to(self.device)
        self.critic2 = Critic(settings, config).to(self.device)
        self.critic1_target = Critic(settings, config).to(self.device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target = Critic(settings, config).to(self.device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.critic1_optimizer = torch.optim.Adam(self.critic1.parameters(), lr=config.lr_critic)
        self.critic1_scheduler = torch.optim.lr_scheduler.StepLR(self.critic1_optimizer,
                                                                 step_size=config.lr_decay_step_size,
                                                                 gamma=config.lr_decay_gamma)
        self.critic2_optimizer = torch.optim.Adam(self.critic2.parameters(), lr=config.lr_critic)
        self.critic2_scheduler = torch.optim.lr_scheduler.StepLR(self.critic2_optimizer,
                                                                 step_size=config.lr_decay_step_size,
                                                                 gamma=config.lr_decay_gamma)

        # set data buffer
        self.buffer = FIFOBuffer(config) if self.config.buffer_type == 'FIFO' else RolloutBuffer(config)
        # set action sample std
        self.action_std = config.init_action_std if hasattr(config, 'init_action_std') else 0.1
        # set balance loss rate
        self.balance_loss_rate = config.balance_loss_rate if hasattr(config, 'balance_loss_rate') else 0.001
        # data for calculate balance loss
        mid_val, min_val, max_val, \
        self.danger_region_lower, self.danger_region_upper, \
        self.warning_region_lower, self.warning_region_upper, \
        self.save_region_lower, self.save_region_upper = calculate_balance_loss_data(config)
        self.balance_mid_val = torch.tensor(mid_val, dtype=torch.float32).to(config.device)
        self.balance_min_val = torch.tensor(min_val, dtype=torch.float32).to(config.device)
        self.balance_max_val = torch.tensor(max_val, dtype=torch.float32).to(config.device)
        if self.config.split_balance_loss:
            self.danger_balance_loss_rate = torch.tensor(
                [config.danger_region_balance_loss_rate for _ in range(config.mini_batch_size)]).to(config.device) \
                if self.config.use_mini_batch else torch.tensor(
                [config.danger_region_balance_loss_rate for _ in range(config.batch_size)]).to(config.device)
            self.warning_balance_loss_rate = torch.tensor(
                [config.warning_region_balance_loss_rate for _ in range(config.mini_batch_size)]).to(config.device) \
                if self.config.use_mini_batch else torch.tensor(
                [config.warning_region_balance_loss_rate for _ in range(config.batch_size)]).to(config.device)
            self.save_balance_loss_rate = torch.tensor(
                [config.save_region_balance_loss_rate for _ in range(config.mini_batch_size)]).to(config.device) \
                if self.config.use_mini_batch else torch.tensor(
                [config.save_region_balance_loss_rate for _ in range(config.batch_size)]).to(config.device)

    def select_action(self, state, action_low, action_high, history_obs, history_actions, feature, adjacency, sample=True):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            action_low = torch.tensor(action_low, dtype=torch.float32).unsqueeze(0).to(self.device)
            action_high = torch.tensor(action_high, dtype=torch.float32).unsqueeze(0).to(self.device)
            history_obs = torch.tensor(history_obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            history_actions = torch.tensor(history_actions, dtype=torch.float32).unsqueeze(0).to(self.device)
            feature = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(self.device)
            adjacency = torch.tensor(adjacency, dtype=torch.float32).unsqueeze(0).to(self.device)
            action = self.actor(
                x=state,
                h_x=history_obs,
                h_a=history_actions,
                feat=feature,
                adj=adjacency,
                action_low=action_low,
                action_high=action_high,
                sample=True,
            )

        return action.detach().cpu().numpy().flatten()

    def add_data(self, obs, action, logprob, reward, done, next_obs=None,
                 action_low=None, action_high=None, next_action_low=None, next_action_high=None,
                 history_obs=None, next_history_obs=None, history_actions=None, next_history_actions=None,
                 feature=None, adjacency=None, next_feature=None, next_adjacency=None):

        self.buffer.add_data(obs, action, logprob, reward, done, next_obs,
                             action_low, action_high, next_action_low, next_action_high,
                             history_obs, next_history_obs, history_actions, next_history_actions,
                             feature, adjacency, next_feature, next_adjacency)

    def split_gen_p(self, gen_p):
        thermal_gen_p = gen_p[:, self.settings.thermal_ids].sum(dim=-1)
        renewable_gen_p = gen_p[:, self.settings.renewable_ids].sum(dim=-1)
        balance_gen_p = gen_p[:, self.settings.balanced_id]
        # print('-------------------------------- print shape --------------------------------')
        # print('thermal_gen_p', gen_p[:, self.settings.thermal_ids].shape)
        # print('renewable_gen_p', gen_p[:, self.settings.renewable_ids].shape)
        # print('balance_gen_p', gen_p[:, self.settings.balanced_id].shape)
        return thermal_gen_p, renewable_gen_p, balance_gen_p

    def calculate_balance_loss(self, obs, next_obs, action, state_norm, real_action, dones):
        # set dependence info
        detail_action_dim = self.config.detail_action_dim
        ori_obs = state_norm.inverse_normalization(obs) if self.config.use_state_norm else obs
        next_ori_obs = state_norm.inverse_normalization(next_obs) if self.config.use_state_norm else next_obs
        if self.config.min_state == 1:
            ori_obs = process_state_sum(ori_obs)
            next_ori_obs = process_state_sum(next_ori_obs)
        # calculate/predict next state power
        next_gen_p = ori_obs[:, 0] + ori_obs[:, 1] + torch.sum(
            action[:, detail_action_dim[0][0]: detail_action_dim[0][1]], dim=-1)
        next_simpleld_p = ori_obs[:, 3]
        next_adjld_p = ori_obs[:, 7] + torch.sum(action[:, detail_action_dim[2][0]: detail_action_dim[2][1]], dim=-1)
        next_stoenergy_p = torch.sum(action[:, detail_action_dim[3][0]: detail_action_dim[3][1]], dim=-1)
        next_grid_loss = ori_obs[:, 6]
        next_predict_balance_p = next_simpleld_p + next_adjld_p + next_stoenergy_p + next_grid_loss - next_gen_p
        # now_balance_p = ori_obs[:, 2]
        # now_predict_balance_gen_p = ori_obs[:, 3] + ori_obs[:, 4] + ori_obs[:, 5] + ori_obs[:, 6] - ori_obs[:, 0] - ori_obs[:, 1]
        # print('now state balance', now_balance_p - now_predict_balance_gen_p)
        # print('------------------------ balance loss -------------------------')
        # predict_next_gen_p = ori_obs[:, 0] + ori_obs[:, 1] + torch.sum(real_action[:, detail_action_dim[0][0]: detail_action_dim[0][1]], dim=-1)
        # predict_next_adjld_p = ori_obs[:, 7] + torch.sum(real_action[:, detail_action_dim[2][0]: detail_action_dim[2][1]], dim=-1)
        # predict_next_stoenergy_p = torch.sum(real_action[:, detail_action_dim[3][0]: detail_action_dim[3][1]], dim=-1)
        # print('predict_next_gen_p', predict_next_gen_p - next_ori_obs[:, 0] - next_ori_obs[:, 1])
        # print('predict_next_adjld_p', predict_next_adjld_p - next_ori_obs[:, 4])
        # print('predict_next_stoenergy_p', predict_next_stoenergy_p - next_ori_obs[:, 5])

        if self.config.split_balance_loss:
            balance_loss_rate = torch.where((next_predict_balance_p >= self.warning_region_lower) &
                                            (next_predict_balance_p <= self.warning_region_upper),
                                            self.warning_balance_loss_rate, self.danger_balance_loss_rate)
            balance_loss_rate = torch.where((next_predict_balance_p >= self.save_region_lower) &
                                            (next_predict_balance_p <= self.save_region_upper),
                                            self.save_balance_loss_rate, balance_loss_rate)
        else:
            balance_loss_rate = self.balance_loss_rate

        not_dones = (-dones + 1).squeeze(-1)
        return (((next_predict_balance_p - self.balance_mid_val) ** 2) * balance_loss_rate * not_dones).mean()

    def update(self, f_learner, state_norm, update_times):
        states, actions, rewards, dones, next_states, logprobs, \
        action_lows, action_highs, next_action_lows, next_action_highs, \
        history_states, next_history_states, history_actions, next_history_actions, \
        features, adjacencys, next_features, next_adjacencys, = \
            self.buffer.sample_data() if self.config.buffer_type == 'FIFO' else self.buffer.get_all_data()
        if self.config.use_mini_batch:
            for index in BatchSampler(SubsetRandomSampler(range(self.config.batch_size)), self.config.mini_batch_size, False):
                mini_states = states[index]
                mini_actions = actions[index]
                mini_rewards = rewards[index]
                mini_dones = dones[index]
                mini_next_states = next_states[index]
                mini_action_lows = action_lows[index]
                mini_action_highs = action_highs[index]
                mini_next_action_lows = next_action_lows[index]
                mini_next_action_highs = next_action_highs[index]
                mini_history_states = history_states[index]
                mini_next_history_states = next_history_states[index]
                mini_history_actions = history_actions[index]
                mini_next_history_actions = next_history_actions[index]
                mini_features = features[index]
                mini_adjacencys = adjacencys[index]
                mini_next_features = next_features[index]
                mini_next_adjacencys = next_adjacencys[index]

                target_Q1 = self.critic1_target(
                    x=mini_next_states,
                    h_x=mini_next_history_states,
                    a=self.actor_target(
                        x=mini_next_states,
                        h_x=mini_next_history_states,
                        h_a=mini_next_history_actions,
                        feat=mini_next_features,
                        adj=mini_next_adjacencys,
                        action_low=mini_next_action_lows,
                        action_high=mini_next_action_highs,
                        sample=False,
                    ),
                    h_a=mini_next_history_actions,
                    feat=mini_next_features,
                    adj=mini_next_adjacencys,
                )
                target_Q2 = self.critic2_target(
                    x=mini_next_states,
                    h_x=mini_next_history_states,
                    a=self.actor_target(
                        x=mini_next_states,
                        h_x=mini_next_history_states,
                        h_a=mini_next_history_actions,
                        feat=mini_next_features,
                        adj=mini_next_adjacencys,
                        action_low=mini_next_action_lows,
                        action_high=mini_next_action_highs,
                        sample=False,
                    ),
                    h_a=mini_next_history_actions,
                    feat=mini_next_features,
                    adj=mini_next_adjacencys,
                )
                target_Q = torch.min(target_Q1, target_Q2)
                target_Q = mini_rewards + (-mini_dones + 1) * self.config.gamma * target_Q.detach()
                Q1 = self.critic1(
                    x=mini_states,
                    h_x=mini_history_states,
                    a=mini_actions,
                    h_a=mini_history_actions,
                    feat=mini_features,
                    adj=mini_adjacencys,
                )
                Q2 = self.critic2(
                    x=mini_states,
                    h_x=mini_history_states,
                    a=mini_actions,
                    h_a=mini_history_actions,
                    feat=mini_features,
                    adj=mini_adjacencys,
                )
                # Optimize Critic1
                critic1_loss = F.mse_loss(Q1, target_Q)
                self.critic1_optimizer.zero_grad()
                critic1_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic1.parameters(), self.config.gradient_clip)
                self.critic1_optimizer.step()
                self.critic1_scheduler.step(None)
                # Optimize Critic2
                critic2_loss = F.mse_loss(Q2, target_Q)
                self.critic2_optimizer.zero_grad()
                critic2_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic2.parameters(), self.config.gradient_clip)
                self.critic2_optimizer.step()
                self.critic2_scheduler.step(None)
                # Optimize Actor
                if update_times % self.config.actor_delay_freq:
                    now_action = self.actor(
                        x=mini_states,
                        h_x=mini_history_states,
                        h_a=mini_history_actions,
                        feat=mini_features,
                        adj=mini_adjacencys,
                        action_low=mini_action_lows,
                        action_high=mini_action_highs,
                        sample=False,
                    )
                    actor_loss = - self.critic1(
                        x=mini_states,
                        h_x=mini_history_states,
                        a=now_action,
                        h_a=mini_history_actions,
                        feat=mini_features,
                        adj=mini_adjacencys,
                    ).mean()
                    if self.config.add_balance_loss:
                        actor_loss += self.calculate_balance_loss(mini_states, mini_next_states, now_action, state_norm,
                                                                  mini_actions, mini_dones)
                    self.actor_optimizer.zero_grad()
                    actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.gradient_clip)
                    self.actor_optimizer.step()
                    self.actor_scheduler.step(None)
                    # output res
                    print('actor loss', actor_loss.mean().item())
                    if self.config.output_res:
                        f_learner.write(' actor loss: %f\n' % actor_loss.mean().item())
                        f_learner.flush()
                print('critic1 loss', critic1_loss.mean().item())
                print('critic2 loss', critic2_loss.mean().item())
                if self.config.output_res:
                    f_learner.write('critic1 loss: %f\n' % critic1_loss.mean().item())
                    f_learner.write('critic2 loss: %f\n\n' % critic2_loss.mean().item())
                    f_learner.flush()
        else:
            target_Q1 = self.critic1_target(
                x=next_states,
                h_x=next_history_states,
                a=self.actor_target(
                    x=next_states,
                    h_x=next_history_states,
                    h_a=next_history_actions,
                    feat=next_features,
                    adj=next_adjacencys,
                    action_low=next_action_lows,
                    action_high=next_action_highs,
                    sample=False,
                ),
                h_a=next_history_actions,
                feat=next_features,
                adj=next_adjacencys,
            )
            target_Q2 = self.critic2_target(
                x=next_states,
                h_x=next_history_states,
                a=self.actor_target(
                    x=next_states,
                    h_x=next_history_states,
                    h_a=next_history_actions,
                    feat=next_features,
                    adj=next_adjacencys,
                    action_low=next_action_lows,
                    action_high=next_action_highs,
                    sample=False,
                ),
                h_a=next_history_actions,
                feat=next_features,
                adj=next_adjacencys,
            )
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = rewards + (-dones + 1) * self.config.gamma * target_Q.detach()
            Q1 = self.critic1(
                x=states,
                h_x=history_states,
                a=actions,
                h_a=history_actions,
                feat=features,
                adj=adjacencys,
            )
            Q2 = self.critic2(
                x=states,
                h_x=history_states,
                a=actions,
                h_a=history_actions,
                feat=features,
                adj=adjacencys,
            )
            # Optimize Critic1
            critic1_loss = F.mse_loss(Q1, target_Q)
            self.critic1_optimizer.zero_grad()
            critic1_loss.backward()
            self.critic1_optimizer.step()
            self.critic1_scheduler.step(None)
            # Optimize Critic2
            critic2_loss = F.mse_loss(Q2, target_Q)
            self.critic2_optimizer.zero_grad()
            critic2_loss.backward()
            self.critic2_optimizer.step()
            self.critic2_scheduler.step(None)
            # Optimize Actor
            if update_times % self.config.actor_delay_freq:
                now_action = self.actor(
                    x=states,
                    h_x=history_states,
                    h_a=history_actions,
                    feat=features,
                    adj=adjacencys,
                    action_low=action_lows,
                    action_high=action_highs,
                    sample=False,
                )
                actor_loss = - self.critic1(
                    x=states,
                    h_x=history_states,
                    a=now_action,
                    h_a=history_actions,
                    feat=features,
                    adj=adjacencys,
                ).mean()
                if self.config.add_balance_loss:
                    actor_loss += self.calculate_balance_loss(states, next_states, now_action, state_norm, actions, dones)
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                self.actor_scheduler.step(None)
                # output res
                print('actor loss', actor_loss.mean().item())
                if self.config.output_res:
                    f_learner.write(' actor loss: %f\n' % actor_loss.mean().item())
                    f_learner.flush()
            print('critic1 loss', critic1_loss.mean().item())
            print('critic2 loss', critic2_loss.mean().item())
            if self.config.output_res:
                f_learner.write('critic1 loss: %f\n' % critic1_loss.mean().item())
                f_learner.write('critic2 loss: %f\n\n' % critic2_loss.mean().item())
                f_learner.flush()

        # soft update actor/critic model
        for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.config.soft_tau) + param.data * self.config.soft_tau
            )
        for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.config.soft_tau) + param.data * self.config.soft_tau
            )
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.config.soft_tau) + param.data * self.config.soft_tau
            )

    def load(self, state_norm=None):
        if self.config.load_model:
            save_model = torch.load(self.config.model_load_path, map_location=lambda storage, loc: storage)
            self.actor.load_state_dict(save_model['actor_network'])
            self.actor_target.load_state_dict(save_model['actor_network'])
            self.critic1.load_state_dict(save_model['critic1_network'])
            self.critic1_target.load_state_dict(save_model['critic1_network'])
            self.critic2.load_state_dict(save_model['critic2_network'])
            self.critic2_target.load_state_dict(save_model['critic2_network'])

        if self.config.load_state_normalization and state_norm is not None:
            save_state_normal = torch.load(self.config.state_normal_load_path, map_location=lambda storage, loc: storage)
            state_norm.set_info(save_state_normal['state_norm'])

    def save(self, update_round, checkpoint_path, state_norm=None):
        if not os.path.isdir(self.config.model_save_dir):
            os.makedirs(self.config.model_save_dir)
        torch.save({
            'update_round': update_round,
            'actor_network': self.actor.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic1_network': self.critic1.state_dict(),
            'critic1_optimizer': self.critic1_optimizer.state_dict(),
            'critic2_network': self.critic2.state_dict(),
            'critic2_optimizer': self.critic2_optimizer.state_dict(),
            'state_norm': state_norm.get_info() if state_norm is not None else None,
        }, checkpoint_path)

    def get_buffer_size(self):
        return self.buffer.get_buffer_size()

    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.actor.action_std = self.action_std

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        self.action_std = self.action_std - action_std_decay_rate
        self.action_std = round(self.action_std, 4)
        if self.action_std <= min_action_std:
            self.action_std = min_action_std
        self.actor.action_std = self.action_std

    def decay_balance_loss_rate(self, balance_loss_rate_decay_rate, min_balance_loss_rate):
        self.balance_loss_rate = self.balance_loss_rate - balance_loss_rate_decay_rate
        if self.balance_loss_rate <= min_balance_loss_rate:
            self.balance_loss_rate = min_balance_loss_rate


class TD3Agent(BaseAgent):
    def __init__(self, settings, config):
        BaseAgent.__init__(self, settings)
        # set basic info for PPO agent
        self.settings = settings
        self.config = config
        self.device = config.device
        # set policy
        self.policy = DDPG(settings, config)
        # record data file
        self.f_actor = None
        self.f_learner = None
        # state normalization
        self.state_norm = StateNormalization(self.config)
        # test env state
        self.state_check = StateChecker(self.config)
        # history states buffer
        self.history_info = HistoryInfoBuffer(self.config)
        # graph info process
        self.graph_processer = GraphProcesser(self.config)
        # DT data sampler
        self.dt_data_sampler = None

    def act(self, obs, reward, done=False):
        pass

    def process_action(self, obs, action):
        config = self.config
        # split action
        detail_action_dim = self.config.detail_action_dim
        adjust_gen_p = action[detail_action_dim[0][0]: detail_action_dim[0][1]]
        adjust_gen_v = action[detail_action_dim[1][0]: detail_action_dim[1][1]]
        adjust_adjld_p = action[detail_action_dim[2][0]: detail_action_dim[2][1]]
        adjust_stoenergy_p = action[detail_action_dim[3][0]: detail_action_dim[3][1]]
        # get action limit
        gen_p_action_space = obs.action_space['adjust_gen_p']
        gen_v_action_space = obs.action_space['adjust_gen_v']
        adjld_p_action_space = obs.action_space['adjust_adjld_p']
        stoenergy_p_action_space = obs.action_space['adjust_stoenergy_p']
        # clip action to action space
        before_clip = np.concatenate([adjust_gen_p, adjust_gen_v, adjust_adjld_p, adjust_stoenergy_p], axis=0)
        adjust_gen_p = np.clip(adjust_gen_p, gen_p_action_space.low, gen_p_action_space.high)
        adjust_gen_v = np.clip(adjust_gen_v, gen_v_action_space.low, gen_v_action_space.high)
        adjust_adjld_p = np.clip(adjust_adjld_p, adjld_p_action_space.low, adjld_p_action_space.high)
        adjust_stoenergy_p = np.clip(adjust_stoenergy_p, stoenergy_p_action_space.low, stoenergy_p_action_space.high)
        # set zero for masked action
        if not config.set_gen_p:
            adjust_gen_p = np.zeros_like(adjust_gen_p)
        if not config.set_gen_v:
            adjust_gen_v = np.zeros_like(adjust_gen_v)
        if not config.set_adjld_p:
            adjust_adjld_p = np.zeros_like(adjust_adjld_p)
        if not config.set_stoenergy_p:
            adjust_stoenergy_p = np.zeros_like(adjust_stoenergy_p)
        # interact_action: 与环境交互的动作, 按类别进行索引
        # clip_action: 存入缓冲池的动作数据, 直接进行拼接
        interact_action = form_action(adjust_gen_p, adjust_gen_v, adjust_adjld_p, adjust_stoenergy_p)
        clip_action = np.concatenate([adjust_gen_p, adjust_gen_v, adjust_adjld_p, adjust_stoenergy_p], axis=0)
        # print('------------------------------------- test clip -----------------------------------------')
        # print('clip rate', (action - clip_action != 0).sum() / action.shape[0])

        return interact_action, clip_action

    def process_reward(self, obs, next_obs, reward, info, round, action=None, clip_action=None):
        # if not use reward form environment, mask origin reward from environment
        if not self.config.reward_from_env:
            reward = 0.0
        reward += self.config.reward_for_survive
        if self.config.punish_balance_out_range:
            min_val = settings.min_balanced_gen_bound * settings.gen_p_min[settings.balanced_id]
            max_val = settings.max_balanced_gen_bound * settings.gen_p_max[settings.balanced_id]
            mid_val = (max_val + min_val) / 2
            balance_gen_p = obs.gen_p[settings.balanced_id]
            next_balanced_gen_p = next_obs.gen_p[settings.balanced_id]
            dist = abs(balance_gen_p - mid_val)
            next_dist = abs(next_balanced_gen_p - mid_val)
            # print('------------------------------------')
            # print('ori_reward', reward)
            reward += (dist - next_dist) * self.config.punish_balance_out_range_rate
            # print('dist punish', (dist - next_dist) * self.config.punish_balance_out_range_rate)
            # print('reward', reward)

        return reward

    def process_action_space(self, obs, action_space):
        # get action limit
        gen_p_action_space = action_space['adjust_gen_p']
        gen_v_action_space = action_space['adjust_gen_v']
        adjld_p_action_space = action_space['adjust_adjld_p']
        stoenergy_p_action_space = action_space['adjust_stoenergy_p']
        # concatenate action limit
        action_low = np.concatenate([
            gen_p_action_space.low, gen_v_action_space.low, adjld_p_action_space.low, stoenergy_p_action_space.low
        ], axis=0)
        action_high = np.concatenate([
            gen_p_action_space.high, gen_v_action_space.high, adjld_p_action_space.high, stoenergy_p_action_space.high
        ], axis=0)
        for idx in self.settings.thermal_ids:
            # process thermal on
            if self.config.ban_thermal_on and obs.last_injection_gen_p[idx] == 0.0:
                action_high[idx] = 0.0
                obs.action_space['adjust_gen_p'].high[idx] = 0.0
            # process thermal off
            if self.config.ban_thermal_off and obs.last_injection_gen_p[idx] == self.settings.gen_p_min[idx]:
                action_low[idx] = 0.0
                obs.action_space['adjust_gen_p'].low[idx] = 0.0

        return action_low, action_high

    def my_act(self, env_obs, norm_obs, action_low, action_high,
               history_obs, history_actions, feature, adjacency, sample=True):
        """
        Args:
            env_obs: origin obs from environment
            norm_obs: obs that has been processed
            action_low: lower bound of actionspace
            action_high: upper bound of actionspace
            history_obs: history obs sequence
            history_actions: history actions sequence
            feature: feature matrix for node
            adjacency: adjacency matrix for node
            sample: if sample when generate actions
        Returns:
            interact_action: action interact with environment
                (split by action type)
            action: action value output from actor model
            clip_action: action clip by action space
                (save in data buffer)
        """
        action = self.policy.select_action(norm_obs, action_low, action_high,
                                           history_obs, history_actions,
                                           feature, adjacency, sample)
        interact_action, clip_action = self.process_action(env_obs, action)

        return interact_action, action, clip_action

    def train(self):
        rounds = []
        update_times = 0
        total_steps = 0
        # load pretrain model(and state normal) if enable
        self.policy.load(self.state_norm)
        # set data sampler info
        if self.config.sample_by_train:
            self.dt_data_sampler = DTtDataBuffer(self.config, self.state_norm.get_info())

        # output res file
        if self.config.output_res:
            self.f_actor = open(self.config.res_file_dir + 'actor_' + self.config.res_file_name, 'w')
            self.f_learner = open(self.config.res_file_dir + 'learner_' + self.config.res_file_name, 'w')
            print_config(self.config, self.f_actor)

        for episode in range(self.config.max_episode):
            # set episode info
            rtg = 0.0
            reward = 0.0
            done = False
            print('---------------------------- episode ', episode, '----------------------------')
            if self.config.output_res:
                self.f_actor.write('---------------------------- episode %d ----------------------------\n' % episode)
                self.f_actor.flush()
            # get and reset environment
            env = Environment(self.settings, "EPRIReward")
            obs = env.reset()
            norm_obs = self.state_norm(obs)
            # clear history states buffer at the start of each episode
            self.history_info.clear()
            # add now obs to the history buffer
            self.history_info.add_state(norm_obs)
            # init graph info at the first episode
            if episode == 0:
                self.graph_processer.init_adjacency(obs)
            # update graph info every step
            feature, adjacency = self.graph_processer.update_feature_and_adjacency(obs)
            action_low, action_high = self.process_action_space(obs, obs.action_space)
            # while not done:
            for timestep in range(self.config.max_timestep):
                interact_action, action, clip_action = self.my_act(
                    env_obs=obs, norm_obs=norm_obs,
                    action_low=action_low, action_high=action_high,
                    history_obs=self.history_info.get_history_states(),
                    history_actions=self.history_info.get_history_actions(),
                    feature=feature, adjacency=adjacency,
                    sample=True,
                )
                # add now action to the history buffer
                self.history_info.add_action(clip_action)
                next_obs, reward, done, info = env.step(interact_action)
                next_norm_obs = self.state_norm(next_obs)
                # add next obs to the history buffer
                self.history_info.add_state(next_norm_obs)
                # update graph info every step
                next_feature, next_adjacency = self.graph_processer.update_feature_and_adjacency(next_obs)

                # self.state_check.check_min_state1(obs, clip_action, next_obs)

                next_action_low, next_action_high = self.process_action_space(next_obs, next_obs.action_space)
                env_reward = reward
                reward = self.process_reward(obs, next_obs, reward, info, timestep, action, clip_action)
                rtg += env_reward
                # print('reward', reward)
                # save step data to data buffer
                self.policy.add_data(
                    obs=norm_obs,
                    action=clip_action,
                    logprob=np.zeros_like(clip_action),
                    reward=reward,
                    done=done,
                    next_obs=next_norm_obs,
                    action_low=action_low,
                    action_high=action_high,
                    next_action_low=next_action_low,
                    next_action_high=next_action_high,
                    history_obs=self.history_info.get_last_history_states(),
                    next_history_obs=self.history_info.get_history_states(),
                    history_actions=self.history_info.get_last_history_actions(),
                    next_history_actions=self.history_info.get_history_actions(),
                    feature=feature, adjacency=adjacency,
                    next_feature=next_feature, next_adjacency=next_adjacency,
                )
                # save data to data sampler
                if self.config.sample_by_train:
                    self.dt_data_sampler.add_step(
                        obs=obs,
                        action=clip_action,
                        reward=reward,
                        done=done,
                        action_low=action_low,
                        action_high=action_high,
                    )
                # update every model_update_freq steps
                if self.policy.get_buffer_size() >= self.config.batch_size and \
                        total_steps % self.config.model_update_freq == 0:
                    update_times += 1
                    total_steps %= self.config.model_update_freq
                    self.policy.update(self.f_learner, self.state_norm, update_times)
                    # decay action sample std every interval
                    if update_times % self.config.action_std_decay_freq == 0:
                        self.policy.decay_action_std(self.config.action_std_decay_rate, self.config.min_action_std)
                    # save model every interval update
                    if self.config.save_model and update_times % self.config.model_save_freq == 0:
                        self.policy.save(
                            update_round=update_times,
                            checkpoint_path=self.config.model_save_dir + str(update_times) + '_save_model.pth',
                            state_norm=self.state_norm
                        )
                    # decay balance rate every interval
                    if update_times % self.config.balance_loss_rate_decay_freq == 0:
                        self.policy.decay_balance_loss_rate(self.config.balance_loss_rate_decay_rate,
                                                            self.config.min_balance_loss_rate)
                # add time step before judge finish episode condition
                total_steps += 1
                if done:
                    print('info', info)
                    print('rtg', rtg)
                    print('round', timestep)
                    if self.config.output_res:
                        self.f_actor.write('info ' + str(info) + '\n')
                        self.f_actor.write('rtg %f \n' % rtg)
                        self.f_actor.write('round %d \n' % timestep)
                        self.f_actor.flush()
                    rounds.append(timestep)
                    break
                # update next obs/action bound to now obs/action bound
                obs, norm_obs = next_obs, next_norm_obs
                action_low, action_high = next_action_low, next_action_high
                feature, adjacency = next_feature, next_adjacency
        if self.config.output_res:
            self.f_actor.close()
            self.f_learner.close()

    def evaluate(self, process_idx):
        PowerGrid.objects.all().delete()
        set_seed(123)
        rounds = []
        rtgs = []
        # load pretrain model(and state normal) if enable
        self.policy.load(self.state_norm)

        # output res file
        if self.config.output_res:
            self.f_actor = open(self.config.res_file_dir +
                                'process' + str(process_idx) +
                                '_evaluate_' + self.config.res_file_name, 'w')
            print_config(self.config, self.f_actor)
        evaluate_tqdm = tqdm(enumerate(np.arange(self.config.total_sample_episode)),
                             total=self.config.total_sample_episode, mininterval=1 if self.config.quick_tqdm else 10) \
            if process_idx == 0 else enumerate(np.arange(self.config.total_sample_episode))
        for _, episode in evaluate_tqdm:
            # set episode info
            rtg = 0.0
            reward = 0.0
            done = False
            print('---------------------------- process', process_idx, ' episode ', episode, '----------------------------')
            if self.config.output_res:
                self.f_actor.write('---------------------------- process %d episode %d ----------------------------\n'
                                   % (process_idx, episode))
                self.f_actor.flush()
            # get and reset environment
            env = Environment(self.settings, "EPRIReward")
            obs = env.reset()
            norm_obs = self.state_norm(obs)
            # clear history states buffer at the start of each episode
            self.history_info.clear()
            # add now obs to the history buffer
            self.history_info.add_state(norm_obs)
            # init graph info at the first episode
            if episode == 0:
                self.graph_processer.init_adjacency(obs)
            # update graph info every step
            feature, adjacency = self.graph_processer.update_feature_and_adjacency(obs)
            self.state_check.set_ori_dict(obs)
            action_low, action_high = self.process_action_space(obs, obs.action_space)

            # while not done:
            for timestep in range(self.config.max_timestep):
                interact_action, action, clip_action = self.my_act(
                    env_obs=obs, norm_obs=norm_obs,
                    action_low=action_low, action_high=action_high,
                    history_obs=self.history_info.get_history_states(),
                    history_actions=self.history_info.get_history_actions(),
                    feature=feature, adjacency=adjacency,
                    sample=True,
                )
                time.sleep(speed)
                rgi = PowerGrid(
                    vTime=next_obs.vTime,
                    gen_p=next_obs.gen_p,
                    gen_q=next_obs.gen_q,
                    gen_v=next_obs.gen_v,
                    target_dispatch=next_obs.target_dispatch,
                    actual_dispatch=next_obs.actual_dispatch,
                    ld_p=next_obs.ld_p,
                    adjld_p=next_obs.adjld_p,
                    stoenergy_p=next_obs.stoenergy_p,
                    ld_q=next_obs.ld_q,
                    ld_v=next_obs.ld_v,
                    p_or=next_obs.p_or,
                    q_or=next_obs.q_or,
                    v_or=next_obs.v_or,
                    a_or=next_obs.a_or,
                    p_ex=next_obs.p_ex,
                    q_ex=next_obs.q_ex,
                    v_ex=next_obs.v_ex,
                    a_ex=next_obs.a_ex,
                    line_status=next_obs.line_status,
                    grid_loss=next_obs.grid_loss,
                    bus_v=next_obs.bus_v,
                    bus_gen=next_obs.bus_gen,
                    bus_load=next_obs.bus_load,
                    bus_branch=next_obs.bus_branch,
                    flag=next_obs.flag,
                    unnameindex=next_obs.unnameindex,
                    action_space=next_obs.action_space,
                    steps_to_reconnect_line=next_obs.steps_to_reconnect_line,
                    count_soft_overflow_steps=next_obs.count_soft_overflow_steps,
                    rho=next_obs.rho,
                    gen_status=next_obs.gen_status,
                    steps_to_recover_gen=next_obs.steps_to_recover_gen,
                    steps_to_close_gen=next_obs.steps_to_close_gen,
                    curstep_renewable_gen_p_max=next_obs.curstep_renewable_gen_p_max,
                    nextstep_renewable_gen_p_max=next_obs.nextstep_renewable_gen_p_max
                )
                rgi.save()
                # add now action to the history buffer
                self.history_info.add_action(clip_action)
                next_obs, reward, done, info = env.step(interact_action)
                next_norm_obs = self.state_norm(next_obs)
                # add next obs to the history buffer
                self.history_info.add_state(next_norm_obs)
                # update graph info every step
                next_feature, next_adjacency = self.graph_processer.update_feature_and_adjacency(next_obs)

                # self.state_check.check_state_info(obs, clip_action, next_obs)

                next_action_low, next_action_high = self.process_action_space(next_obs, next_obs.action_space)
                env_reward = reward
                reward = self.process_reward(obs, next_obs, reward, info, timestep, action, clip_action)
                rtg += env_reward
                if done:
                    print('info', info)
                    print('rtg', rtg)
                    print('round', timestep)
                    if self.config.output_res:
                        self.f_actor.write('info ' + str(info) + '\n')
                        self.f_actor.write('rtg %f \n' % rtg)
                        self.f_actor.write('round %d \n' % timestep)
                        self.f_actor.flush()
                    rounds.append(timestep)
                    rtgs.append(rtg)
                    break
                # update next obs/action bound to now obs/action bound
                obs, norm_obs = next_obs, next_norm_obs
                action_low, action_high = next_action_low, next_action_high
                feature, adjacency = next_feature, next_adjacency
        print('mean_rtg', sum(rtgs) / len(rtgs))
        print('mean_round', sum(rounds) / len(rounds))
        if self.config.output_res:
            self.f_actor.write('mean_rtg ' + str(sum(rtgs) / len(rtgs)) + '\n')
            self.f_actor.write('mean_round ' + str(sum(rounds) / len(rounds)) + '\n')
            self.f_actor.close()

    def sample_predict_data(self):
        rounds = []
        rtgs = []
        # load pretrain model(and state normal) if enable
        self.policy.load(self.state_norm)
        predict_data_buffer = PredictDataBuffer(self.config)

        # output res file
        if self.config.output_res:
            self.f_actor = open(self.config.res_file_dir + 'sample_' + self.config.res_file_name, 'w')
            print_config(self.config, self.f_actor)
        for episode in range(self.config.max_episode):
            # set episode info
            rtg = 0.0
            reward = 0.0
            done = False
            finish_sample = False
            print('---------------------------- episode ', episode, '----------------------------')
            if self.config.output_res:
                self.f_actor.write('---------------------------- episode %d ----------------------------\n' % episode)
                self.f_actor.flush()
            # get and reset environment
            env = Environment(self.settings, "EPRIReward")
            obs = env.reset()
            norm_obs = self.state_norm(obs)
            predict_data_buffer.add_step(copy.deepcopy(obs))
            # clear history states buffer at the start of each episode
            self.history_info.clear()
            # add now obs to the history buffer
            self.history_info.add_state(norm_obs)
            action_low, action_high = self.process_action_space(obs, obs.action_space)

            # while not done:
            for timestep in range(self.config.max_timestep):
                interact_action, action, clip_action = self.my_act(
                    env_obs=obs, norm_obs=norm_obs,
                    action_low=action_low, action_high=action_high,
                    history_obs=self.history_info.get_history_states(),
                    history_actions=self.history_info.get_history_actions(),
                    sample=True,
                )
                # add now action to the history buffer
                self.history_info.add_action(clip_action)
                next_obs, reward, done, info = env.step(interact_action)
                next_norm_obs = self.state_norm(next_obs)
                predict_data_buffer.add_step(copy.deepcopy(next_obs))
                # add next obs to the history buffer
                self.history_info.add_state(next_norm_obs)
                next_action_low, next_action_high = self.process_action_space(next_obs, next_obs.action_space)
                env_reward = reward
                reward = self.process_reward(obs, next_obs, reward, info, timestep, action, clip_action)
                rtg += env_reward
                if done:
                    print('info', info)
                    print('rtg', rtg)
                    print('round', timestep)
                    if self.config.output_res:
                        self.f_actor.write('info ' + str(info) + '\n')
                        self.f_actor.write('rtg %f \n' % rtg)
                        self.f_actor.write('round %d \n' % timestep)
                        self.f_actor.flush()
                    rounds.append(timestep)
                    rtgs.append(rtg)
                    # add episode data to sample buffer
                    finish_sample = predict_data_buffer.add_episode()
                    break
                # update next obs/action bound to now obs/action bound
                obs, norm_obs = next_obs, next_norm_obs
                action_low, action_high = next_action_low, next_action_high
            if finish_sample:
                print('finish_sample')
                break
        print('mean_rtg', sum(rtgs) / len(rtgs))
        print('mean_round', sum(rounds) / len(rounds))
        if self.config.output_res:
            self.f_actor.write('mean_rtg ' + str(sum(rtgs) / len(rtgs)) + '\n')
            self.f_actor.write('mean_round ' + str(sum(rounds) / len(rounds)) + '\n')
            self.f_actor.close()

    def test_model(self):
        rounds = []
        total_steps = 0
        all_trajectory_obs = []
        # load pretrain model(and state normal) if enable
        self.policy.load(self.state_norm)

        # output res file
        if self.config.output_res:
            self.f_actor = open(self.config.res_file_dir + 'test_model_res_' + self.config.res_file_name, 'w')

        for episode in range(self.config.total_sample_episode):
            # set episode info
            rtg = 0.0
            reward = 0.0
            done = False
            trajectory_obs = []
            print('---------------------------- episode ', episode, '----------------------------')
            if self.config.output_res:
                self.f_actor.write('---------------------------- episode %d ----------------------------\n' % episode)
                self.f_actor.flush()
            # get and reset environment
            env = Environment(self.settings, "EPRIReward")
            obs = env.reset()
            norm_obs = self.state_norm(obs)
            # add obs to trajectory obs
            trajectory_obs.append(copy.deepcopy(obs))
            # clear history states buffer at the start of each episode
            self.history_info.clear()
            # add now obs to the history buffer
            self.history_info.add_state(norm_obs)
            action_low, action_high = self.process_action_space(obs, obs.action_space)

            # while not done:
            for timestep in range(self.config.max_timestep):
                interact_action, action, clip_action = self.my_act(
                    env_obs=obs, norm_obs=norm_obs,
                    action_low=action_low, action_high=action_high,
                    history_obs=self.history_info.get_history_states(),
                    history_actions=self.history_info.get_history_actions(),
                    sample=False,
                )
                next_obs, reward, done, info = env.step(interact_action)
                next_norm_obs = self.state_norm(next_obs)
                # add obs to trajectory obs
                trajectory_obs.append(copy.deepcopy(next_obs))
                # add next obs to the history buffer
                self.history_info.add_state(next_norm_obs)
                next_action_low, next_action_high = self.process_action_space(next_obs, next_obs.action_space)
                env_reward = reward
                # reward = self.process_reward(obs, next_obs, reward, info, timestep, action, clip_action)
                rtg += env_reward
                # print('reward', reward)
                # add time step before judge finish episode condition
                total_steps += 1
                if done:
                    print('info', info)
                    print('rtg', rtg)
                    print('round', timestep)
                    if rtg > self.config.min_good_rtgs and timestep > self.config.min_good_rounds:
                        all_trajectory_obs.append(select_info_from_obs(trajectory_obs))
                    if self.config.output_res:
                        self.f_actor.write('info ' + str(info) + '\n')
                        self.f_actor.write('rtg %f \n' % rtg)
                        self.f_actor.write('round %d \n' % timestep)
                        self.f_actor.flush()
                    rounds.append(timestep)
                    break
                # update next obs/action bound to now obs/action bound
                obs, norm_obs = next_obs, next_norm_obs
                action_low, action_high = next_action_low, next_action_high
            # end sampling if get enough data
            if len(all_trajectory_obs) >= self.config.total_sample_episode:
                break
        if self.config.output_res:
            self.f_actor.close()
        # save data into file
        if self.config.output_data:
            f_traj = open(self.config.res_file_dir + self.config.trajectory_file_name, 'wb')
            pickle.dump(all_trajectory_obs, f_traj)
