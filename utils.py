# -*- coding: UTF-8 -*-
import os
import json
import copy
import torch
import random
import pickle
import numpy as np
import torch.nn as nn

from torch.utils.data import Dataset
from utilize.settings import settings


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def print_config(config, fw):
    # Training config
    fw.write('buffer_type ' + str(config.buffer_type) + '\n')
    fw.write('max_buffer_size ' + str(config.max_buffer_size) + '\n')
    fw.write('min_state ' + str(config.min_state) + '\n')
    fw.write('model_update_freq ' + str(config.model_update_freq) + '\n')
    fw.write('init_model ' + str(config.init_model) + '\n')
    fw.write('load_model ' + str(config.load_model) + '\n')
    fw.write('load_state_normalization ' + str(config.load_state_normalization) + '\n')
    fw.write('update_state_normalization ' + str(config.update_state_normalization) + '\n')
    fw.write('use_mini_batch ' + str(config.use_mini_batch) + '\n')
    fw.write('use_state_norm ' + str(config.use_state_norm) + '\n')
    fw.write('reflect_actionspace ' + str(config.reflect_actionspace) + '\n')
    fw.write('add_balance_loss ' + str(config.add_balance_loss) + '\n')
    fw.write('balance_loss_rate ' + str(config.balance_loss_rate) + '\n')
    fw.write('balance_loss_rate_decay_rate ' + str(config.balance_loss_rate_decay_rate) + '\n')
    fw.write('balance_loss_rate_decay_freq ' + str(config.balance_loss_rate_decay_freq) + '\n')
    fw.write('min_balance_loss_rate ' + str(config.min_balance_loss_rate) + '\n')
    fw.write('split_balance_loss ' + str(config.split_balance_loss) + '\n')
    fw.write('danger_region_rate ' + str(config.danger_region_rate) + '\n')
    fw.write('save_region_rate ' + str(config.save_region_rate) + '\n')
    fw.write('save_region_balance_loss_rate ' + str(config.save_region_balance_loss_rate) + '\n')
    fw.write('warning_region_balance_loss_rate ' + str(config.warning_region_balance_loss_rate) + '\n')
    fw.write('danger_region_balance_loss_rate ' + str(config.danger_region_balance_loss_rate) + '\n')
    # use history info
    fw.write('use_history_state ' + str(config.use_history_state) + '\n')
    fw.write('use_history_action ' + str(config.use_history_action) + '\n')
    fw.write('history_state_len ' + str(config.history_state_len) + '\n')
    fw.write('gru_num_layers ' + str(config.gru_num_layers) + '\n')
    fw.write('gru_hidden_size ' + str(config.gru_hidden_size) + '\n')
    # use topology info
    fw.write('use_topology_info ' + str(config.use_topology_info) + '\n')
    fw.write('feature_num ' + str(config.feature_num) + '\n')
    fw.write('gcn_hidden_size ' + str(config.gcn_hidden_size) + '\n')
    fw.write('gcn_dropout ' + str(config.gcn_dropout) + '\n')
    # sample data setting
    # fw.write('sample_by_train ' + str(config.sample_by_train) + '\n')
    # fw.write('data_block_size ' + str(config.data_block_size) + '\n')
    # fw.write('min_episode_length ' + str(config.min_episode_length) + '\n')
    fw.write('active_function ' + str(config.active_function) + '\n')
    fw.write('punish_balance_out_range ' + str(config.punish_balance_out_range) + '\n')
    fw.write('punish_balance_out_range_rate ' + str(config.punish_balance_out_range_rate) + '\n')
    fw.write('reward_from_env ' + str(config.reward_from_env) + '\n')
    fw.write('reward_for_survive ' + str(config.reward_for_survive) + '\n')
    # Hipper parameters
    fw.write('lr_actor ' + str(config.lr_actor) + '\n')
    fw.write('lr_critic ' + str(config.lr_critic) + '\n')
    fw.write('lr_decay_step_size ' + str(config.lr_decay_step_size) + '\n')
    fw.write('lr_decay_gamma ' + str(config.lr_decay_gamma) + '\n')
    fw.write('batch_size ' + str(config.batch_size) + '\n')
    fw.write('mini_batch_size ' + str(config.mini_batch_size) + '\n')
    fw.write('gradient_clip ' + str(config.gradient_clip) + '\n')
    fw.write('gamma ' + str(config.gamma) + '\n')
    fw.write('lamb ' + str(config.lamb) + '\n')
    fw.write('eps_clip ' + str(config.eps_clip) + '\n')
    fw.write('init_action_std ' + str(config.init_action_std) + '\n')
    fw.write('action_std_decay_rate ' + str(config.action_std_decay_rate) + '\n')
    fw.write('action_std_decay_freq ' + str(config.action_std_decay_freq) + '\n')
    fw.write('min_action_std ' + str(config.min_action_std) + '\n')
    # Environment settings
    fw.write('ban_prob_disconnection ' + str(config.ban_prob_disconnection) + '\n')
    fw.write('ban_check_gen_status ' + str(config.ban_check_gen_status) + '\n')
    fw.write('ban_thermal_on ' + str(config.ban_thermal_on) + '\n')
    fw.write('ban_thermal_off ' + str(config.ban_thermal_off) + '\n')
    fw.write('restrict_thermal_on_off ' + str(config.restrict_thermal_on_off) + '\n\n')


def _round_p(p):
    dig = settings.keep_decimal_digits
    # return [(round(x * 10 ** dig)) / (10 ** dig) for x in p]
    return [round(x, dig) for x in p]


def init_model(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(module.bias)


def clip_by_tensor(t, t_min, t_max):
    """
    clip_by_tensor
    :param t: tensor
    :param t_min: min
    :param t_max: max
    :return: cliped tensor
    """
    t = t.float()
    t_min = t_min.float()
    t_max = t_max.float()
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max

    return result


def cmp_action_space(obs, action_low, action_high):
    gen_p_action_space = obs.action_space['adjust_gen_p']
    gen_v_action_space = obs.action_space['adjust_gen_v']
    adjld_p_action_space = obs.action_space['adjust_adjld_p']
    stoenergy_p_action_space = obs.action_space['adjust_stoenergy_p']

    now_action_shape_low = np.concatenate([
        gen_p_action_space.low, gen_v_action_space.low, adjld_p_action_space.low, stoenergy_p_action_space.low,
    ], axis=0)

    print('--------------- print shape ---------------')
    print('diff', action_low - now_action_shape_low)


def calculate_balance_loss_data(config):
    balanced_id = settings.balanced_id
    min_balanced_bound = settings.min_balanced_gen_bound
    max_balanced_bound = settings.max_balanced_gen_bound
    min_val = min_balanced_bound * settings.gen_p_min[balanced_id]
    max_val = max_balanced_bound * settings.gen_p_max[balanced_id]
    mid_val = (min_val + max_val) / 2
    # set default attribute for config
    if not hasattr(config, 'danger_region_rate'):
        config.danger_region_rate = 0.1
    if not hasattr(config, 'save_region_rate'):
        config.save_region_rate = 0.2
    # danger region range
    danger_region_lower = (1 - config.danger_region_rate) * settings.gen_p_min[balanced_id]
    danger_region_upper = (1 + config.danger_region_rate) * settings.gen_p_max[balanced_id]
    # warning region range
    warning_region_lower = settings.gen_p_min[balanced_id]
    warning_region_upper = settings.gen_p_max[balanced_id]
    # save region range
    save_region_lower = (1 + config.save_region_rate) * settings.gen_p_min[balanced_id]
    save_region_upper = (1 - config.save_region_rate) * settings.gen_p_max[balanced_id]

    return mid_val, min_val, max_val, \
           danger_region_lower, danger_region_upper, \
           warning_region_lower, warning_region_upper, \
           save_region_lower, save_region_upper


def change_config_in_setting(config):
    settings.prob_disconnection = 0 if config.ban_prob_disconnection else settings.prob_disconnection
    settings.ban_check_gen_status = config.ban_check_gen_status
    settings.ban_legal_check = 0
    settings.ban_check_steps_to_close_gen = 0


def process_state_sum(obs):
    # set generate/load num
    thermal_num = len(settings.thermal_ids)
    renewable_num = len(settings.renewable_ids)
    balanced_num = 1
    simpleld_num = len(settings.simpleld_ids)
    adjld_num = len(settings.adjld_ids)
    stoenergy_num = len(settings.stoenergy_ids)
    simpleld_ids = [idx + thermal_num + renewable_num + balanced_num for idx in settings.simpleld_ids]
    adjld_ids = [idx + thermal_num + renewable_num + balanced_num for idx in settings.adjld_ids]
    stoenergy_ids = [idx + thermal_num + renewable_num + balanced_num for idx in settings.stoenergy_ids]
    # calculate sum value
    thermal_gen_p = torch.sum(obs[:, 0: thermal_num], dim=-1).unsqueeze(-1)
    renewable_gen_p = torch.sum(obs[:, thermal_num:
                                       thermal_num + renewable_num], dim=-1).unsqueeze(-1)
    balance_gen_p = obs[:, thermal_num + renewable_num:
                           thermal_num + renewable_num + balanced_num]
    simpleld_p = torch.sum(obs[:, thermal_num + renewable_num + balanced_num:
                                  thermal_num + renewable_num + balanced_num + simpleld_num], dim=-1).unsqueeze(-1)
    adjld_p = torch.sum(obs[:, thermal_num + renewable_num + balanced_num + simpleld_num:
                               thermal_num + renewable_num + balanced_num + simpleld_num + adjld_num], dim=-1).unsqueeze(-1)
    stoenergy_p = torch.sum(obs[:, thermal_num + renewable_num + balanced_num + simpleld_num + adjld_num:
                                   thermal_num + renewable_num + balanced_num + simpleld_num + adjld_num + stoenergy_num], dim=-1).unsqueeze(-1)
    grid_loss = obs[:, thermal_num + renewable_num + balanced_num + simpleld_num + adjld_num + stoenergy_num].unsqueeze(-1)
    nextstep_adjld_p = torch.sum(obs[:, thermal_num + renewable_num + balanced_num + simpleld_num + adjld_num + stoenergy_num + 1:
                                        thermal_num + renewable_num + balanced_num + simpleld_num + adjld_num + stoenergy_num + 1 + adjld_num], dim=-1).unsqueeze(-1)

    sum_obs = torch.cat([
        thermal_gen_p, renewable_gen_p, balance_gen_p, simpleld_p, adjld_p, stoenergy_p, grid_loss, nextstep_adjld_p
    ], dim=-1)

    return sum_obs


def select_info_from_obs(trajectory_obs):
    obs_info = {
        'thermal_gen_p': np.array([[obs.gen_p[idx] for idx in settings.thermal_ids] for obs in trajectory_obs], dtype=np.float32),
        'renewable_gen_p':  np.array([[obs.gen_p[idx] for idx in settings.renewable_ids] for obs in trajectory_obs], dtype=np.float32),
        'balance_gen_p': np.array([obs.gen_p[settings.balanced_id] for obs in trajectory_obs], dtype=np.float32),
        'simpleld_p':  np.array([[obs.ld_p[idx] for idx in settings.simpleld_ids] for obs in trajectory_obs], dtype=np.float32),
        'adjld_p': np.array([obs.adjld_p for obs in trajectory_obs], dtype=np.float32),
        'stoenergy_p': np.array([obs.stoenergy_p for obs in trajectory_obs], dtype=np.float32),
        'grid_loss': np.array([obs.grid_loss[0] for obs in trajectory_obs], dtype=np.float32) / 0.01,
        'nextstep_adjld_p':  np.array([[obs.nextstep_ld_p[idx] for idx in settings.adjld_ids] for obs in trajectory_obs], dtype=np.float32),
    }
    print('thermal_gen_p', obs_info['thermal_gen_p'].shape)
    print('renewable_gen_p', obs_info['renewable_gen_p'].shape)
    print('balance_gen_p', obs_info['balance_gen_p'].shape)
    print('simpleld_p', obs_info['simpleld_p'].shape)
    print('adjld_p', obs_info['adjld_p'].shape)
    print('stoenergy_p', obs_info['stoenergy_p'].shape)
    print('grid_loss', obs_info['grid_loss'].shape)
    print('nextstep_adjld_p', obs_info['nextstep_adjld_p'].shape)

    return obs_info


def get_discount_rtgs(r, gamma):
    r = np.array(r, dtype=np.float32)
    discount_rtgs = np.zeros_like(r)
    discount_rtgs[-1] = r[-1]
    for t in reversed(range(r.shape[0] - 1)):
        discount_rtgs[t] = r[t] + gamma * discount_rtgs[t + 1]
    sparse_rtgs = np.array([discount_rtgs[0] for _ in range(r.shape[0])])

    return discount_rtgs, sparse_rtgs


def func_normal_distribution(x, mean, sigma):
    return torch.exp(-1 * ((x - mean) ** 2) / (2 * (sigma ** 2)))/(math.sqrt(2 * np.pi) * sigma)


class RolloutBuffer:
    def __init__(self, config):
        self.config = config
        self.states = []
        self.actions = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.mc_rewards = []
        self.done_idxs = []
        self.next_states = []
        self.action_lows = []
        self.action_highs = []
        self.next_action_lows = []
        self.next_action_highs = []
        self.history_states = []
        self.next_history_states = []
        self.history_actions = []
        self.next_history_actions = []

    def add_data(self, obs, action, logprob, reward, done, next_obs=None,
                 action_low=None, action_high=None, next_action_low=None, next_action_high=None,
                 history_obs=None, next_history_obs=None, history_actions=None, next_history_actions=None):
        self.states.append(obs)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.rewards.append(reward)
        self.is_terminals.append(done)
        self.next_states.append(next_obs)
        self.action_lows.append(action_low)
        self.action_highs.append(action_high)
        self.next_action_lows.append(next_action_low)
        self.next_action_highs.append(next_action_high)
        self.history_states.append(history_obs)
        self.next_history_states.append(next_history_obs)
        self.history_actions.append(history_actions)
        self.next_history_actions.append(next_history_actions)

    def clear(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.mc_rewards = []
        self.done_idxs = []
        self.next_states = []
        self.action_lows = []
        self.action_highs = []
        self.next_action_lows = []
        self.next_action_highs = []
        self.history_states = []
        self.next_history_states = []
        self.history_actions = []
        self.next_history_actions = []

    def get_buffer_size(self):
        return len(self.states)

    def get_all_data(self):
        batch_states = torch.tensor(self.states, dtype=torch.float32).to(self.config.device)
        batch_actions = torch.tensor(self.actions, dtype=torch.float32).to(self.config.device)
        batch_rewards = torch.tensor(self.rewards, dtype=torch.float32).unsqueeze(-1).to(self.config.device)
        batch_dones = torch.tensor(self.is_terminals, dtype=torch.float32).unsqueeze(-1).to(self.config.device)
        batch_next_states = torch.tensor(self.next_states, dtype=torch.float32).to(self.config.device)
        batch_logprobs = torch.cat(self.logprobs)
        batch_action_lows = torch.tensor(self.action_lows, dtype=torch.float32).to(self.config.device)
        batch_action_highs = torch.tensor(self.action_highs, dtype=torch.float32).to(self.config.device)
        batch_next_action_lows = torch.tensor(self.next_action_lows, dtype=torch.float32).to(self.config.device)
        batch_next_action_highs = torch.tensor(self.next_action_highs, dtype=torch.float32).to(self.config.device)
        # batch_history_states = torch.tensor(self.history_states).to(self.config.device)
        # batch_next_history_states = torch.tensor(self.next_history_states).to(self.config.device)
        # batch_history_actions = torch.tensor(self.history_actions).to(self.config.device)
        # batch_next_history_actions = torch.tensor(self.next_history_actions).to(self.config.device)
        self.clear()

        return batch_states, batch_actions, batch_rewards, batch_dones, batch_next_states, batch_logprobs, \
               batch_action_lows, batch_action_highs, batch_next_action_lows, batch_next_action_highs


class FIFOBuffer:
    def __init__(self, config):
        self.size = 0
        self.count = 0
        self.config = config
        self.states = np.zeros((self.config.max_buffer_size, self.config.state_dim), dtype=np.float32)
        self.actions = np.zeros((self.config.max_buffer_size, self.config.action_dim), dtype=np.float32)
        self.logprobs = np.zeros((self.config.max_buffer_size, self.config.action_dim), dtype=np.float32)
        self.rewards = np.zeros(self.config.max_buffer_size, dtype=np.float32)
        self.is_terminals = np.zeros(self.config.max_buffer_size, dtype=np.bool)
        self.next_states = np.zeros((self.config.max_buffer_size, self.config.state_dim), dtype=np.float32)
        self.action_lows = np.zeros((self.config.max_buffer_size, self.config.action_dim), dtype=np.float32)
        self.action_highs = np.zeros((self.config.max_buffer_size, self.config.action_dim), dtype=np.float32)
        self.next_action_lows = np.zeros((self.config.max_buffer_size, self.config.action_dim), dtype=np.float32)
        self.next_action_highs = np.zeros((self.config.max_buffer_size, self.config.action_dim), dtype=np.float32)
        self.history_states = np.zeros(
            (self.config.max_buffer_size, self.config.history_state_len, self.config.state_dim), dtype=np.float32)
        self.next_history_states = np.zeros(
            (self.config.max_buffer_size, self.config.history_state_len, self.config.state_dim), dtype=np.float32)
        self.history_actions = np.zeros(
            (self.config.max_buffer_size, self.config.history_state_len, self.config.action_dim), dtype=np.float32)
        self.next_history_actions = np.zeros(
            (self.config.max_buffer_size, self.config.history_state_len, self.config.action_dim), dtype=np.float32)
        self.features = np.zeros(
            (self.config.max_buffer_size, settings.gen_num + settings.ld_num, self.config.feature_num), dtype=np.float32)
        self.adjacencys = np.zeros(
            (self.config.max_buffer_size, settings.gen_num + settings.ld_num, settings.gen_num + settings.ld_num), dtype=np.float32)
        self.next_features = np.zeros(
            (self.config.max_buffer_size, settings.gen_num + settings.ld_num, self.config.feature_num), dtype=np.float32)
        self.next_adjacencys = np.zeros(
            (self.config.max_buffer_size, settings.gen_num + settings.ld_num, settings.gen_num + settings.ld_num), dtype=np.float32)

    def add_data(self, obs, action, logprob, reward, done, next_obs=None,
                 action_low=None, action_high=None, next_action_low=None, next_action_high=None,
                 history_obs=None, next_history_obs=None, history_actions=None, next_history_actions=None,
                 feature=None, adjacency=None, next_feature=None, next_adjacency=None):
        self.states[self.count] = obs
        self.actions[self.count] = action
        self.logprobs[self.count] = logprob
        self.rewards[self.count] = reward
        self.is_terminals[self.count] = done
        self.next_states[self.count] = next_obs
        self.action_lows[self.count] = action_low
        self.action_highs[self.count] = action_high
        self.next_action_lows[self.count] = next_action_low
        self.next_action_highs[self.count] = next_action_high
        self.history_states[self.count] = history_obs
        self.next_history_states[self.count] = next_history_obs
        self.history_actions[self.count] = history_actions
        self.next_history_actions[self.count] = next_history_actions
        self.features[self.count] = feature
        self.adjacencys[self.count] = adjacency
        self.next_features[self.count] = next_feature
        self.next_adjacencys[self.count] = next_adjacency
        self.size = min(self.size + 1, self.config.max_buffer_size)
        self.count = (self.count + 1) % self.config.max_buffer_size

    def clear(self):
        self.size = 0
        self.count = 0
        self.states = np.zeros((self.config.max_buffer_size, self.config.state_dim), dtype=np.float32)
        self.actions = np.zeros((self.config.max_buffer_size, self.config.action_dim), dtype=np.float32)
        self.logprobs = np.zeros(self.config.max_buffer_size, dtype=np.float32)
        self.rewards = np.zeros(self.config.max_buffer_size, dtype=np.float32)
        self.is_terminals = np.zeros(self.config.max_buffer_size, dtype=np.bool)
        self.next_states = np.zeros((self.config.max_buffer_size, self.config.state_dim), dtype=np.float32)
        self.action_lows = np.zeros((self.config.max_buffer_size, self.config.action_dim), dtype=np.float32)
        self.action_highs = np.zeros((self.config.max_buffer_size, self.config.action_dim), dtype=np.float32)
        self.next_action_lows = np.zeros((self.config.max_buffer_size, self.config.action_dim), dtype=np.float32)
        self.next_action_highs = np.zeros((self.config.max_buffer_size, self.config.action_dim), dtype=np.float32)
        self.history_states = np.zeros(
            (self.config.max_buffer_size, self.config.history_state_len, self.config.state_dim), dtype=np.float32)
        self.next_history_states = np.zeros(
            (self.config.max_buffer_size, self.config.history_state_len, self.config.state_dim), dtype=np.float32)
        self.history_actions = np.zeros(
            (self.config.max_buffer_size, self.config.history_state_len, self.config.action_dim), dtype=np.float32)
        self.next_history_actions = np.zeros(
            (self.config.max_buffer_size, self.config.history_state_len, self.config.action_dim), dtype=np.float32)
        self.features = np.zeros(
            (self.config.max_buffer_size, settings.gen_num + settings.ld_num, self.config.feature_num), dtype=np.float32)
        self.adjacencys = np.zeros(
            (self.config.max_buffer_size, settings.gen_num + settings.ld_num, settings.gen_num + settings.ld_num), dtype=np.float32)
        self.next_features = np.zeros(
            (self.config.max_buffer_size, settings.gen_num + settings.ld_num, self.config.feature_num), dtype=np.float32)
        self.next_adjacencys = np.zeros(
            (self.config.max_buffer_size, settings.gen_num + settings.ld_num, settings.gen_num + settings.ld_num), dtype=np.float32)

    def get_buffer_size(self):
        return self.size

    def sample_data(self):
        device = self.config.device
        index = np.random.choice(self.size, size=self.config.batch_size)  # Randomly sampling
        batch_states = torch.tensor(self.states[index], dtype=torch.float32).to(device)
        batch_actions = torch.tensor(self.actions[index], dtype=torch.float32).to(device)
        batch_rewards = torch.tensor(self.rewards[index], dtype=torch.float32).unsqueeze(-1).to(device)
        batch_dones = torch.tensor(self.is_terminals[index], dtype=torch.float32).unsqueeze(-1).to(device)
        batch_next_states = torch.tensor(self.next_states[index], dtype=torch.float32).to(device)
        batch_logprobs = torch.tensor(self.logprobs[index], dtype=torch.float32).to(device)
        batch_action_lows = torch.tensor(self.action_lows[index], dtype=torch.float32).to(device)
        batch_action_highs = torch.tensor(self.action_highs[index], dtype=torch.float32).to(device)
        batch_next_action_lows = torch.tensor(self.next_action_lows[index], dtype=torch.float32).to(device)
        batch_next_action_highs = torch.tensor(self.next_action_highs[index], dtype=torch.float32).to(device)
        batch_history_states = torch.tensor(self.history_states[index], dtype=torch.float32).to(device)
        batch_next_history_states = torch.tensor(self.next_history_states[index], dtype=torch.float32).to(device)
        batch_history_actions = torch.tensor(self.history_actions[index], dtype=torch.float32).to(device)
        batch_next_history_actions = torch.tensor(self.next_history_actions[index], dtype=torch.float32).to(device)
        batch_features = torch.tensor(self.features[index], dtype=torch.float32).to(device)
        batch_adjacencys = torch.tensor(self.adjacencys[index], dtype=torch.float32).to(device)
        batch_next_features = torch.tensor(self.next_features[index], dtype=torch.float32).to(device)
        batch_next_adjacencys = torch.tensor(self.next_adjacencys[index], dtype=torch.float32).to(device)

        return batch_states, batch_actions, batch_rewards, batch_dones, batch_next_states, batch_logprobs, \
               batch_action_lows, batch_action_highs, batch_next_action_lows, batch_next_action_highs, \
               batch_history_states, batch_next_history_states, batch_history_actions, batch_next_history_actions, \
               batch_features, batch_adjacencys, batch_next_features, batch_next_adjacencys


class HistoryInfoBuffer:
    def __init__(self, config):
        self.config = config
        self.history_states = []
        self.history_actions = []

    def add_state(self, obs):
        self.history_states.append(obs)

    def add_action(self, action):
        self.history_actions.append(action)

    def clear(self):
        self.history_states = []
        self.history_actions = []

    def get_last_history_states(self):
        upper_bound = len(self.history_states) - 1
        lower_bound = max(0, upper_bound - self.config.history_state_len)
        history_states = np.array(self.history_states[lower_bound: upper_bound], dtype=np.float32)
        # print('last history_states', history_states.shape)
        history_states = np.concatenate([history_states,
                                         np.zeros((self.config.history_state_len - history_states.shape[0],
                                                   self.config.state_dim), dtype=np.float32)], axis=0)

        return history_states

    def get_last_history_actions(self):
        upper_bound = max(0, len(self.history_actions) - 1)
        lower_bound = max(0, upper_bound - self.config.history_state_len)
        history_actions = np.array(self.history_actions[lower_bound: upper_bound], dtype=np.float32)
        # print('last history_actions', history_actions.shape)
        history_actions = np.concatenate([history_actions,
                                          np.zeros((self.config.history_state_len - history_actions.shape[0],
                                                    self.config.action_dim), dtype=np.float32)], axis=0) \
            if history_actions.shape[0] > 0 else np.zeros((self.config.history_state_len,
                                                           self.config.action_dim), dtype=np.float32)

        return history_actions

    def get_history_states(self):
        upper_bound = len(self.history_states)
        lower_bound = max(0, upper_bound - self.config.history_state_len)
        history_states = np.array(self.history_states[lower_bound: upper_bound], dtype=np.float32)
        # print('history_states', history_states.shape)
        history_states = np.concatenate([history_states,
                                         np.zeros((self.config.history_state_len - history_states.shape[0],
                                                   self.config.state_dim), dtype=np.float32)], axis=0)

        return history_states

    def get_history_actions(self):
        upper_bound = len(self.history_actions)
        lower_bound = max(0, upper_bound - self.config.history_state_len)
        history_actions = np.array(self.history_actions[lower_bound: upper_bound], dtype=np.float32)
        # print('history_actions', history_actions.shape)
        history_actions = np.concatenate([history_actions,
                                          np.zeros((self.config.history_state_len - history_actions.shape[0],
                                                    self.config.action_dim), dtype=np.float32)], axis=0) \
            if history_actions.shape[0] > 0 else np.zeros((self.config.history_state_len,
                                                           self.config.action_dim), dtype=np.float32)

        return history_actions


class PredictDataBuffer:
    def __init__(self, config):
        self.config = config
        self.episode_data = []
        self.step_data = []
        self.block_id = 0
        self.total_episode = 0

    def add_step(self, obs):
        self.step_data.append(obs)

    def add_episode(self):
        # judge and add long episode data to episode data
        if len(self.step_data) >= self.config.min_good_rounds:
            self.episode_data.append(select_info_from_obs(self.step_data)['simpleld_p'])
            # save data into data block if sample enough episode
            if len(self.episode_data) >= self.config.sample_block_size:
                if not os.path.isdir(self.config.sample_data_file_dir):
                    os.makedirs(self.config.sample_data_file_dir)
                f_traj = open(self.config.sample_data_file_dir + 'block' +
                              str(self.block_id) + '_' +
                              self.config.sample_data_file_name, 'wb')
                pickle.dump(
                    {'data': self.episode_data, 'len': [episode.shape[0] for episode in self.episode_data]},
                    f_traj
                )
                print('len', [episode.shape[0] for episode in self.episode_data])
                self.total_episode += len(self.episode_data)
                self.episode_data = []
                self.block_id += 1
        self.step_data = []
        return self.total_episode >= self.config.total_sample_episode


class DTtDataBuffer:
    def __init__(self, config, info):
        self.config = config
        self.info = info
        self.episode_data = {
            'data': [],
            'info': {
                'traj_lens': [],
                'traj_rtgs': [],
                'state_mean': self.info['mean'],
                'state_std': self.info['std'],
            },
        }
        self.step_data = {
            'states': [],
            'actions': [],
            'rewards': [],
            'terminals': [],
            'action_lows': [],
            'action_highs': [],
            'timesteps': [],
            'rtgs': [],
            'sparse_rtgs': [],
        }
        self.block_id = 0
        self.total_steps = 0

    def process_obs(self, obs):
        thermal_gen_p = np.array([obs.gen_p[idx] for idx in settings.thermal_ids], dtype=np.float32)
        renewable_gen_p = np.array([obs.gen_p[idx] for idx in settings.renewable_ids], dtype=np.float32)
        balance_gen_p = np.array([obs.gen_p[settings.balanced_id]], dtype=np.float32)
        adjld_p = np.array(obs.adjld_p, dtype=np.float32)
        stoenergy_p = np.array(obs.stoenergy_p, dtype=np.float32)
        simpleld_p = np.array([obs.ld_p[idx] for idx in settings.simpleld_ids], dtype=np.float32)
        grid_loss = np.array(obs.grid_loss, dtype=np.float32) / 0.01
        nextstep_adjld_p = np.array([obs.nextstep_ld_p[idx] for idx in settings.adjld_ids], dtype=np.float32)
        state = np.concatenate([thermal_gen_p, renewable_gen_p, balance_gen_p,
                                     simpleld_p, adjld_p, stoenergy_p, grid_loss, nextstep_adjld_p], axis=0)
        # padding thermal state to obs if restrict_thermal_on_off
        if self.config.restrict_thermal_on_off:
            thermal_states = []
            for idx in settings.thermal_ids:
                # normal close state
                if obs.gen_p[idx] == 0.0 and obs.steps_to_recover_gen[idx] != 0:
                    thermal_states.extend([0.0, 0.0])
                # critical open state
                elif obs.gen_p[idx] == 0.0 and obs.steps_to_recover_gen[idx] == 0:
                    thermal_states.extend([0.0, 1.0])
                # critical close state
                elif obs.gen_p[idx] == settings.gen_p_min[idx] and obs.steps_to_close_gen[idx] == 0:
                    thermal_states.extend([1.0, 0.0])
                # normal open state
                else:
                    thermal_states.extend([1.0, 1.0])
            state = np.concatenate([state, np.array(thermal_states, dtype=np.float32)], axis=0)

        return state

    def add_step(self, obs, action, reward, done, action_low, action_high):
        self.step_data['states'].append(self.process_obs(obs))
        self.step_data['actions'].append(action)
        self.step_data['rewards'].append(reward)
        self.step_data['terminals'].append(done)
        self.step_data['action_lows'].append(action_low)
        self.step_data['action_highs'].append(action_high)
        if done:
            # calculate timesteps and rtgs
            self.step_data['timesteps'] = list(np.arange(len(self.step_data['states'])))
            rtgs, sparse_rtgs = get_discount_rtgs(self.step_data['rewards'], 1.0)
            self.step_data['rtgs'] = list(rtgs)
            self.step_data['sparse_rtgs'] = list(sparse_rtgs)
            self.add_episode()

    def add_episode(self):
        # finish an episode and save it to episode buffer
        if len(self.step_data['states']) >= self.config.min_episode_length:
            self.episode_data['data'].append(copy.deepcopy(self.step_data))
            self.episode_data['info']['traj_lens'].append(len(self.step_data['states']))
            self.episode_data['info']['traj_rtgs'].append(self.step_data['rtgs'][0])
            self.total_steps += len(self.step_data['states'])
            # save data into data block if sample enough steps
            if self.total_steps >= self.config.data_block_size:
                if not os.path.isdir(self.config.sample_data_dir):
                    os.makedirs(self.config.sample_data_dir)
                f_traj = open(self.config.sample_data_dir + 'block' +
                              str(self.block_id) + '_dt_episode_data', 'wb')
                pickle.dump(self.episode_data, f_traj)
                # clear episode data
                self.episode_data = {
                    'data': [],
                    'info': {
                        'traj_lens': [],
                        'traj_rtgs': [],
                        'state_mean': self.info['mean'],
                        'state_std': self.info['std'],
                    },
                }
                # update block id and reset total step
                self.block_id += 1
                self.total_steps = 0
        # clear step data
        self.step_data = {
            'states': [],
            'actions': [],
            'rewards': [],
            'terminals': [],
            'action_lows': [],
            'action_highs': [],
            'timesteps': [],
            'rtgs': [],
            'sparse_rtgs': [],
        }


class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)


class StateNormalization:
    def __init__(self, config):
        self.config = config
        self.running_ms = RunningMeanStd(shape=config.state_dim)

    def __call__(self, obs, changeType=True):
        # collect all needed state from obs
        if changeType and self.config.min_state == 1:
            thermal_gen_p = np.array([obs.gen_p[idx] for idx in settings.thermal_ids], dtype=np.float32)
            renewable_gen_p = np.array([obs.gen_p[idx] for idx in settings.renewable_ids], dtype=np.float32)
            balance_gen_p = np.array([obs.gen_p[settings.balanced_id]], dtype=np.float32)
            adjld_p = np.array(obs.adjld_p, dtype=np.float32)
            stoenergy_p = np.array(obs.stoenergy_p, dtype=np.float32)
            simpleld_p = np.array([obs.ld_p[idx] for idx in settings.simpleld_ids], dtype=np.float32)
            grid_loss = np.array(obs.grid_loss, dtype=np.float32) / 0.01
            nextstep_adjld_p = np.array([obs.nextstep_ld_p[idx] for idx in settings.adjld_ids], dtype=np.float32)
            origin_obs = np.concatenate([thermal_gen_p, renewable_gen_p, balance_gen_p,
                                         simpleld_p, adjld_p, stoenergy_p, grid_loss, nextstep_adjld_p], axis=0)
        elif changeType and self.config.min_state == 2:
            thermal_gen_p = sum([obs.gen_p[idx] for idx in settings.thermal_ids])
            renewable_gen_p = sum([obs.gen_p[idx] for idx in settings.renewable_ids])
            balance_gen_p = obs.gen_p[settings.balanced_id]
            adjld_p = sum(obs.adjld_p)
            stoenergy_p = sum(obs.stoenergy_p)
            simpleld_p = sum(obs.ld_p) - adjld_p - stoenergy_p
            grid_loss = sum(obs.grid_loss) / 0.01
            nextstep_adjld_p = sum([obs.nextstep_ld_p[idx] for idx in settings.adjld_ids])
            origin_obs = np.array([thermal_gen_p, renewable_gen_p, balance_gen_p,
                                   simpleld_p, adjld_p, stoenergy_p, grid_loss, nextstep_adjld_p], dtype=np.float32)
        else:
            origin_obs = obs

        # padding thermal state to obs if restrict_thermal_on_off
        if self.config.restrict_thermal_on_off:
            thermal_states = []
            for idx in settings.thermal_ids:
                # normal close state
                if obs.gen_p[idx] == 0.0 and obs.steps_to_recover_gen[idx] != 0:
                    thermal_states.extend([0.0, 0.0])
                # critical open state
                elif obs.gen_p[idx] == 0.0 and obs.steps_to_recover_gen[idx] == 0:
                    thermal_states.extend([0.0, 1.0])
                # critical close state
                elif obs.gen_p[idx] == settings.gen_p_min[idx] and obs.steps_to_close_gen[idx] == 0:
                    thermal_states.extend([1.0, 0.0])
                # normal open state
                else:
                    thermal_states.extend([1.0, 1.0])
            origin_obs = np.concatenate([origin_obs, np.array(thermal_states, dtype=np.float32)], axis=0)
        if self.config.update_state_normalization:
            self.running_ms.update(origin_obs)
        normal_obs = (origin_obs - self.running_ms.mean) / (self.running_ms.std + 1e-8) \
            if self.config.use_state_norm else origin_obs
        # print('-------------------------------------')
        # print('origin_obs mean', np.mean(origin_obs))
        # print('normal_obs mean', np.mean(normal_obs))
        # print('origin_obs max', np.max(origin_obs))
        # print('normal_obs max', np.max(normal_obs))

        return normal_obs

    def inverse_normalization(self, obs):
        ori_obs = obs * (torch.tensor(self.running_ms.std + 1e-8, dtype=torch.float32).to(self.config.device)) + \
                  torch.tensor(self.running_ms.mean, dtype=torch.float32).to(self.config.device)

        return ori_obs

    def inverse_single_normalization(self, obs):
        ori_obs = obs * self.running_ms.std + 1e-8 + self.running_ms.mean

        return ori_obs

    def process_action_space(self, action_space):
        return np.concatenate([
            action_space['adjust_gen_p'].low, action_space['adjust_gen_p'].high,
            action_space['adjust_gen_v'].low, action_space['adjust_gen_v'].high,
            action_space['adjust_adjld_p'].low, action_space['adjust_adjld_p'].high,
            action_space['adjust_stoenergy_p'].low, action_space['adjust_stoenergy_p'].high,
        ], axis=0)

    def get_info(self):
        return {
            'n': self.running_ms.n,
            'mean': self.running_ms.mean,
            'S': self.running_ms.S,
            'std': self.running_ms.std,
        }

    def set_info(self, save_info):
        self.running_ms.n = save_info['n']
        self.running_ms.mean = save_info['mean']
        self.running_ms.S = save_info['S']
        self.running_ms.std = save_info['std']
        # if restrict_thermal_on_off
        # padding N(0, 1) normalization vector for padding state
        if self.config.restrict_thermal_on_off:
            self.running_ms.mean = np.concatenate([self.running_ms.mean,
                                                   np.zeros(2 * len(settings.thermal_ids), dtype=np.float32)], axis=0)
            self.running_ms.S = np.concatenate([self.running_ms.S,
                                                np.ones(2 * len(settings.thermal_ids), dtype=np.float32) * self.running_ms.n], axis=0)
            self.running_ms.std = np.concatenate([self.running_ms.std,
                                                  np.ones(2 * len(settings.thermal_ids), dtype=np.float32)], axis=0)

    def print_info(self):
        print('n', self.running_ms.n)
        print('mean', self.running_ms.mean.shape)
        print('S', self.running_ms.S.shape)
        print('std', self.running_ms.std.shape)


class RewardScaling:
    def __init__(self, shape, gamma):
        self.shape = shape  # reward shape=1
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)

    def __call__(self, x, update=True):
        self.R = self.gamma * self.R + x
        if update:
            self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)  # Only divided std
        return x

    def reset(self):  # When an episode is done,we should reset 'self.R'
        self.R = np.zeros(self.shape)


class GraphProcesser:
    def __init__(self, config):
        self.config = config
        self.num_node = settings.gen_num + settings.ld_num
        self.feature = None
        self.path = None
        self.ori_path = None
        self.adjacency = None
        self.ori_adjacency = None
        # tool info
        self.name_to_index = {}
        self.index_to_name = {}
        self.branch_to_bus = {}
        self.bus_to_node = {}
        self.type_feature = None
        self.name_to_index, self.index_to_name = self.map_name_index()

    def map_name_index(self):
        name_to_index = {}
        index_to_name = {}
        idx = 0
        for gen_name in settings.gen_name_list:
            name_to_index[gen_name] = idx
            index_to_name[idx] = gen_name
            idx += 1
        for ld_name in settings.ld_name:
            name_to_index[ld_name] = idx
            index_to_name[idx] = ld_name
            idx += 1

        return name_to_index, index_to_name

    def map_branch_to_bus(self, init_obs):
        self.branch_to_bus = {}
        for bus_name in init_obs.bus_branch:
            branch_list = init_obs.bus_branch[bus_name]
            for branch_port in branch_list:
                name, port = branch_port[0:-3], branch_port[-2:]
                if name in self.branch_to_bus:
                    self.branch_to_bus[name][port] = bus_name
                else:
                    self.branch_to_bus[name] = {port: bus_name}

    def init_type_feature(self):
        self.type_feature = copy.deepcopy(settings.gen_type)
        ld_type = list(np.array(settings.ld_type) + 3)
        for i in range(len(self.type_feature)):
            if self.type_feature[i] == 5:
                self.type_feature[i] = 0
        self.type_feature.extend(ld_type)
        self.type_feature = np.array(self.type_feature, dtype=np.float32)

    def init_adjacency(self, init_obs):
        # init type feature for node
        self.init_type_feature()
        # get map between bus and branch
        self.map_branch_to_bus(init_obs)
        # init zero matrix for origin path matrix
        self.ori_path = np.zeros((self.num_node, self.num_node), dtype=np.float32)
        # level one adjacency
        for key in init_obs.bus_gen:
            # remove empty name string if list is empty
            if '' in init_obs.bus_gen[key]:
                init_obs.bus_gen[key].remove('')
            if '' in init_obs.bus_load[key]:
                init_obs.bus_load[key].remove('')
            node_list = init_obs.bus_gen[key]
            node_list = node_list + init_obs.bus_load[key]
            # map bus to node
            self.bus_to_node[key] = node_list
            for n1 in node_list:
                for n2 in node_list:
                    self.ori_path[self.name_to_index[n1]][self.name_to_index[n2]] += 1
        # level two adjacency
        for key in self.branch_to_bus:
            bus_n1 = self.branch_to_bus[key]['ex']
            bus_n2 = self.branch_to_bus[key]['or']
            node_list = self.bus_to_node[bus_n1]
            node_list1 = self.bus_to_node[bus_n2]
            node_list = node_list + node_list1
            for n1 in node_list:
                for n2 in node_list:
                    self.ori_path[self.name_to_index[n1]][self.name_to_index[n2]] += 1
        # get access matrix from path matrix
        self.ori_adjacency = np.clip(self.ori_path, 0.0, 1.0)
        self.path = self.ori_path
        self.adjacency = self.ori_adjacency

        return None

    def update_feature_and_adjacency(self, obs):
        # update feature
        node_type = self.type_feature.reshape(1, -1)
        node_p = np.concatenate([
            np.array(obs.gen_p, dtype=np.float32),
            np.array(obs.ld_p, dtype=np.float32),
        ], axis=0).reshape(1, -1)
        node_q = np.concatenate([
            np.array(obs.gen_q, dtype=np.float32),
            np.array(obs.ld_q, dtype=np.float32),
        ], axis=0).reshape(1, -1)
        node_v = np.concatenate([
            np.array(obs.gen_v, dtype=np.float32),
            np.array(obs.ld_v, dtype=np.float32),
        ], axis=0).reshape(1, -1)
        self.feature = np.concatenate([node_type, node_p, node_q, node_v], axis=0)
        self.feature = self.feature.reshape(self.num_node, self.config.feature_num)
        # update adjacency
        ## process disconnect event
        for disc_branch in obs.disc_name:
            bus_n1 = self.branch_to_bus[disc_branch]['ex']
            bus_n2 = self.branch_to_bus[disc_branch]['or']
            node_list = self.bus_to_node[bus_n1]
            node_list1 = self.bus_to_node[bus_n2]
            node_list = node_list + node_list1
            for n1 in node_list:
                for n2 in node_list:
                    self.path[self.name_to_index[n1]][self.name_to_index[n2]] -= 1
        ## process connect event
        for disc_branch in obs.recover_name:
            bus_n1 = self.branch_to_bus[disc_branch]['ex']
            bus_n2 = self.branch_to_bus[disc_branch]['or']
            node_list = self.bus_to_node[bus_n1]
            node_list1 = self.bus_to_node[bus_n2]
            node_list = node_list + node_list1
            for n1 in node_list:
                for n2 in node_list:
                    self.path[self.name_to_index[n1]][self.name_to_index[n2]] += 1
        # update adjacency matrix
        self.adjacency = np.clip(self.path, 0.0, 1.0)

        return copy.deepcopy(self.feature), copy.deepcopy(self.adjacency)


class StateActionDataset:
    def __init__(self, config, states, actions):
        self.config = config
        self.states = [self.form_states(state) for state in states]
        self.actions = actions
        self.state_norm = StateNormalization(config)

    def form_states(self, obs):
        # obd states
        gens = np.concatenate([obs['gen_p'], obs['gen_q'], obs['gen_v'],
                               obs['target_dispatch'], obs['actual_dispatch']], axis=0)
        # load states
        loads = np.concatenate([obs['ld_p'], obs['adjld_p'], obs['stoenergy_p'],
                                obs['ld_q'], obs['ld_v']], axis=0)
        # line states
        lines = np.concatenate([obs['p_or'], obs['q_or'], obs['v_or'], obs['a_or'],
                                obs['p_ex'], obs['q_ex'], obs['v_ex'], obs['a_ex'],
                                obs['line_status'], obs['grid_loss'], obs['bus_v'],
                                obs['steps_to_reconnect_line'], obs['count_soft_overflow_steps'],
                                obs['rho']], axis=0)
        # other info
        other = np.concatenate([obs['gen_status'], obs['steps_to_recover_gen'], obs['steps_to_close_gen'],
                                obs['curstep_renewable_gen_p_max'], obs['nextstep_renewable_gen_p_max'],
                                obs['curstep_ld_p'], obs['nextstep_ld_p'],
                                obs['total_adjld'], obs['total_stoenergy']], axis=0)
        origin_obs = np.concatenate([gens, loads, lines, other])

        return origin_obs

    def get_batch(self):
        idxs = np.random.choice(
            np.arange(len(self.states)),
            size=self.config.batch_size,
            replace=True
        )
        """
            actions torch.Size([128, 123])
            states torch.Size([128, 3485])
        """
        states = torch.from_numpy(
            np.array([self.state_norm(self.states[idx], changeType=False) for idx in idxs], dtype=np.float32))
        actions = torch.from_numpy(np.array([self.actions[idx] for idx in idxs], dtype=np.float32))

        return states, actions


class StateChecker:
    def __init__(self, config):
        self.config = config
        self.bus_gen = None
        self.bus_load = None
        self.bus_branch = None

    def check_state_change(self, state, action, next_state):
        gen_p = np.array(state.gen_p)
        next_gen_p = np.array(next_state.gen_p)
        np.array(state.adjld_p)
        nextstep_adjld_p = np.array([state.nextstep_ld_p[idx] for idx in settings.adjld_ids])
        adjust_gen_p = action['adjust_gen_p']
        adjust_gen_v = action['adjust_gen_v']
        adjust_adjld_p = action['adjust_adjld_p']
        adjust_stoenergy_p = action['adjust_stoenergy_p']

        next_adjld_p = np.array(next_state.adjld_p)
        next_stoenergy_p = np.array(next_state.stoenergy_p)

        # print('----------------------- check state change ------------------------------')
        # print('gen_p', _round_p(adjust_gen_p + gen_p - next_gen_p))
        # print('adjld_p', _round_p(adjust_adjld_p + nextstep_adjld_p - next_adjld_p))
        # print('stoenergy_p', _round_p(next_stoenergy_p - adjust_stoenergy_p))

    def check_state_change_new(self, norm_obs, clip_action, next_norm_obs, state_norm):
        detail_action_dim = self.config.detail_action_dim
        ori_obs = state_norm.inverse_single_normalization(norm_obs)
        next_ori_obs = state_norm.inverse_single_normalization(next_norm_obs)

        thermal_gen_p, renewable_gen_p = ori_obs[0], ori_obs[1]
        next_thermal_gen_p, next_renewable_gen_p = next_ori_obs[0], next_ori_obs[1]

        adjust_gen_p = clip_action[detail_action_dim[0][0]: detail_action_dim[0][1]]
        adjust_gen_v = clip_action[detail_action_dim[1][0]: detail_action_dim[1][1]]
        adjust_adjld_p = clip_action[detail_action_dim[2][0]: detail_action_dim[2][1]]
        adjust_stoenergy_p = clip_action[detail_action_dim[3][0]: detail_action_dim[3][1]]

        thermal_adjust_gen_p = np.array([adjust_gen_p[idx] for idx in settings.thermal_ids]).sum()
        renewable_adjust_gen_p = np.array([adjust_gen_p[idx] for idx in settings.renewable_ids]).sum()

        print('----------------------- check state change ------------------------------')
        # print('thermal_adjust_gen_p', thermal_adjust_gen_p)
        # print('real next thermal gen_p', thermal_adjust_gen_p + thermal_gen_p)
        # print('predict next thermal gen_p', next_thermal_gen_p)
        print('renewable_adjust_gen_p', renewable_adjust_gen_p)
        print('real next renewable gen_p', renewable_adjust_gen_p + renewable_gen_p)
        print('predict next renewable gen_p', next_renewable_gen_p)

    def check_balance_gen(self, state, action, next_state):
        balanced_id = settings.balanced_id
        min_balanced_bound = settings.min_balanced_gen_bound
        max_balanced_bound = settings.max_balanced_gen_bound
        gen_p_min = settings.gen_p_min
        gen_p_max = settings.gen_p_max
        min_val = min_balanced_bound * gen_p_min[balanced_id]
        max_val = max_balanced_bound * gen_p_max[balanced_id]

        now_balance_gen_p = state.gen_p[balanced_id]
        next_balance_gen_p = next_state.gen_p[balanced_id]

        print('------------------------- print balance -------------------------')
        print('max bound', max_val)
        print('min bound', min_val)
        print('now balance', now_balance_gen_p)
        print('next balance', next_balance_gen_p)

    def check_min_state1(self, state, action, next_state):
        detail_action_dim = self.config.detail_action_dim
        print('')
        print('adjust action', action[detail_action_dim[0][0]: detail_action_dim[0][1]].sum())
        print('actor now gen_p', sum(state.gen_p) - state.gen_p[settings.balanced_id])
        print('actor next gen_p', sum(next_state.gen_p) - next_state.gen_p[settings.balanced_id])

    def check_env_change(self, state, action, next_state):
        simpleld_p = np.array([state.ld_p[idx] for idx in settings.simpleld_ids], dtype=np.float32)
        grid_loss = np.array(state.grid_loss, dtype=np.float32) / 0.01
        next_simpleld_p = np.array([next_state.ld_p[idx] for idx in settings.simpleld_ids], dtype=np.float32)
        next_grid_loss = np.array(next_state.grid_loss, dtype=np.float32) / 0.01
        print('----- step change -----')
        print('simpleld_p change', np.sum((next_simpleld_p - simpleld_p)))
        print('grid_loss change', next_grid_loss - grid_loss)

    def set_ori_dict(self, state):
        self.bus_gen = state.bus_gen
        self.bus_load = state.bus_load
        self.bus_branch = state.bus_branch
        # f_info = open(self.config.res_file_dir + 'top_struct_info.txt', 'w')
        # f_info.write('bus_gen\n')
        # f_info.write(str(json.dumps(state.bus_gen, indent=4)) + '\n')
        # f_info.write('bus_load\n')
        # f_info.write(str(json.dumps(state.bus_load, indent=4)) + '\n')
        # f_info.write('bus_branch\n')
        # f_info.write(str(json.dumps(state.bus_branch, indent=4)) + '\n')
        # f_info.close()

    def check_state_info(self, state, action, next_state):
        # print('bus_gen', json.dumps(state.bus_gen, indent=4))
        # print('bus_load', json.dumps(state.bus_load, indent=4))
        # print('bus_branch', json.dumps(state.bus_branch, indent=4)
        cut_line_name = []
        cut_ld_name = []
        for i in range(settings.ln_num):
            if state.steps_to_reconnect_line[i] > 0:
                cut_line_name.append(settings.ln_name[i])

        thermal_gen_p = sum([state.gen_p[idx] for idx in settings.thermal_ids])
        renewable_gen_p = sum([state.gen_p[idx] for idx in settings.renewable_ids])
        balance_gen_p = state.gen_p[settings.balanced_id]
        adjld_p = sum(state.adjld_p)
        stoenergy_p = sum(state.stoenergy_p)
        simpleld_p = sum(state.ld_p) - adjld_p - stoenergy_p
        grid_loss = sum(state.grid_loss) / 0.01
        nextstep_adjld_p = sum([state.nextstep_ld_p[idx] for idx in settings.adjld_ids])
        # print(cut_ld_name)
        print('------------------------------------')
        print('disc_name', next_state.disc_name)
        print('recover_name', next_state.recover_name)


        # for key in state.bus_branch:
        #     for gen1 in self.bus_branch[key]:
        #         if gen1 not in state.bus_branch[key]:
        #             print('!!!!!!!!!!!!!!!!!!!!!!')
        #             print(gen1)
        #             print(gen1[:-3] in cut_line_name)




