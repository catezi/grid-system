# -*- coding: UTF-8 -*-
import os
import torch
import pickle
import random
import argparse
import numpy as np
import pandas as pd

from utils import *
from utilize.form_action import *
from Agent.PPOAgent_old import PPOAgent
from utilize.settings import settings
from Agent.RandomAgent import RandomAgent
from Environment.base_env import Environment
from Agent.DoNothingAgent import DoNothingAgent


parser = argparse.ArgumentParser(description='DDPG MC Hunter testing program')
# Training config
parser.add_argument('--max_episode', default=1, type=int, help='Max episode for training')
parser.add_argument('--max_timestep', default=10000, type=int, help='Max timestep per episode')
parser.add_argument('--max_buffer_size', default=50, type=int, help='Max buffer size')
parser.add_argument('--K_epochs', default=10, type=int, help='Update times for a update stage')
parser.add_argument('--set_gen_p', default=1, type=int, help='If set gen_p in action space')
parser.add_argument('--set_gen_v', default=0, type=int, help='If set gen_v in action space')
parser.add_argument('--set_adjld_p', default=1, type=int, help='If set adjld_p in action space')
parser.add_argument('--set_stoenergy_p', default=1, type=int, help='If set stoenergy_p in action space')
parser.add_argument('--model_save_freq', default=100, type=float, help='Model save frequency')
parser.add_argument('--output_res', default=0, type=float, help='If output res into file')
parser.add_argument('--seed', default=123, type=int, help='Random seed for training')
parser.add_argument('--active_function', default='tanh', type=str, help='Active function for actor/critic model')
parser.add_argument('--punish_out_actionspace', default=0, type=int, help='If punish action out of action space')
parser.add_argument('--update_state_normalization', default=0, type=int, help='If update state normalization while training')
parser.add_argument('--mode', default='pretrain', type=str, help='To pretrain or to generate data')
parser.add_argument('--use_state_norm', default=1, type=int, help='If use state normalization')
# Environment setting
parser.add_argument('--prob_disconnection', default=0, type=float, help='Probability if disconnection in env')
# Hipper parameters
parser.add_argument('--lr_actor', default=1e-5, type=float, help='Init lr for actor model')
parser.add_argument('--lr_critic', default=1e-4, type=float, help='Init lr for critic model')
parser.add_argument('--lr_decay_step_size', default=1000, type=int, help='Lr decay frequency')
parser.add_argument('--lr_decay_gamma', default=0.99, type=float, help='lr decay rate')
parser.add_argument('--batch_size', default=128, type=int, help='Batch size for training')
parser.add_argument('--gamma', default=0.9, type=float, help='Gamma for future reward decay')
parser.add_argument('--lamb', default=0.98, type=float, help='Lambda for gae advantage')
parser.add_argument('--eps_clip', default=0.1, type=float, help='Eps clip for PPO')
parser.add_argument('--init_action_std', default=0.3, type=float, help='Init std for action')
parser.add_argument('--action_std_decay_rate', default=0.01, type=float, help='Decay rate for std of action')
parser.add_argument('--action_std_decay_freq', default=100, type=float, help='Decay freq for std of action')
parser.add_argument('--min_action_std', default=0.01, type=float, help='Min action std')
parser.add_argument('--fail_punish', default=-5.0, type=float, help='Punishment for not converged or out of bound')
# File path
parser.add_argument('--mean_std_info_path', default='./data/mean_std_info.pkl', type=str, help='Path of file save mean and std')
parser.add_argument('--save_dataset_path', default='./data/train_data.pkl', type=str, help='Path to save train data')
parser.add_argument('--model_save_dir', default='./save_model/pretrain_model/', type=str, help='Path of file save mean and std')
parser.add_argument('--res_file_dir', default='./res/', type=str, help='Path of output res file')
parser.add_argument('--res_file_name', default='output_res.txt', type=str, help='name of output res file')


class StaticData:
    def __init__(self):
        self.gen_p = pd.read_csv(settings.gen_p_filepath)
        self.gen_q = pd.read_csv(settings.gen_q_filepath)
        self.load_p = pd.read_csv(settings.ld_p_filepath)
        self.load_q = pd.read_csv(settings.ld_q_filepath)
        self.max_renewable_gen_p = pd.read_csv(settings.max_renewable_gen_p_filepath)
        self.states = []
        self.actions = []

    def get_next_step_action(self, last_obs, next_idx):
        # adjust_gen_p
        now_gen_p = self.gen_p.loc[next_idx, :]
        for idx in settings.renewable_ids:
            name = settings.gen_name_list[idx]
            now_gen_p.loc[name] = self.max_renewable_gen_p.loc[next_idx, name] \
                if now_gen_p.loc[name] > self.max_renewable_gen_p.loc[next_idx, name] else now_gen_p.loc[name]
        adjust_gen_p = np.array(now_gen_p.values, dtype=np.float32) - np.array(last_obs.gen_p, dtype=np.float32)
        adjust_gen_v = np.zeros_like(adjust_gen_p)
        adjust_adjld_p = np.array([self.load_p.loc[next_idx, ld] for ld in settings.adjld_name], dtype=np.float32) -\
                         np.array([last_obs.nextstep_ld_p[ld] for ld in settings.adjld_ids], dtype=np.float32)
        adjust_stoenergy_p = np.array([self.load_p.loc[next_idx, ld] for ld in settings.stoenergy_name], dtype=np.float32)

        return form_action(adjust_gen_p, adjust_gen_v, adjust_adjld_p, adjust_stoenergy_p)

    def get_gen_p_data_from_index(self, index):
        return self.gen_p.loc[index, :].values

    def get_load_p_data_from_index(self, index):
        return self.load_p.loc[index, :].values

    def obs_process(self, obs):
        # collect all needed state from obs
        all_action_space = np.concatenate([
            obs.action_space['adjust_gen_p'].low, obs.action_space['adjust_gen_p'].high,
            obs.action_space['adjust_gen_v'].low, obs.action_space['adjust_gen_v'].high,
            obs.action_space['adjust_adjld_p'].low, obs.action_space['adjust_adjld_p'].high,
            obs.action_space['adjust_stoenergy_p'].low, obs.action_space['adjust_stoenergy_p'].high,
        ], axis=0)
        print('all_action_space', all_action_space.shape)
        all_obs = {
            'gen_p': obs.gen_p, 'gen_q': obs.gen_q, 'gen_v': obs.gen_v,
            'target_dispatch': obs.target_dispatch, 'actual_dispatch': obs.actual_dispatch,
            'ld_p': obs.ld_p, 'adjld_p': obs.adjld_p, 'stoenergy_p': obs.stoenergy_p, 'ld_q': obs.ld_q, 'ld_v': obs.ld_v,
            'p_or': obs.p_or, 'q_or': obs.q_or, 'v_or': obs.v_or, 'a_or': obs.a_or,
            'p_ex': obs.p_ex, 'q_ex': obs.q_ex, 'v_ex': obs.v_ex, 'a_ex': obs.a_ex,
            'line_status': obs.line_status, 'grid_loss': obs.grid_loss, 'bus_v': obs.bus_v,
            'action_space': all_action_space,
            'steps_to_reconnect_line': obs.steps_to_reconnect_line, 'count_soft_overflow_steps': obs.count_soft_overflow_steps, 'rho': obs.rho,
            'gen_status': obs.gen_status, 'steps_to_recover_gen': obs.steps_to_recover_gen, 'steps_to_close_gen': obs.steps_to_close_gen,
            'curstep_renewable_gen_p_max': obs.curstep_renewable_gen_p_max, 'nextstep_renewable_gen_p_max': obs.nextstep_renewable_gen_p_max,
            'curstep_ld_p': obs.curstep_ld_p, 'nextstep_ld_p': obs.nextstep_ld_p,
            'total_adjld': obs.total_adjld,  'total_stoenergy': obs.total_stoenergy,
        }

        return all_obs

    def action_process(self, action):
        return np.concatenate([
            action['adjust_gen_p'],
            action['adjust_gen_v'],
            action['adjust_adjld_p'],
            action['adjust_stoenergy_p'],
        ], axis=0)

    def save_data(self, obs, action):
        self.states.append(self.obs_process(obs))
        self.actions.append(self.action_process(action))

    def save_all_data(self):
        f = open(config.save_dataset_path, 'wb')
        pickle.dump({
            'state': self.states,
            'action': self.actions,
        }, f)
        f.close()


def change_config_in_setting():
    settings.prob_disconnection = config.prob_disconnection
    settings.ban_legal_check = 1
    settings.ban_check_gen_status = 1
    settings.ban_check_steps_to_close_gen = 1


def generate_train_data(my_agent):
    fail_points = []
    now_step = 0
    env = Environment(settings, "EPRIReward")
    while True:
        obs = env.reset(seed=config.seed, start_sample_idx=now_step)
        reward = 0.0
        done = False
        while True:
            print('----------- step ', now_step, '-----------')
            now_step += 1
            action = static_data.get_next_step_action(obs, now_step)
            static_data.save_data(obs, action)
            obs, reward, done, info = env.step(action)
            static_data.obs_process(obs)
            print('reward', reward)
            if info['fail_info'] is not None:
                print('!!!!!!!!!!!!!!!!!!!!!!!! fail info ', info)
                fail_points.append(now_step)
                if info['fail_info'] == 'grid is not converged':
                    break
            if done:
                print('!!!!!!!!!!!!!!!!!!!!!!!! fail info ', info)
                print('now_step', now_step)
                break
        if info['fail_info'] == 'grid is not converged':
            break
        if now_step >= settings.sample_num:
            break
    print('fail_points', fail_points)
    print('fail_points', len(fail_points))
    # save all data
    static_data.save_all_data()


def pretrain_actor(my_agent):
    with open(config.save_dataset_path, 'rb') as f:
        train_data = pickle.load(f)
        my_agent.pretrain(train_data['state'], train_data['action'])


if __name__ == '__main__':
    config = parser.parse_args()
    # set basic info in args
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config.state_dim = 3546
    config.action_dim = 2 * settings.gen_num + settings.adjld_num + settings.stoenergy_num
    config.detail_action_dim = [
        [0, settings.gen_num],
        [settings.gen_num, 2 * settings.gen_num],
        [2 * settings.gen_num, 2 * settings.gen_num + settings.adjld_num],
        [2 * settings.gen_num + settings.adjld_num, 2 * settings.gen_num + settings.adjld_num + settings.stoenergy_num],
    ]
    config.has_continuous_action_space = True
    # set random seed
    set_seed(123)
    # change config in setting
    change_config_in_setting()
    # set agent and start training
    my_agent = PPOAgent(settings, config)
    # # gen train data from static data
    if config.mode == 'generate_':
        static_data = StaticData()
        generate_train_data(my_agent)
    else:
        pretrain_actor(my_agent)


