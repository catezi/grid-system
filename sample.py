# -*- coding: UTF-8 -*-
import torch
import random
import argparse
import numpy as np

from utils import *
from Agent.PPOAgent_old import PPOAgent
from Agent.DDPGAgent import DDPGAgent
from utilize.settings import settings
from Agent.RandomAgent import RandomAgent
from Environment.base_env import Environment
from Agent.DoNothingAgent import DoNothingAgent

parser = argparse.ArgumentParser(description='Grid system control program')
##################################### Training config
parser.add_argument('--buffer_type', default='FIFO', type=str, help='Type of data buffer for training')
parser.add_argument('--max_episode', default=200, type=int, help='Max episode for training')
parser.add_argument('--max_timestep', default=1000, type=int, help='Max timestep per episode')
parser.add_argument('--max_buffer_size', default=1000000, type=int, help='Max buffer size')
parser.add_argument('--set_gen_p', default=1, type=int, help='If set gen_p in action space')
parser.add_argument('--set_gen_v', default=0, type=int, help='If set gen_v in action space')
parser.add_argument('--set_adjld_p', default=1, type=int, help='If set adjld_p in action space')
parser.add_argument('--set_stoenergy_p', default=1, type=int, help='If set stoenergy_p in action space')
parser.add_argument('--min_state', default=0, type=int, help='If use min state space')
parser.add_argument('--model_update_freq', default=50, type=int, help='Model update frequency')
parser.add_argument('--model_save_freq', default=100, type=int, help='Model save frequency')
parser.add_argument('--init_model', default=0, type=int, help='If init model parameters')
parser.add_argument('--save_model', default=0, type=int, help='If save model at stable frequency')
parser.add_argument('--output_res', default=0, type=int, help='If output res into file')
parser.add_argument('--load_model', default=0, type=int, help='If load pretrain model')
parser.add_argument('--load_state_normalization', default=0, type=int, help='If load saved state normalization')
parser.add_argument('--update_state_normalization', default=0, type=int, help='If update state normalization while training')
parser.add_argument('--use_mini_batch', default=0, type=int, help='If train by mini batch')
parser.add_argument('--use_state_norm', default=0, type=int, help='If use state normalization')
parser.add_argument('--reflect_actionspace', default=0, type=int, help='If reflect action to action space')
########## setting for balance loss
parser.add_argument('--add_balance_loss', default=0, type=int, help='If add balance loss to actor loss')
parser.add_argument('--balance_loss_rate', default=0.01, type=float, help='Rate for balance loss')
parser.add_argument('--balance_loss_rate_decay_rate', default=0.01, type=float, help='Decay rate for decay balance loss rate')
parser.add_argument('--balance_loss_rate_decay_freq', default=100, type=int, help='Decay freq for decay balance loss rate')
parser.add_argument('--min_balance_loss_rate', default=0.001, type=float, help='Min rate for balance loss')
parser.add_argument('--split_balance_loss', default=0, type=int, help='If split balance loss function')
parser.add_argument('--danger_region_rate', default=0.1, type=float, help='Rate of range for danger region')
parser.add_argument('--save_region_rate', default=0.2, type=float, help='Rate of range for warning region')
parser.add_argument('--save_region_balance_loss_rate', default=0.0001, type=float, help='Balance loss rate for save region')
parser.add_argument('--warning_region_balance_loss_rate', default=0.001, type=float, help='Balance loss rate for warning region')
parser.add_argument('--danger_region_balance_loss_rate', default=0.01, type=float, help='Balance loss rate for danger region')
########## use gru to process history state
parser.add_argument('--use_history_state', default=0, type=int, help='If use gua to load history state')
parser.add_argument('--use_history_action', default=0, type=int, help='If use gua to load history action')
parser.add_argument('--history_state_len', default=5, type=int, help='Len of history state sequence')
parser.add_argument('--gru_num_layers', default=2, type=int, help='Numbers of gru layers')
parser.add_argument('--gru_hidden_size', default=64, type=int, help='Hidden size for gru structure')
########## model structure
parser.add_argument('--active_function', default='tanh', type=str, help='Active function for actor/critic model')
########## settings for reward/punish
parser.add_argument('--punish_balance_out_range', default=0, type=int, help='If punish balance gen power high or low')
parser.add_argument('--punish_balance_out_range_rate', default=0.1, type=float, help='Rate for punish balance out range')
parser.add_argument('--reward_from_env', default=1, type=int, help='If use reward from environment')
parser.add_argument('--reward_for_survive', default=0.0, type=float, help='Additional reward for survive per round')
########## parameters for evaluation
parser.add_argument('--min_good_rtgs', default=200.0, type=float, help='Min rtgs for a good episode')
parser.add_argument('--min_good_rounds', default=672, type=int, help='Min rounds for a good episode')
parser.add_argument('--total_sample_episode', default=250, type=int, help='Num episode per process')
parser.add_argument('--sample_block_size', default=100, type=int, help='Min rounds for a good episode')
##################################### Hipper parameters
parser.add_argument('--lr_actor', default=1e-3, type=float, help='Init lr for actor model')
parser.add_argument('--lr_critic', default=1e-4, type=float, help='Init lr for critic model')
parser.add_argument('--lr_decay_step_size', default=1000, type=int, help='Lr decay frequency')
parser.add_argument('--lr_decay_gamma', default=0.99, type=float, help='lr decay rate')
parser.add_argument('--batch_size', default=1024, type=int, help='Batch size for training')
parser.add_argument("--mini_batch_size", default=64, type=int, help='Minibatch size for training')
parser.add_argument('--gradient_clip', default=1.0, type=float, help='Clip for para gradient')
parser.add_argument('--gamma', default=0.9, type=float, help='Gamma for future reward decay')
parser.add_argument('--lamb', default=0.98, type=float, help='Lambda for gae advantage')
parser.add_argument('--eps_clip', default=0.1, type=float, help='Eps clip for PPO')
parser.add_argument('--soft_tau', default=0.01, type=float, help='Soft update rate for target actor/critic model')
parser.add_argument('--init_action_std', default=0.3, type=float, help='Init std for action')
parser.add_argument('--action_std_decay_rate', default=0.01, type=float, help='Decay rate for std of action')
parser.add_argument('--action_std_decay_freq', default=100, type=float, help='Decay freq for std of action')
parser.add_argument('--min_action_std', default=0.01, type=float, help='Min action std')
##################################### Environment settings
parser.add_argument('--ban_prob_disconnection', default=0, type=int, help='Probability if disconnection in env')
parser.add_argument('--ban_check_gen_status', default=0, type=int, help='If ban check gen status')
parser.add_argument('--ban_thermal_on', default=0, type=int, help='If ban thermal turn on')
parser.add_argument('--ban_thermal_off', default=0, type=int, help='If ban thermal turn off')
parser.add_argument('--restrict_thermal_on_off', default=0, type=int, help='If restrict thermal turn on/off')
##################################### File path
parser.add_argument('--mean_std_info_path', default='./data/mean_std_info.pkl', type=str, help='Path of file save mean and std')
parser.add_argument('--model_save_dir', default='./save_model/train_coverage/', type=str, help='Dir of model to save')
parser.add_argument('--model_load_path', default='./save_model/pretrain_model.pth', type=str, help='Path of model to load')
parser.add_argument('--state_normal_load_path', default='./save_model/pretrain_model.pth', type=str, help='Path of state normal to load')
parser.add_argument('--res_file_dir', default='./res/', type=str, help='Path of output res file')
parser.add_argument('--res_file_name', default='output_res.txt', type=str, help='name of output res file')
parser.add_argument('--sample_data_file_dir', default='./sample_data/', type=str, help='Path of sample data file')
parser.add_argument('--sample_data_file_name', default='simpleld_p.pkl', type=str, help='name of sample data file')


if __name__ == "__main__":
    config = parser.parse_args()
    # set basic info in args
    config.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config.state_dim = 8 if config.min_state == 2 else \
         (156 if config.min_state == 1 else 3300)
    if config.restrict_thermal_on_off:
        config.state_dim += 2 * len(settings.thermal_ids)  # 2-dim one hot vector for thermal state
    config.action_dim = 2 * settings.gen_num + settings.adjld_num + settings.stoenergy_num
    config.detail_action_dim = [
        [0, settings.gen_num],
        [settings.gen_num, 2 * settings.gen_num],
        [2 * settings.gen_num, 2 * settings.gen_num + settings.adjld_num],
        [2 * settings.gen_num + settings.adjld_num, 2 * settings.gen_num + settings.adjld_num + settings.stoenergy_num],
    ]
    config.more_detail_action_dim = {
        'thermal_gen_p': len(settings.thermal_ids),
        'renewable_gen_p': len(settings.renewable_ids),
        'adjld_p': settings.adjld_num,
        'stoenergy_p': settings.stoenergy_num,
    }
    config.has_continuous_action_space = True
    config.load_model = 1
    config.load_state_normalization = 1
    config.update_state_normalization = 0
    config.use_state_norm = 1
    config.init_action_std = 0.0
    # change global setting from my config
    change_config_in_setting(config)
    # set random seed
    set_seed(123)
    # start sampling
    my_agent = DDPGAgent(settings, config)
    my_agent.sample_predict_data()
