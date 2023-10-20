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
parser.add_argument('--max_episode', default=10, type=int, help='Max episode for training')
parser.add_argument('--max_timestep', default=1000, type=int, help='Max timestep per episode')
parser.add_argument('--max_buffer_size', default=1000000, type=int, help='Max buffer size')
parser.add_argument('--set_gen_p', default=1, type=int, help='If set gen_p in action space')
parser.add_argument('--set_gen_v', default=0, type=int, help='If set gen_v in action space')
parser.add_argument('--set_adjld_p', default=1, type=int, help='If set adjld_p in action space')
parser.add_argument('--set_stoenergy_p', default=1, type=int, help='If set stoenergy_p in action space')
parser.add_argument('--min_state', default=0, type=int, help='If use min state space')
parser.add_argument('--use_state_norm', default=0, type=int, help='If use state normalization')
parser.add_argument('--reflect_actionspace', default=0, type=int, help='If reflect action to action space')
parser.add_argument('--output_res', default=0, type=int, help='If output res into file')
parser.add_argument('--output_data', default=0, type=int, help='If output trajectory data')
parser.add_argument('--load_model', default=1, type=int, help='If load pretrain model')
parser.add_argument('--load_state_normalization', default=1, type=int, help='If load saved state normalization')
parser.add_argument('--update_state_normalization', default=0, type=int, help='If update state normalization while training')
parser.add_argument('--split_balance_loss', default=0, type=int, help='If split balance loss function')
########## parameters to select good actions
parser.add_argument('--min_good_rtgs', default=200.0, type=float, help='Min rtgs for a good episode')
parser.add_argument('--min_good_rounds', default=300, type=int, help='Min rounds for a good episode')
parser.add_argument('--total_sample_episode', default=10, type=int, help='Min rounds for a good episode')
########## use gru to process history state
parser.add_argument('--use_history_state', default=0, type=int, help='If use gua to load history state')
parser.add_argument('--use_history_action', default=0, type=int, help='If use gua to load history action')
parser.add_argument('--history_state_len', default=5, type=int, help='Len of history state sequence')
parser.add_argument('--gru_num_layers', default=2, type=int, help='Numbers of gru layers')
parser.add_argument('--gru_hidden_size', default=64, type=int, help='Hidden size for gru structure')
########## model structure
parser.add_argument('--active_function', default='tanh', type=str, help='Active function for actor/critic model')
# Environment settings
parser.add_argument('--ban_prob_disconnection', default=0, type=int, help='Probability if disconnection in env')
parser.add_argument('--ban_check_gen_status', default=0, type=int, help='If ban check gen status')
parser.add_argument('--ban_thermal_on', default=0, type=int, help='If ban thermal turn on')
parser.add_argument('--ban_thermal_off', default=0, type=int, help='If ban thermal turn off')
parser.add_argument('--restrict_thermal_on_off', default=0, type=int, help='If restrict thermal turn on/off')
##################################### Hipper parameters
parser.add_argument('--lr_actor', default=1e-3, type=float, help='Init lr for actor model')
parser.add_argument('--lr_critic', default=1e-4, type=float, help='Init lr for critic model')
parser.add_argument('--lr_decay_step_size', default=1000, type=int, help='Lr decay frequency')
parser.add_argument('--lr_decay_gamma', default=0.99, type=float, help='lr decay rate')
parser.add_argument('--init_model', default=0, type=int, help='If init model parameters')
# File path
parser.add_argument('--model_load_path', default='./save_model/pretrain_model.pth', type=str, help='Path of model to load')
parser.add_argument('--state_normal_load_path', default='./save_model/pretrain_model.pth', type=str, help='Path of state normal to load')
parser.add_argument('--res_file_dir', default='./res/', type=str, help='Path of output res file')
parser.add_argument('--res_file_name', default='output_res.txt', type=str, help='name of output res file')
parser.add_argument('--trajectory_file_name', default='trajectory_data.pkl', type=str, help='name of output trajectory data file')

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
    # change setting
    change_config_in_setting(config)
    # set random seed
    set_seed(123)
    # set agent and start training
    my_agent = DDPGAgent(settings, config)
    my_agent.test_model()
