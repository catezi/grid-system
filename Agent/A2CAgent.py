import os
import sys
import django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "power.power.settings")
django.setup()
from django.conf import settings
from grid.models import PowerGrid
import gym
import copy
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from utils import *
from collections import deque
from utilize.form_action import *
from Agent.BaseAgent import BaseAgent
from Agent.model.Actor_Critic import *
from torch.distributions import Normal
from Environment.base_env import Environment
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


GAMMA = 0.93
LR_ACTOR = 1e-4
LR_CRITIC = 1e-3
ENTROPY_COEF = 1e-1

seed = 39
seed_to_test = 21
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class A2CAgent(BaseAgent):
    def __init__(self, settings, config, ):
        BaseAgent.__init__(self, settings)
        # set basic info
        self.settings = settings
        self.config = config
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        self.device = config.device
        # set actor model, optimizer and scheduler
        self.actor = Actor_logProp(settings, config).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.lr_actor)
        self.actor_scheduler = torch.optim.lr_scheduler.StepLR(self.actor_optimizer,
                                                               step_size=config.lr_decay_step_size,
                                                               gamma=config.lr_decay_gamma)
        # set critic model, optimizer and scheduler
        self.critic = Critic_Value(settings, config).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.lr_critic)
        self.critic_scheduler = torch.optim.lr_scheduler.StepLR(self.critic_optimizer,
                                                                step_size=config.lr_decay_step_size,
                                                                gamma=config.lr_decay_gamma)
        # set data buffer
        self.buffer = RolloutBuffer(config)
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

        # record data file
        self.f_actor = None
        self.f_learner = None
        # state normalization
        self.state_norm = StateNormalization(self.config)

    def load(self, state_norm=None):
        if self.config.load_model:
            save_model = torch.load(self.config.model_load_path, map_location=lambda storage, loc: storage)
            self.actor.load_state_dict(save_model['actor_network'])
            self.critic.load_state_dict(save_model['critic_network'])

        if self.config.load_state_normalization and state_norm is not None:
            save_state_normal = torch.load(self.config.state_normal_load_path,
                                           map_location=lambda storage, loc: storage)
            state_norm.set_info(save_state_normal['state_norm'])

    def save(self, update_round, checkpoint_path, state_norm=None):
        if not os.path.isdir(self.config.model_save_dir):
            os.makedirs(self.config.model_save_dir)
        torch.save({
            'update_round': update_round,
            'actor_network': self.actor.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_network': self.critic.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'state_norm': state_norm.get_info() if state_norm is not None else None,
        }, checkpoint_path)

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
        # ban thermal on off event to make env stable
        if self.config.ban_thermal_on:
            for idx in self.settings.thermal_ids:
                # process thermal on
                if obs.last_injection_gen_p[idx] == 0.0:
                    action_high[idx] = 0.0
                    obs.action_space['adjust_gen_p'].high[idx] = 0.0
                # process thermal off
                if obs.last_injection_gen_p[idx] == self.settings.gen_p_min[idx]:
                    action_low[idx] = 0.0
                    obs.action_space['adjust_gen_p'].low[idx] = 0.0

        return action_low, action_high

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

    def my_act(self, env_obs, norm_obs, action_low, action_high, sample=True):
        state = torch.tensor(norm_obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        action_low = torch.tensor(action_low, dtype=torch.float32).unsqueeze(0).to(self.device)
        action_high = torch.tensor(action_high, dtype=torch.float32).unsqueeze(0).to(self.device)
        action, log_prob, _ = self.actor(
            x=state,
            h_x=None,
            h_a=None,
            feat=None,
            adj=None,
            action_low=action_low,
            action_high=action_high,
            sample=True,
        )
        action = action.detach().cpu().numpy().flatten()
        log_prob = log_prob.flatten()
        interact_action, clip_action = self.process_action(env_obs, action)

        return interact_action, action, clip_action, log_prob

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

    def update(self, f_learner, state_norm):
        states, actions, rewards, dones, next_states, logprobs, \
        action_lows, action_highs, next_action_lows, next_action_highs = self.buffer.get_all_data()

        V = self.critic(
            x=states,
            h_x=None,
            h_a=None,
            feat=None,
            adj=None,
        )
        Q = rewards + self.config.gamma * self.critic(
            x=next_states,
            h_x=None,
            h_a=None,
            feat=None,
            adj=None,
        ).detach() * (1 - dones)
        ################ Actor Update ###################
        actor_loss = -((Q - V.detach()) * logprobs).mean()
        # use loss to confirm balance
        if self.config.add_balance_loss:
            now_action, _ = self.actor(
                x=states,
                h_x=None,
                h_a=None,
                feat=None,
                adj=None,
                action_low=action_lows,
                action_high=action_highs,
                sample=False,
            )
            actor_loss += self.calculate_balance_loss(states, next_states, now_action, state_norm, actions, dones)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        ################ Critic Update ###################
        critic_loss = F.mse_loss(Q, V)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # clear data buffer
        self.buffer.clear()

        # output res
        print('actor loss', actor_loss.mean().item())
        print('critic loss', critic_loss.mean().item())
        if self.config.output_res:
            f_learner.write(' actor loss: %f\n' % actor_loss.mean().item())
            f_learner.write('critic loss: %f\n\n' % critic_loss.mean().item())
            f_learner.flush()

    def train(self):
        rounds = []
        update_times = 0
        total_steps = 0
        # load pretrain model(and state normal) if enable
        self.load(self.state_norm)

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
            action_low, action_high = self.process_action_space(obs, obs.action_space)
            # while not done:
            for timestep in range(self.config.max_timestep):
                interact_action, action, clip_action, log_prob = self.my_act(
                    env_obs=obs, norm_obs=norm_obs,
                    action_low=action_low, action_high=action_high, sample=True,
                )
                next_obs, reward, done, info = env.step(interact_action)
                next_norm_obs = self.state_norm(next_obs)
                # self.state_check.check_min_state1(obs, clip_action, next_obs)
                next_action_low, next_action_high = self.process_action_space(next_obs, next_obs.action_space)
                env_reward = reward
                reward = self.process_reward(obs, next_obs, reward, info, timestep, action, clip_action)
                rtg += env_reward
                # print('reward', reward)
                # save step data to data buffer
                self.buffer.add_data(
                    obs=norm_obs,
                    action=clip_action,
                    logprob=log_prob,
                    reward=reward,
                    done=done,
                    next_obs=next_norm_obs,
                    action_low=action_low,
                    action_high=action_high,
                    next_action_low=next_action_low,
                    next_action_high=next_action_high,
                    history_obs=None,
                    next_history_obs=None,
                    history_actions=None,
                    next_history_actions=None,
                )
                # update every model_update_freq steps
                if self.buffer.get_buffer_size() >= self.config.batch_size:
                    update_times += 1
                    total_steps %= self.config.model_update_freq
                    self.update(self.f_learner, self.state_norm)
                    # decay action sample std every interval
                    if update_times % self.config.action_std_decay_freq == 0:
                        self.decay_action_std(self.config.action_std_decay_rate, self.config.min_action_std)
                    # save model every interval update
                    if self.config.save_model and update_times % self.config.model_save_freq == 0:
                        self.save(
                            update_round=update_times,
                            checkpoint_path=self.config.model_save_dir + str(update_times) + '_save_model.pth',
                            state_norm=self.state_norm
                        )
                    # decay balance rate every interval
                    if update_times % self.config.balance_loss_rate_decay_freq == 0:
                        self.decay_balance_loss_rate(self.config.balance_loss_rate_decay_rate,
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
        if self.config.output_res:
            self.f_actor.close()
            self.f_learner.close()

    def act(self, obs, reward, done=False):
        pass

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
            action_low, action_high = self.process_action_space(obs, obs.action_space)
            # while not done:
            for timestep in range(self.config.max_timestep):
                interact_action, action, clip_action, log_prob = self.my_act(
                    env_obs=obs, norm_obs=norm_obs,
                    action_low=action_low, action_high=action_high, sample=True,
                )
                next_obs, reward, done, info = env.step(interact_action)
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
                next_norm_obs = self.state_norm(next_obs)
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
        print('mean_rtg', sum(rtgs) / len(rtgs))
        print('mean_round', sum(rounds) / len(rounds))
        if self.config.output_res:
            self.f_actor.write('mean_rtg ' + str(sum(rtgs) / len(rtgs)) + '\n')
            self.f_actor.write('mean_round ' + str(sum(rounds) / len(rounds)) + '\n')
            self.f_actor.close()

