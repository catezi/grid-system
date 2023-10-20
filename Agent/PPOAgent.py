import os
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


class PPOAgent(BaseAgent):
    def __init__(self, settings, config):
        BaseAgent.__init__(self, settings)
        # set basic info
        self.settings = settings
        self.config = config
        self.state_dim = config.state_dim
        self.action_dim = config.action_dim
        self.device = config.device

        # set actor model, optimizer and scheduler
        self.actor = Actor_logProp(settings, config).to(self.device)
        self.actor_target = Actor_logProp(settings, config).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.lr_actor)
        self.actor_scheduler = torch.optim.lr_scheduler.StepLR(self.actor_optimizer,
                                                               step_size=config.lr_decay_step_size,
                                                               gamma=config.lr_decay_gamma)
        # set critic model, optimizer and scheduler
        self.critic = Critic_Value(settings, config).to(self.device)
        self.critic_target = Critic_Value(settings, config).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.lr_critic)
        self.critic_scheduler = torch.optim.lr_scheduler.StepLR(self.critic_optimizer,
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
        # PPO update num for an epoch
        self.K_epochs = 10
        # record data file
        self.f_actor = None
        self.f_learner = None
        # state normalization
        self.state_norm = StateNormalization(self.config)
        # history states buffer
        self.history_info = HistoryInfoBuffer(self.config)
        # graph info process
        self.graph_processer = GraphProcesser(self.config)

    def load(self, state_norm=None):
        if self.config.load_model:
            save_model = torch.load(self.config.model_load_path, map_location=lambda storage, loc: storage)
            self.actor.load_state_dict(save_model['actor_network'])
            self.actor_target.load_state_dict(save_model['actor_network'])
            self.critic.load_state_dict(save_model['critic_network'])
            self.critic_target.load_state_dict(save_model['critic_network'])

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

    def update(self, f_learner, state_norm):
        states, actions, rewards, dones, next_states, logprobs, \
        action_lows, action_highs, next_action_lows, next_action_highs, \
        history_states, next_history_states, history_actions, next_history_actions, \
        features, adjacencys, next_features, next_adjacencys, = \
            self.buffer.sample_data() if self.config.buffer_type == 'FIFO' else self.buffer.get_all_data()
        # Calculate the advantage using GAE
        advantages = []
        gae = 0
        with torch.no_grad():  # adv and v_target have no gradient
            vs = self.critic(
                x=states,
                h_x=history_states,
                h_a=history_actions,
                feat=features,
                adj=adjacencys,
            )
            vsn = self.critic(
                x=next_states,
                h_x=next_history_states,
                h_a=next_history_actions,
                feat=next_features,
                adj=next_adjacencys,
            )
            deltas = rewards + self.config.gamma * (-dones + 1.0) * vsn - vs
            for delta, d in zip(reversed(deltas.flatten().numpy()), reversed(dones.flatten().numpy())):
                gae = delta + self.config.gamma * self.config.lamb * gae * (1.0 - d)
                advantages.insert(0, gae)
            advantages = torch.tensor(advantages, dtype=torch.float).view(-1, 1)
            v_targets = advantages + vs
        for _ in range(self.K_epochs):
            if self.config.use_mini_batch:
                for index in BatchSampler(
                    SubsetRandomSampler(range(self.config.batch_size)), self.config.mini_batch_size, False
                ):
                    mini_states = states[index]
                    mini_actions = actions[index]
                    mini_rewards = rewards[index]
                    mini_dones = dones[index]
                    mini_next_states = next_states[index]
                    mini_logprobs = logprobs[index]
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
                    mini_advantages = advantages[index]
                    mini_v_targets = v_targets[index]

                    # calculate actor for PPO
                    now_logprobs, dist_entropy = self.actor.get_logprob(
                        x=mini_states, a=mini_actions,
                        h_x=mini_history_states, h_a=mini_history_actions,
                        feat=mini_features, adj=mini_adjacencys,
                        action_low=mini_action_lows, action_high=mini_action_highs,
                    )
                    ratios = torch.exp(now_logprobs - mini_logprobs)
                    surr1 = ratios * mini_advantages
                    surr2 = torch.clamp(ratios, 1 - self.config.eps_clip, 1 + self.config.eps_clip) * mini_advantages
                    actor_loss = -torch.min(surr1, surr2).mean()
                    # calculate critic for PPO
                    state_values = self.critic(
                        x=mini_states,
                        h_x=mini_history_states,
                        h_a=mini_history_actions,
                        feat=mini_features,
                        adj=mini_adjacencys,
                    )
                    critic_loss = F.mse_loss(state_values, mini_v_targets)
                    # calculate entropy loss for PPO
                    regular = - 0.01 * dist_entropy
                    # combine all loss for PPO
                    loss = actor_loss + critic_loss + regular
                    # take gradient step
                    self.actor_optimizer.zero_grad()
                    self.critic_optimizer.zero_grad()
                    loss.mean().backward()
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.gradient_clip)
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.gradient_clip)
                    self.critic_optimizer.step()
                    self.actor_optimizer.step()
                    self.critic_scheduler.step(None)
                    self.actor_scheduler.step(None)
                    # output res
                    print('actor loss', actor_loss.mean().item())
                    print('critic loss', critic_loss.mean().item())
                    if self.config.output_res:
                        f_learner.write(' actor loss: %f\n' % actor_loss.mean().item())
                        f_learner.write('critic loss: %f\n\n' % critic_loss.mean().item())
            else:
                # calculate actor for PPO
                now_logprobs, dist_entropy = self.actor.get_logprob(
                    x=states, a=actions,
                    h_x=history_states, h_a=history_actions,
                    feat=features, adj=adjacencys,
                    action_low=action_lows, action_high=action_highs,
                )
                ratios = torch.exp(now_logprobs - logprobs)
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.config.eps_clip, 1 + self.config.eps_clip) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                # calculate critic for PPO
                state_values = self.critic(
                    x=states,
                    h_x=history_states,
                    h_a=history_actions,
                    feat=features,
                    adj=adjacencys,
                )
                critic_loss = F.mse_loss(state_values, v_targets)
                # calculate entropy loss for PPO
                regular = - 0.01 * dist_entropy
                # combine all loss for PPO
                loss = actor_loss + critic_loss + regular
                # take gradient step
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.gradient_clip)
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.gradient_clip)
                self.critic_optimizer.step()
                self.actor_optimizer.step()
                self.critic_scheduler.step(None)
                self.actor_scheduler.step(None)
                # output res
                print('actor loss', actor_loss.mean().item())
                print('critic loss', critic_loss.mean().item())
                if self.config.output_res:
                    f_learner.write(' actor loss: %f\n' % actor_loss.mean().item())
                    f_learner.write('critic loss: %f\n\n' % critic_loss.mean().item())
        # do IO only after last update epoch
        if self.config.output_res:
            f_learner.flush()

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

    def my_act(self, env_obs, norm_obs, action_low, action_high,
               history_obs, history_actions, feature, adjacency, sample=True):
        state = torch.tensor(norm_obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        action_low = torch.tensor(action_low, dtype=torch.float32).unsqueeze(0).to(self.device)
        action_high = torch.tensor(action_high, dtype=torch.float32).unsqueeze(0).to(self.device)
        history_obs = torch.tensor(history_obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        history_actions = torch.tensor(history_actions, dtype=torch.float32).unsqueeze(0).to(self.device)
        feature = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(self.device) if self.config.use_topology_info else None
        adjacency = torch.tensor(adjacency, dtype=torch.float32).unsqueeze(0).to(self.device) if self.config.use_topology_info else None
        action, log_prob, ori_action = self.actor(
            x=state,
            h_x=history_obs,
            h_a=history_actions,
            feat=feature,
            adj=adjacency,
            action_low=action_low,
            action_high=action_high,
            sample=True,
        )
        action = action.detach().cpu().numpy().flatten()
        log_prob = log_prob.detach().cpu().numpy().flatten()
        ori_action = ori_action.detach().cpu().numpy().flatten()
        interact_action, clip_action = self.process_action(env_obs, action)

        return interact_action, action, clip_action, ori_action, log_prob

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
            # clear history states buffer at the start of each episode
            self.history_info.clear()
            # add now obs to the history buffer
            self.history_info.add_state(norm_obs)
            # init graph info at the first episode
            if episode == 0 and self.config.use_topology_info:
                self.graph_processer.init_adjacency(obs)
            # update graph info every step
            feature, adjacency = self.graph_processer.update_feature_and_adjacency(obs) if self.config.use_topology_info else None, None
            action_low, action_high = self.process_action_space(obs, obs.action_space)
            # while not done:
            for timestep in range(self.config.max_timestep):
                interact_action, action, clip_action, ori_action, log_prob = self.my_act(
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
                next_feature, next_adjacency = self.graph_processer.update_feature_and_adjacency(next_obs) if self.config.use_topology_info else None, None
                next_action_low, next_action_high = self.process_action_space(next_obs, next_obs.action_space)
                env_reward = reward
                reward = self.process_reward(obs, next_obs, reward, info, timestep, action, clip_action)
                rtg += env_reward
                # print('reward', reward)
                # save step data to data buffer
                self.buffer.add_data(
                    obs=norm_obs,
                    action=ori_action,  # PPO algorithm only need action without reflection, restriction and clamp to calculate new logprob
                    logprob=log_prob,
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
                # update every model_update_freq steps
                if self.buffer.get_buffer_size() >= self.config.batch_size and \
                        total_steps % self.config.model_update_freq == 0:
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
                feature, adjacency = next_feature, next_adjacency
        if self.config.output_res:
            self.f_actor.close()
            self.f_learner.close()

    def act(self, obs, reward, done=False):
        pass

