import os
import torch
import pickle
import numpy as np
import torch.nn as nn

from utils import *
from utilize.form_action import *
from Agent.BaseAgent import BaseAgent
from torch.distributions import Categorical
from torch.nn.modules.activation import ReLU
from Environment.base_env import Environment
from torch.distributions import MultivariateNormal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class ActorCritic(nn.Module):
    def __init__(self, config):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = config.has_continuous_action_space
        self.device = config.device

        if config.has_continuous_action_space:
            self.action_dim = config.action_dim
            self.action_var = torch.full((config.action_dim,),
                                         config.init_action_std * config.init_action_std).to(config.device)

        # actor
        if config.has_continuous_action_space:
            self.actor = nn.Sequential(
                nn.LayerNorm(config.state_dim),
                nn.Linear(config.state_dim, 256),
                nn.Tanh() if config.active_function == 'tanh' else nn.PReLU(),

                nn.Linear(256, 256),
                nn.Tanh() if config.active_function == 'tanh' else nn.PReLU(),

                nn.Linear(256, 128),
                nn.Tanh() if config.active_function == 'tanh' else nn.PReLU(),

                nn.Linear(128, config.action_dim),
                # nn.Tanh()
                # nn.Sigmoid()
            )
        else:
            self.actor = nn.Sequential(
                nn.LayerNorm(config.state_dim),
                nn.Linear(config.state_dim, 1024),
                nn.Tanh() if config.active_function == 'tanh' else nn.PReLU(),

                nn.Linear(1024, 256),
                nn.Tanh() if config.active_function == 'tanh' else nn.PReLU(),

                nn.Linear(256, config.action_dim),
                nn.Softmax(dim=-1)
            )

        # critic
        self.critic = nn.Sequential(
            nn.Linear(config.state_dim, 256),
            nn.Tanh() if config.active_function == 'tanh' else nn.PReLU(),

            nn.Linear(256, 128),
            nn.Tanh() if config.active_function == 'tanh' else nn.PReLU(),

            nn.Linear(128, 1)
        )
        #  parameters init
        for m in self.actor:
            if isinstance(m, nn.Linear):
                # nn.init.orthogonal_(m.weight, 1.0)
                # nn.init.constant_(m.bias, 1e-6)
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.critic:
            if isinstance(m, nn.Linear):
                # nn.init.orthogonal_(m.weight, 1.0)
                # nn.init.constant_(m.bias, 1e-6)
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def set_action_std(self, new_action_std):
        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(self.device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError

    def act(self, state):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()

    def get_critic_value(self, state):
        return self.critic(state)

    def get_actor_logprob(self, state, action):
        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(self.device)
            dist = MultivariateNormal(action_mean, cov_mat)
            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        if np.isnan(action_mean.detach().mean().item()):
            print('action_mean', action_mean.detach().mean().detach())

        return action_logprobs.unsqueeze(-1), dist_entropy.unsqueeze(-1)

    def evaluate(self, state, action):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)

            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(self.device)
            dist = MultivariateNormal(action_mean, cov_mat)

            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)

        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, config):
        self.config = config
        self.has_continuous_action_space = config.has_continuous_action_space

        if self.has_continuous_action_space:
            self.action_std = config.init_action_std

        self.gamma = config.gamma
        self.lamda = config.lamb
        self.eps_clip = config.eps_clip
        self.K_epochs = config.K_epochs

        self.buffer = RolloutBuffer()
        self.device = config.device

        self.policy = ActorCritic(config).to(config.device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': config.lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': config.lr_critic}
        ])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=config.lr_decay_step_size, gamma=config.lr_decay_gamma)

        self.policy_old = ActorCritic(config).to(config.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()
        self.total_loss = {"surr": [], "vloss": [], "dist": []}

    def pre_train(self, states, actions):
        action_mean = self.policy.actor(states)
        loss = self.MseLoss(action_mean, actions)
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()
        self.scheduler.step(None)

        return loss.mean().item()

    def set_action_std(self, new_action_std):

        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)

        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")

        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if self.action_std <= min_action_std:
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")

        print("--------------------------------------------------------------------------------------------")

    def select_action(self, state):

        if self.has_continuous_action_space:
            with torch.no_grad():
                # state = torch.FloatTensor(state).to(self.device)
                state = torch.tensor(state, dtype=torch.float32).to(self.device)
                action, action_logprob = self.policy_old.act(state)
            # self.buffer.states.append(state)
            # self.buffer.actions.append(action)
            # self.buffer.logprobs.append(action_logprob)

            return action.detach().cpu().numpy().flatten(), action_logprob.detach().cpu().numpy().flatten()

        else:
            with torch.no_grad():
                # state = torch.FloatTensor(state).to(self.device)
                state = torch.tensor(state, dtype=torch.float32).to(self.device)
                action, action_logprob = self.policy_old.act(state)
            # self.buffer.states.append(state)
            # self.buffer.actions.append(action)
            # self.buffer.logprobs.append(action_logprob)

            return action.item(), action_logprob.item()

    def cal_mc_rewards(self):
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            self.buffer.mc_rewards.insert(0, discounted_reward)

    def update(self, f_learner):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        # old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        # old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        # old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)
        old_states = torch.tensor(np.array(self.buffer.states), dtype=torch.float32).to(self.device)
        old_actions = torch.tensor(np.array(self.buffer.actions), dtype=torch.float32).to(self.device)
        old_logprobs = torch.tensor(np.array(self.buffer.logprobs), dtype=torch.float32).to(self.device)
        """
        old_states torch.Size([51, 2990])
        old_actions torch.Size([51, 123])
        old_logprobs torch.Size([51, 1])
        """
        # print('old_states', old_states.shape)
        # print('old_actions', old_actions.shape)
        # print('old_logprobs', old_logprobs.shape)

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.squeeze(-1))

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            surr_obj = -torch.min(surr1, surr2)
            self.total_loss["surr"].append(torch.mean(surr_obj).item())

            v_loss = 0.5 * self.MseLoss(state_values, rewards)
            self.total_loss["vloss"].append(v_loss.item())

            regular = - 0.01 * dist_entropy
            self.total_loss["dist"].append(torch.mean(regular).item())
            loss = surr_obj + v_loss + regular

            print('actor loss', surr_obj.mean().item())
            print('critic loss', v_loss.mean().item())
            if self.config.output_res:
                f_learner.write(' actor loss: %f\n' % surr_obj.mean().item())
                f_learner.write('critic loss: %f\n\n' % v_loss.mean().item())
                f_learner.flush()

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.gradient_clip)
            self.optimizer.step()
            self.scheduler.step(None)

        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def gae_update(self, f_learner):
        """
            old_states torch.Size([128, 2990])
            old_actions torch.Size([128, 123])
            old_logprobs torch.Size([128, 1])
            old_rewards torch.Size([128])
            old_dones torch.Size([128])
            old_next_states torch.Size([128, 2990])
        """
        old_states = torch.tensor(np.array(self.buffer.states), dtype=torch.float32).to(self.device)
        old_actions = torch.tensor(np.array(self.buffer.actions), dtype=torch.float32).to(self.device)
        old_logprobs = torch.tensor(np.array(self.buffer.logprobs), dtype=torch.float32).to(self.device)
        old_rewards = torch.tensor(np.array(self.buffer.rewards), dtype=torch.float32).to(self.device)
        old_dones = torch.tensor(np.array(self.buffer.is_terminals), dtype=torch.float32).to(self.device)
        old_next_states = torch.tensor(np.array(self.buffer.next_states), dtype=torch.float32).to(self.device)
        # Calculate the advantage using GAE
        advantages = []
        gae = 0
        with torch.no_grad():  # adv and v_target have no gradient
            """
                vs torch.Size([128, 1])
                vsn torch.Size([128, 1])
                advantages torch.Size([128, 1])
                v_target torch.Size([128, 1])
            """
            vs = self.policy.get_critic_value(old_states)
            vsn = self.policy.get_critic_value(old_next_states)
            deltas = old_rewards + self.gamma * (-old_dones + 1.0) * vsn - vs
            for delta, d in zip(reversed(deltas.flatten().numpy()), reversed(old_dones.flatten().numpy())):
                gae = delta + self.gamma * self.lamda * gae * (1.0 - d)
                advantages.insert(0, gae)
            advantages = torch.tensor(advantages, dtype=torch.float).view(-1, 1)
            v_target = advantages + vs
            # if self.use_adv_norm:  # Trick 1:advantage normalization
            #     adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Random sampling and no repetition. 'False' indicates that training will continue even if the number of samples in the last time is less than mini_batch_size
            if self.config.use_mini_batch:
                for index in BatchSampler(SubsetRandomSampler(range(self.config.batch_size)), self.config.mini_batch_size, False):
                    now_logprobs, dist_entropy = self.policy.get_actor_logprob(old_states[index], old_actions[index])
                    """
                        now_logprobs torch.Size([32, 1])
                        dist_entropy torch.Size([32, 1])
                        surr1 torch.Size([32, 1])
                        surr2 torch.Size([32, 1])
                        state_values torch.Size([32, 1])
                        surr_obj torch.Size([32, 1])
                        v_loss torch.Size([])
                        regular torch.Size([32, 1])
                        loss torch.Size([32, 1])
                    """
                    ratios = torch.exp(now_logprobs - old_logprobs[index])
                    surr1 = ratios * advantages[index]
                    surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages[index]

                    # final loss of clipped objective PPO
                    surr_obj = -torch.min(surr1, surr2)
                    self.total_loss["surr"].append(torch.mean(surr_obj).item())

                    state_values = self.policy.get_critic_value(old_states[index])
                    v_loss = 0.5 * self.MseLoss(state_values, v_target[index])
                    self.total_loss["vloss"].append(v_loss.item())

                    regular = - 0.01 * dist_entropy
                    self.total_loss["dist"].append(torch.mean(regular).item())
                    loss = surr_obj + v_loss + regular

                    print('actor loss', surr_obj.mean().item())
                    print('critic loss', v_loss.mean().item())
                    if self.config.output_res:
                        f_learner.write(' actor loss: %f\n' % surr_obj.mean().item())
                        f_learner.write('critic loss: %f\n\n' % v_loss.mean().item())
                        f_learner.flush()

                    # take gradient step
                    self.optimizer.zero_grad()
                    loss.mean().backward()
                    torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.gradient_clip)
                    self.optimizer.step()
                    self.scheduler.step(None)
            else:
                now_logprobs, dist_entropy = self.policy.get_actor_logprob(old_states, old_actions)
                """
                    now_logprobs torch.Size([128, 1])
                    dist_entropy torch.Size([128, 1])
                    surr1 torch.Size([128, 1])
                    surr2 torch.Size([128, 1])
                    state_values torch.Size([128, 1])
                    surr_obj torch.Size([128, 1])
                    v_loss torch.Size([])
                    regular torch.Size([128, 1])
                    loss torch.Size([128, 1])
                """
                ratios = torch.exp(now_logprobs - old_logprobs)
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

                # final loss of clipped objective PPO
                surr_obj = -torch.min(surr1, surr2)
                self.total_loss["surr"].append(torch.mean(surr_obj).item())

                state_values = self.policy.get_critic_value(old_states)
                v_loss = 0.5 * self.MseLoss(state_values, v_target)
                self.total_loss["vloss"].append(v_loss.item())

                regular = - 0.01 * dist_entropy
                self.total_loss["dist"].append(torch.mean(regular).item())
                loss = surr_obj + v_loss + regular

                print('actor loss', surr_obj.mean().item())
                print('critic loss', v_loss.mean().item())
                if self.config.output_res:
                    f_learner.write(' actor loss: %f\n' % surr_obj.mean().item())
                    f_learner.write('critic loss: %f\n\n' % v_loss.mean().item())
                    f_learner.flush()

                # take gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.gradient_clip)
                self.optimizer.step()
                self.scheduler.step(None)
        self.policy_old.load_state_dict(self.policy.state_dict())
        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy.state_dict(), checkpoint_path)

    def save_all(self, update_round, checkpoint_path, state_norm=None):
        if not os.path.isdir(self.config.model_save_dir):
            os.makedirs(self.config.model_save_dir)
        torch.save({
            'update_round': update_round,
            'network': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'state_norm': state_norm.get_info() if state_norm is not None else None,
        }, checkpoint_path)

    def load(self, state_norm=None):
        save_model = torch.load(self.config.model_load_path, map_location=lambda storage, loc: storage)
        self.policy_old.load_state_dict(save_model['network'])
        self.policy.load_state_dict(save_model['network'])
        if self.config.load_state_normalization and state_norm is not None:
            save_state_normal = torch.load(self.config.state_normal_load_path, map_location=lambda storage, loc: storage)
            state_norm.set_info(save_state_normal['state_norm'])

    def get_buffer_size(self):
        return self.buffer.get_buffer_size()

    def add_data(self, obs, action, logprob, reward, done, next_obs=None):
        self.buffer.add_data(obs, action, logprob, reward, done, next_obs)

    def get_old_actor_logprob(self, state, action):
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        action = torch.tensor(action, dtype=torch.float32).to(self.device)

        return self.policy_old.get_actor_logprob(state, action)


class PPOAgent(BaseAgent):
    def __init__(self, settings, config):
        BaseAgent.__init__(self, settings)
        # set basic info for PPO agent
        self.settings = settings
        self.config = config
        self.device = config.device
        with open(config.mean_std_info_path, 'rb') as f:
            self.mean_std_info = pickle.load(f)
        # set policy
        self.policy = PPO(config)
        self.mask = None
        # record data file
        self.f_actor = None
        self.f_learner = None
        # state normalization
        self.state_norm = StateNormalization(self.config)
        # test env state
        self.state_check = StateChecker(self.config)

    def process_state(self, obs):
        settings = self.settings
        # gen states
        gens = np.concatenate([
            (np.array(obs.gen_p) - np.array(settings.gen_p_max)) / (np.array(settings.gen_p_max) - np.array(settings.gen_p_min)),
            (np.array(obs.gen_q) - np.array(settings.gen_q_max)) / (np.array(settings.gen_q_max) - np.array(settings.gen_q_min)),
            (np.array(obs.gen_v) - np.array(settings.gen_v_max)) / (np.array(settings.gen_v_max) - np.array(settings.gen_v_min)),
            (np.array(obs.target_dispatch) - np.array(settings.gen_p_max)) / (np.array(settings.gen_p_max) - np.array(settings.gen_p_min)),
            (np.array(obs.actual_dispatch) - np.array(settings.gen_p_max)) / (np.array(settings.gen_p_max) - np.array(settings.gen_p_min)),
        ], axis=0)
        # print('gens', gens.shape)
        # load states
        loads = np.concatenate([
            (np.array(obs.ld_p) - np.array(self.mean_std_info['ld_p']['mean'])) / np.array(self.mean_std_info['ld_p']['std']),
            (np.array(obs.adjld_p) - np.array(self.mean_std_info['adjld_p']['mean'])) / np.array(self.mean_std_info['adjld_p']['std']),
            # [(np.array(obs.stoenergy_p) - np.array(self.mean_std_info['stoenergy_p']['mean'])) / np.array(self.mean_std_info['stoenergy_p']['std'])],
            (np.array(obs.stoenergy_p) + np.array(settings.stoenergy_dischargerate_max)) / (np.array(settings.stoenergy_chargerate_max) + np.array(settings.stoenergy_dischargerate_max)),
            (np.array(obs.ld_q) - np.array(self.mean_std_info['ld_q']['mean'])) / np.array(self.mean_std_info['ld_q']['std']),
            (np.array(obs.ld_v) - self.mean_std_info['ld_v']['mean']) / self.mean_std_info['ld_v']['std'],
            # [np.array(obs.ld_p)],
            # [np.array(obs.adjld_p)],
            # [np.array(obs.stoenergy_p)],
            # [np.array(obs.ld_q)],
            # [np.array(obs.ld_v)],
        ], axis=0)
        # print('loads', loads.shape)
        # line states
        lines = np.concatenate([
            (np.array(obs.p_or) - self.mean_std_info['p_or']['mean']) / self.mean_std_info['p_or']['std'],
            (np.array(obs.q_or) - self.mean_std_info['q_or']['mean']) / self.mean_std_info['q_or']['std'],
            (np.array(obs.v_or) - self.mean_std_info['v_or']['mean']) / self.mean_std_info['v_or']['std'],
            (np.array(obs.a_or) - self.mean_std_info['a_or']['mean']) / self.mean_std_info['a_or']['std'],
            (np.array(obs.p_ex) - self.mean_std_info['p_ex']['mean']) / self.mean_std_info['p_ex']['std'],
            (np.array(obs.q_ex) - self.mean_std_info['q_ex']['mean']) / self.mean_std_info['q_ex']['std'],
            (np.array(obs.v_ex) - self.mean_std_info['v_ex']['mean']) / self.mean_std_info['v_ex']['std'],
            (np.array(obs.a_ex) - self.mean_std_info['a_ex']['mean']) / self.mean_std_info['a_ex']['std'],
            # [np.array(obs.p_or)],
            # [np.array(obs.q_or)],
            # [np.array(obs.v_or)],
            # [np.array(obs.a_or)],
            # [np.array(obs.p_ex)],
            # [np.array(obs.q_ex)],
            # [np.array(obs.v_ex)],
            # [np.array(obs.a_ex)],
            np.array(obs.line_status).astype(np.float32),
            (np.array(obs.grid_loss) - self.mean_std_info['grid_loss']['mean']) / self.mean_std_info['grid_loss']['std'],
            # np.array(obs.flag).astype(np.float32),
            (np.array(obs.steps_to_reconnect_line) - settings.max_steps_to_reconnect_line) / settings.max_steps_to_reconnect_line,
            (np.array(obs.count_soft_overflow_steps) - settings.max_steps_soft_overflow) / settings.max_steps_soft_overflow,
            (np.array(obs.rho) - self.mean_std_info['rho']['mean']) / self.mean_std_info['rho']['std'],
        ], axis=0)
        # print('lines', lines.shape)
        # other info
        other = np.concatenate([
            obs.gen_status.astype(np.float32),
            (np.array(obs.steps_to_recover_gen) - settings.max_steps_to_recover_gen) / settings.max_steps_to_recover_gen,
            (np.array(obs.steps_to_close_gen) - settings.max_steps_to_close_gen) / settings.max_steps_to_close_gen,
            (np.array(obs.curstep_renewable_gen_p_max) - self.mean_std_info['curstep_renewable_gen_p_max']['mean']) / self.mean_std_info['curstep_renewable_gen_p_max']['std'],
            (np.array(obs.nextstep_renewable_gen_p_max) - self.mean_std_info['nextstep_renewable_gen_p_max']['mean']) / self.mean_std_info['nextstep_renewable_gen_p_max']['std'],
            (np.array(obs.curstep_ld_p) - self.mean_std_info['nextstep_ld_p']['mean']) / self.mean_std_info['nextstep_ld_p']['std'],
            (np.array(obs.nextstep_ld_p) - self.mean_std_info['nextstep_ld_p']['mean']) / self.mean_std_info['nextstep_ld_p']['std'],
            (np.array(obs.total_adjld) + np.array(settings.adjld_capacity)) / 2 * np.array(settings.adjld_capacity),
            np.array(obs.total_stoenergy) / np.array(settings.stoenergy_capacity),
        ], axis=0)
        # print('other', other.shape)
        normal_obs = np.concatenate([gens, loads, lines, other])
        # print('normal_obs', normal_obs.shape)

        return normal_obs

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
        # reflect action to action space if necessary
        if config.action_restrict_method == 'reflect':
            adjust_gen_p = (gen_p_action_space.high - gen_p_action_space.low) * adjust_gen_p + gen_p_action_space.low
            adjust_gen_v = (gen_v_action_space.high - gen_v_action_space.low) * adjust_gen_v + gen_v_action_space.low
            adjust_adjld_p = (adjld_p_action_space.high - adjld_p_action_space.low) * adjust_adjld_p + adjld_p_action_space.low
            adjust_stoenergy_p = (stoenergy_p_action_space.high - stoenergy_p_action_space.low) * adjust_stoenergy_p + stoenergy_p_action_space.low
        # clip action to action space
        before_clip = adjust_gen_p
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
        # adjust_gen_p[settings.balanced_id] = 0.
        # interact_action: 与环境交互的动作, 按类别进行索引
        # clip_action: 存入缓冲池的动作数据, 直接进行拼接
        interact_action = form_action(adjust_gen_p, adjust_gen_v, adjust_adjld_p, adjust_stoenergy_p)
        clip_action = np.concatenate([adjust_gen_p, adjust_gen_v, adjust_adjld_p, adjust_stoenergy_p], axis=0)

        return interact_action, clip_action

    def process_reward(self, next_obs, reward, info, round, action=None, clip_action=None):
        # if train survive, mask optimize reward and give agent survive reward
        if self.config.train_survive:
            reward = self.config.reward_for_survive
        # if agent fail, give agent a fail reward(fixed fail reward or relative to survive timesteps)
        if 'fail_info' in info and info['fail_info'] == 'grid is not converged':
            if self.config.train_target_survive:
                reward += round - self.config.train_target_survive
            else:
                reward += self.config.punish_fail
        elif 'fail_info' in info and info['fail_info'] == 'balance gen out of bound':
            if self.config.train_target_survive:
                reward += round - self.config.train_target_survive
            else:
                reward += self.config.punish_fail
        if self.config.punish_out_actionspace and action is not None and clip_action is not None:
            reward -= np.fabs(self.config.punish_out_actionspace_rate * np.mean(action - clip_action))
        if self.config.punish_balance_out_range:
            balanced_gen_p = next_obs.gen_p[settings.balanced_id]
            min_val = settings.min_balanced_gen_bound * settings.gen_p_min[settings.balanced_id]
            max_val = settings.max_balanced_gen_bound * settings.gen_p_max[settings.balanced_id]
            mid_val = (max_val + min_val) / 2
            health_range = (max_val - min_val) / 6
            if not (mid_val - health_range < balanced_gen_p < mid_val + health_range):
                reward -= self.config.punish_balance_out_range_rate * abs(balanced_gen_p - mid_val)

        return reward

    def act(self, obs, reward, done=False):
        """
        Returns:
            # interact_action: 与环境交互的动作, 按类别进行索引
            # clip_action: 经过clip处理后, 存入缓冲池的动作数据, 直接进行拼接
            # logprob: 经过clip处理后, action的logprop
            # norm_obs: 归一化后的状态
        """
        norm_obs = self.process_state(obs)
        action, logprob = self.policy.select_action(norm_obs)
        interact_action, clip_action = self.process_action(obs, action)
        logprob, _ = self.policy.get_old_actor_logprob(norm_obs, clip_action)
        logprob = logprob.detach().numpy()

        return interact_action, clip_action, logprob, norm_obs

    def gae_act(self, ori_obs, norm_obs):
        """
        Args:
            ori_obs: 对象类型, 环境交互生成的原始状态, 包含所有信息
            norm_obs: 数组类型, 由原始状态进行加工筛选和归一化后的状态
        Returns:
            interact_action: 与环境交互的动作, 按类别进行索引
            clip_action: 存入缓冲池的动作数据, 直接进行拼接
        """
        action, logprob = self.policy.select_action(norm_obs)
        interact_action, clip_action = self.process_action(ori_obs, action)
        clip_logprob, _ = self.policy.get_old_actor_logprob(norm_obs, clip_action)
        clip_logprob = clip_logprob.detach().numpy()

        return interact_action, action, clip_action, logprob, clip_logprob

    def train(self):
        rounds = []
        update_times = 0
        # output res file
        if self.config.output_res:
            self.f_actor = open(self.config.res_file_dir + 'actor_' + self.config.res_file_name, 'w')
            self.f_learner = open(self.config.res_file_dir + 'learner_' + self.config.res_file_name, 'w')
        for episode in range(self.config.max_episode):
            rtg = 0.0
            print('---------------------------- episode ', episode, '----------------------------')
            if self.config.output_res:
                self.f_actor.write('---------------------------- episode %d ----------------------------\n' % episode)
                self.f_actor.flush()
            env = Environment(self.settings, "EPRIReward")
            obs = env.reset()
            reward = 0.0
            done = False
            # while not done:
            for timestep in range(self.config.max_timestep):
                # print('----------- step ', timestep, '-----------')
                interact_action, clip_action, logprob, norm_obs = self.act(obs, reward, done)
                obs, reward, done, info = env.step(interact_action)
                reward = self.process_reward(obs, reward, info, timestep)
                rtg += reward
                # print('reward', reward)
                self.policy.add_data(norm_obs, clip_action, logprob, reward, done)
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
            # update every max_buffer_size steps
            if self.policy.get_buffer_size() >= self.config.max_buffer_size:
                update_times += 1
                self.policy.update(self.f_learner)
                if update_times % self.config.action_std_decay_freq == 0:
                    self.policy.decay_action_std(self.config.action_std_decay_rate, self.config.min_action_std)
                if self.config.save_model and update_times % self.config.model_save_freq == 0:
                    self.policy.save_all(update_times, self.config.model_save_dir + str(update_times) + '_save_model.pth')
        if self.config.output_res:
            self.f_actor.close()
            self.f_learner.close()

    def gae_train(self):
        rounds = []
        update_times = 0
        # load pretrain model(and state normal)
        if self.config.load_model:
            self.policy.load(self.state_norm)

        # output res file
        if self.config.output_res:
            self.f_actor = open(self.config.res_file_dir + 'actor_' + self.config.res_file_name, 'w')
            self.f_learner = open(self.config.res_file_dir + 'learner_' + self.config.res_file_name, 'w')
            print_config(self.config, self.f_actor)
        for episode in range(self.config.max_episode):
            rtg = 0.0
            print('---------------------------- episode ', episode, '----------------------------')
            if self.config.output_res:
                self.f_actor.write('---------------------------- episode %d ----------------------------\n' % episode)
                self.f_actor.flush()
            env = Environment(self.settings, "EPRIReward")
            obs = env.reset()
            norm_obs = self.state_norm(obs)
            reward = 0.0
            done = False
            # while not done:
            for timestep in range(self.config.max_timestep):
                # print('----------- step ', timestep, '-----------')
                interact_action, action, clip_action, logprob, clip_logprob = self.gae_act(obs, norm_obs)
                next_obs, reward, done, info = env.step(interact_action)

                # self.state_check.check_state_change(obs, interact_action, next_obs)

                next_norm_obs = self.state_norm(next_obs)
                reward = self.process_reward(next_obs, reward, info, timestep, action, clip_action)
                rtg += reward
                # print('reward', reward)
                if self.config.save_clip_action:
                    self.policy.add_data(norm_obs, clip_action, clip_logprob, reward, done, next_norm_obs)
                else:
                    self.policy.add_data(norm_obs, action, logprob, reward, done, next_norm_obs)
                # update every max_buffer_size steps
                if self.policy.get_buffer_size() >= self.config.batch_size:
                    update_times += 1
                    self.policy.gae_update(self.f_learner)
                    # decay action sample std every interval
                    if update_times % self.config.action_std_decay_freq == 0:
                        self.policy.decay_action_std(self.config.action_std_decay_rate, self.config.min_action_std)
                    # save model every interval
                    if self.config.save_model and update_times % self.config.model_save_freq == 0:
                        self.policy.save_all(update_times, self.config.model_save_dir + str(update_times) + '_save_model.pth', self.state_norm)
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
                # update next obs to now obs
                obs = next_obs
                norm_obs = next_norm_obs
        if self.config.output_res:
            self.f_actor.close()
            self.f_learner.close()

    def pretrain(self, all_states, all_actions):
        state_action_dataset = StateActionDataset(self.config, all_states, all_actions)
        if self.config.output_res:
            self.f_actor = open(self.config.res_file_dir + 'pretrain_' + self.config.res_file_name, 'w')

        for step in range(self.config.max_timestep):
            states, actions = state_action_dataset.get_batch()
            loss = self.policy.pre_train(states, actions)
            print('loss', loss)
            if self.config.output_res:
                self.f_actor.write('loss ' + str(loss) + '\n')
                self.f_actor.flush()
            if step % self.config.model_save_freq == 0:
                self.policy.save_all(0, self.config.model_save_dir + str(step) + '_pretrain_model.pth', state_action_dataset.state_norm)

        if self.config.output_res:
            self.f_actor.close()

    def load_model(self, path):
        self.policy.load(path)
