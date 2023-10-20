# -*- coding: UTF-8 -*-
import os
import pickle
import numpy as np
import pandas as pd

from Agent.DoNothingAgent import DoNothingAgent
from Agent.RandomAgent import RandomAgent
from Environment.base_env import Environment
from utilize.settings import settings


class Buffer:
    def __init__(self):
        self.ld_v = []
        self.p_or = []
        self.q_or = []
        self.v_or = []
        self.a_or = []
        self.p_ex = []
        self.q_ex = []
        self.v_ex = []
        self.a_ex = []
        self.grid_loss = []
        self.rho = []

    def add_data(self, obs):
        self.ld_v.append(obs.ld_v)
        self.p_or.append(obs.p_or)
        self.q_or.append(obs.q_or)
        self.v_or.append(obs.v_or)
        self.a_or.append(obs.a_or)
        self.p_ex.append(obs.p_ex)
        self.q_ex.append(obs.q_ex)
        self.v_ex.append(obs.v_ex)
        self.a_ex.append(obs.a_ex)
        self.grid_loss.append(obs.grid_loss)
        self.rho.append(obs.rho)

    def clear(self):
        self.ld_v = []
        self.p_or = []
        self.q_or = []
        self.v_or = []
        self.a_or = []
        self.p_ex = []
        self.q_ex = []
        self.v_ex = []
        self.a_ex = []
        self.grid_loss = []
        self.rho = []

    def get_length(self):
        return len(self.ld_v)

    def statistics(self):
        all_ld_v = {
            'mean': np.mean(np.array(self.ld_v)),
            'std': np.std(np.array(self.ld_v)),
        }

        all_p_or = {
            'mean': np.mean(np.array(self.p_or)),
            'std': np.std(np.array(self.p_or)),
        }
        all_q_or = {
            'mean': np.mean(np.array(self.q_or)),
            'std': np.std(np.array(self.q_or)),
        }
        all_v_or = {
            'mean': np.mean(np.array(self.v_or)),
            'std': np.std(np.array(self.v_or)),
        }
        all_a_or = {
            'mean': np.mean(np.array(self.a_or)),
            'std': np.std(np.array(self.a_or)),
        }
        all_p_ex = {
            'mean': np.mean(np.array(self.p_ex)),
            'std': np.std(np.array(self.p_ex)),
        }
        all_q_ex = {
            'mean': np.mean(np.array(self.q_ex)),
            'std': np.std(np.array(self.q_ex)),
        }
        all_v_ex = {
            'mean': np.mean(np.array(self.v_ex)),
            'std': np.std(np.array(self.v_ex)),
        }
        all_a_ex = {
            'mean': np.mean(np.array(self.a_ex)),
            'std': np.std(np.array(self.a_ex)),
        }
        all_grid_loss = {
            'mean': np.mean(np.array(self.grid_loss)),
            'std': np.std(np.array(self.grid_loss)),
        }
        all_rho = {
            'mean': np.mean(np.array(self.rho)),
            'std': np.std(np.array(self.rho)),
        }

        return all_ld_v, all_p_or, all_q_or, all_v_or, all_a_or, \
               all_p_ex, all_q_ex, all_v_ex, all_a_ex, all_grid_loss, all_rho


def statistics():
    # statistics ld_p, ld_q nextstep_ld_p
    ld_p_data = pd.read_csv(settings.ld_p_filepath)
    ld_q_data = pd.read_csv(settings.ld_q_filepath)
    record_ld_data = {
        'ld': {
            'ld_p': [],
            'ld_q': [],
        },
        'adjld': {
            'ld_p': [],
            'ld_q': [],
        },
        'stoenergy': {
            'ld_p': [],
            'ld_q': [],
        }
    }
    adjld_name_set = set(settings.adjld_name)
    stoenergy_name_set = set(settings.stoenergy_name)
    for ld_name in settings.ld_name:
        if ld_name in adjld_name_set:
            record_ld_data['adjld']['ld_p'].extend(list(ld_p_data.loc[:, ld_name].values))
            record_ld_data['adjld']['ld_q'].extend(list(ld_q_data.loc[:, ld_name].values))
        elif ld_name in stoenergy_name_set:
            record_ld_data['stoenergy']['ld_p'].extend(list(ld_p_data.loc[:, ld_name].values))
            record_ld_data['stoenergy']['ld_q'].extend(list(ld_q_data.loc[:, ld_name].values))
        else:
            record_ld_data['ld']['ld_p'].extend(list(ld_p_data.loc[:, ld_name].values))
            record_ld_data['ld']['ld_q'].extend(list(ld_q_data.loc[:, ld_name].values))
    res_ld_data = {
        'ld': {
            'ld_p': [],
            'ld_q': [],
        },
        'adjld': {
            'ld_p': [],
            'ld_q': [],
        },
        'stoenergy': {
            'ld_p': [],
            'ld_q': [],
        }
    }
    for k1 in record_ld_data:
        for k2 in record_ld_data[k1]:
            res_ld_data[k1][k2] = [
                np.mean(record_ld_data[k1][k2]),
                1 if np.std(record_ld_data[k1][k2]) == 0 else np.std(record_ld_data[k1][k2])
            ]
    all_ld_p = {'mean': [], 'std': []}
    all_ld_q = {'mean': [], 'std': []}
    all_adjld_p = {'mean': res_ld_data['adjld']['ld_p'][0], 'std': res_ld_data['adjld']['ld_p'][1]}
    all_stoenergy_p = {'mean': res_ld_data['stoenergy']['ld_p'][0], 'std': res_ld_data['stoenergy']['ld_p'][1]}
    for ld_name in settings.ld_name:
        if ld_name in adjld_name_set:
            all_ld_p['mean'].append(res_ld_data['adjld']['ld_p'][0])
            all_ld_p['std'].append(res_ld_data['adjld']['ld_p'][1])
            all_ld_q['mean'].append(res_ld_data['adjld']['ld_q'][0])
            all_ld_q['std'].append(res_ld_data['adjld']['ld_q'][1])
        elif ld_name in stoenergy_name_set:
            all_ld_p['mean'].append(res_ld_data['stoenergy']['ld_p'][0])
            all_ld_p['std'].append(res_ld_data['stoenergy']['ld_p'][1])
            all_ld_q['mean'].append(res_ld_data['stoenergy']['ld_q'][0])
            all_ld_q['std'].append(res_ld_data['stoenergy']['ld_q'][1])
        else:
            all_ld_p['mean'].append(res_ld_data['ld']['ld_p'][0])
            all_ld_p['std'].append(res_ld_data['ld']['ld_p'][1])
            all_ld_q['mean'].append(res_ld_data['ld']['ld_q'][0])
            all_ld_q['std'].append(res_ld_data['ld']['ld_q'][1])
    all_nextstep_ld_p = all_ld_p

    # statistics curstep_renewable_gen_p_max nextstep_renewable_gen_p_max
    renewable_gen_p_max_data = pd.read_csv(settings.max_renewable_gen_p_filepath)
    all_renewable_gen_p_max = []
    for index, row in renewable_gen_p_max_data.iteritems():
        all_renewable_gen_p_max.extend((list(row.values)))
    all_curstep_renewable_gen_p_max = {
        'mean': np.mean(np.array(all_renewable_gen_p_max)),
        'std': np.std(np.array(all_renewable_gen_p_max))
    }
    all_nextstep_renewable_gen_p_max = {
        'mean': np.mean(np.array(all_renewable_gen_p_max)),
        'std': np.std(np.array(all_renewable_gen_p_max))
    }

    return all_ld_p, all_adjld_p, all_stoenergy_p, all_ld_q, all_nextstep_ld_p, all_curstep_renewable_gen_p_max, all_nextstep_renewable_gen_p_max


def sampling(my_agent):
    for episode in range(max_episode):
        env = Environment(settings, "EPRIReward")
        obs = env.reset()
        buffer.add_data(obs)
        reward = 0.0
        done = False
        # while not done:
        for timestep in range(max_timestep):
            action = my_agent.act(obs, reward, done)
            # action['adjust_adjld_p'][0] = 15
            # xx = obs.action_space['adjust_adjld_p']
            # action['adjust_stoenergy_p'] = np.zeros(len(settings.stoenergy_ids))
            obs, reward, done, info = env.step(action)
            buffer.add_data(obs)
            # print('done=', done)
            # print('info=', info)
            # print(obs.line_status)
            if done:
                break
        if buffer.get_length() >= global_max_timestep:
            break


if __name__ == "__main__":
    # statistics
    ld_p, adjld_p, stoenergy_p, ld_q, nextstep_ld_p, curstep_renewable_gen_p_max, nextstep_renewable_gen_p_max = statistics()
    # sampling
    buffer = Buffer()
    max_timestep = 1000  # 最大时间步数
    max_episode = 10000  # 回合数
    global_max_timestep = 10000
    my_agent = RandomAgent(settings)
    sampling(my_agent)
    ld_v, p_or, q_or, v_or, a_or, p_ex, q_ex, v_ex, a_ex, grid_loss, rho = buffer.statistics()
    # save info
    save_dataset_path = './data/'
    all_info = {
        # statistics res
        'ld_p': ld_p,
        'adjld_p': adjld_p,
        'stoenergy_p': stoenergy_p,
        'ld_q': ld_q,
        'nextstep_ld_p': nextstep_ld_p,
        'curstep_renewable_gen_p_max': curstep_renewable_gen_p_max,
        'nextstep_renewable_gen_p_max': nextstep_renewable_gen_p_max,
        # sampling res
        'ld_v': ld_v,
        'p_or': p_or,
        'q_or': q_or,
        'v_or': v_or,
        'a_or': a_or,
        'p_ex': p_ex,
        'q_ex': q_ex,
        'v_ex': v_ex,
        'a_ex': a_ex,
        'grid_loss': grid_loss,
        'rho': rho,
    }
    if not os.path.isdir(save_dataset_path):
        os.makedirs(save_dataset_path)
    f = open(save_dataset_path + 'mean_std_info.pkl', 'wb')
    pickle.dump(all_info, f)
    f.close()
