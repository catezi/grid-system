# -*- coding: UTF-8 -*-
import numpy as np

from Agent.DoNothingAgent import DoNothingAgent
from Agent.RandomAgent import RandomAgent
from Environment.base_env import Environment
from utilize.settings import settings

def run_task(my_agent):
    for episode in range(max_episode):
        print('------ episode ', episode)
        env = Environment(settings, "EPRIReward")
        print('------ reset ')
        obs = env.reset()
        reward = 0.0
        done = False
        
        # while not done:
        for timestep in range(max_timestep):
            print('------step: ', timestep)
            action = my_agent.act(obs, reward, done)
            # action['adjust_adjld_p'][0] = 15
            # xx = obs.action_space['adjust_adjld_p']
            # action['adjust_stoenergy_p'] = np.zeros(len(settings.stoenergy_ids))
            obs, reward, done, info = env.step(action)
            print('reward=', reward)
            # print('done=', done)
            # print('info=', info)
            # print(obs.line_status)
            if done:
                print('info', info)
                print('fail flag', obs.flag)
                break


if __name__ == "__main__":
    max_timestep = 10  # 最大时间步数
    max_episode = 20  # 回合数
    my_agent = RandomAgent(settings)
    # my_agent = DoNothingAgent(settings)
    run_task(my_agent)
