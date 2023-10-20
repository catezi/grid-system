import logging

from Observation.observation import Observation
from Reward.rewards import *
from utilize.read_forecast_value import ForecastReader
from utilize.line_cutting import Disconnect
from utilize.action_space import ActionSpace
from utilize.legal_action import *
import example
import copy
import numpy as np
import warnings
import math

warnings.filterwarnings('ignore')


class Environment:
    def __init__(self, settings, reward_type="EPRIReward"):
        self.settings = copy.deepcopy(settings)
        self.forecast_reader = ForecastReader(self.settings)
        self.reward_type = reward_type
        self.done = True
        self.info = None
        self.action_space_cls = ActionSpace(settings)

    def reset_attr(self):
        # Reset attr in the base env
        self.grid = example.Print()
        self.done = False
        self.info = None
        self.timestep = 0
        # 机组开关机状态，1 表示开机，0 表示关机
        self.gen_status = np.ones(self.settings.gen_num)
        # 关机机组允许重启的剩余时间步数，0 表示机组是开启状态
        self.steps_to_recover_gen = np.zeros(self.settings.gen_num, dtype=int)
        # 重启机组允许关机的剩余时间步数
        self.steps_to_close_gen = np.zeros(self.settings.gen_num, dtype=int)
        # 已断开支路恢复连接的剩余时间步数
        self.steps_to_reconnect_line = np.zeros(self.settings.ln_num, dtype=int)
        # 支路已连续软过载的时间步数
        self.count_soft_overflow_steps = np.zeros(self.settings.ln_num, dtype=int)
        # 各可调负荷的负荷调整值累加（近288时间步内）
        self.total_adjld = np.zeros(self.settings.adjld_num, dtype=float)  # 调整值之和
        # 各储能设备剩余电量（各储能设备的充放电实际值累加）
        self.total_stoenergy = np.zeros(self.settings.stoenergy_num, dtype=float)  # 实际值之和

    def reset(self, seed=None, start_sample_idx=None):
        self.reset_attr()
        settings = self.settings
        grid = self.grid

        # Instead of using `np.random` use `self.np_random`.
        # It won't be affected when user using `np.random`.
        self.np_random = np.random.RandomState()
        if seed is not None:
            self.np_random.seed(seed=seed)

        self.disconnect = Disconnect(self.np_random, self.settings)

        if start_sample_idx is not None:
            self.sample_idx = start_sample_idx
        else:
            self.sample_idx = self.np_random.randint(0, settings.sample_num)
        assert settings.sample_num > self.sample_idx >= 0

        # Read self.sample_idx timestep data
        """
        NOTE:
            1. C++ read the data by the row number of the csv file;
            2. The first row of the csv file is the header.
        """
        row_idx = self.sample_idx + 1
        grid.readdata(row_idx, settings.ld_p_filepath, settings.ld_q_filepath, settings.gen_p_filepath,
                      settings.gen_q_filepath)
        injection_gen_p = self._round_p(grid.itime_unp[0])
        injection_adjld = [grid.itime_ldp[0][i] for i in settings.adjld_ids]
        injection_stoenergy = [grid.itime_ldp[0][i] for i in settings.stoenergy_ids]

        # 潮流计算
        grid.env_feedback(settings.name_index, injection_gen_p, [], row_idx, [], injection_adjld + injection_stoenergy)
        rounded_gen_p = self._round_p(grid.prod_p[0])
        rounded_ld_p = self._round_p(grid.load_p[0])

        self._update_gen_status(injection_gen_p)
        self._check_gen_status(injection_gen_p, rounded_gen_p)
        self.last_injection_gen_p = copy.deepcopy(injection_gen_p)
        rho = self._calc_rho(grid, settings)

        # 更新预测值
        curstep_renewable_gen_p_max, nextstep_renewable_gen_p_max = \
            self.forecast_reader.read_step_renewable_gen_p_max(self.sample_idx)
        # 检查新能源预测值.csv文件中数据是否有错
        for i in range(len(nextstep_renewable_gen_p_max)):
            assert nextstep_renewable_gen_p_max[i] >= 0, i

        # 检查储能设备初始值.csv文件中数据是否有错
        curstep_ld_p, nextstep_ld_p = self.forecast_reader.read_step_ld_p(self.sample_idx)
        nextstep_sto_p = [nextstep_ld_p[i] for i in self.settings.stoenergy_ids]
        for i in range(len(nextstep_sto_p)):
            assert nextstep_sto_p[i] == 0, str(i) + str(nextstep_sto_p[i])

        # 更新动作空间
        action_space = self.action_space_cls.update(grid, self.steps_to_recover_gen, self.steps_to_close_gen,
                                                    rounded_gen_p, rounded_ld_p,
                                                    nextstep_renewable_gen_p_max, nextstep_ld_p,
                                                    self.total_adjld, self.total_stoenergy)

        self.obs = Observation(
            grid=grid, timestep=self.timestep, action_space=action_space,
            steps_to_reconnect_line=self.steps_to_reconnect_line,
            count_soft_overflow_steps=self.count_soft_overflow_steps, rho=rho,
            gen_status=self.gen_status, steps_to_recover_gen=self.steps_to_recover_gen,
            steps_to_close_gen=self.steps_to_close_gen,
            curstep_renewable_gen_p_max=curstep_renewable_gen_p_max,
            nextstep_renewable_gen_p_max=nextstep_renewable_gen_p_max,
            rounded_gen_p=rounded_gen_p,
            curstep_ld_p=curstep_ld_p,
            nextstep_ld_p=nextstep_ld_p,
            total_adjld=self.total_adjld,
            total_stoenergy=self.total_stoenergy,
            last_injection_gen_p=self.last_injection_gen_p,
            disc_name=[], recover_name=[],
        )
        logging.info("info")
        return copy.deepcopy(self.obs)

    def step(self, act):
        if self.done:
            raise Exception("The env is game over, please reset.")
        settings = self.settings
        last_obs = self.obs
        grid = self.grid

        self._check_action(act)
        act['adjust_gen_p'] = self._round_p(act['adjust_gen_p'])

        # 计算动作注入值
        adjust_gen_p = act['adjust_gen_p']
        injection_gen_p = [adjust_gen_p[i] + last_obs.gen_p[i] for i in range(len(adjust_gen_p))]
        injection_gen_p = self._round_p(injection_gen_p)

        adjust_gen_v = act['adjust_gen_v']
        injection_gen_v = [adjust_gen_v[i] + last_obs.gen_v[i] for i in range(len(adjust_gen_v))]

        adjust_adjld_p = act['adjust_adjld_p']
        # injection_adjld_p = [adjust_adjld_p[i] + last_obs.adjld_p[i] for i in range(settings.adjld_num)]
        injection_adjld_p = [adjust_adjld_p[i] + last_obs.nextstep_ld_p[x] for i, x in enumerate(settings.adjld_ids)]

        adjust_stoenergy_p = act['adjust_stoenergy_p']
        # injection_stoenergy_p = [adjust_stoenergy_p[i] + last_obs.stoenergy_p[i] for i in range(settings.stoenergy_num)]
        injection_stoenergy_p = [adjust_stoenergy_p[i] for i in range(settings.stoenergy_num)]
        injection_ld = injection_adjld_p + injection_stoenergy_p

        # 判断动作合法性
        legal_flag, fail_info = is_legal(act, last_obs, settings)
        if not legal_flag:
            if hasattr(settings, 'ban_legal_check') and settings.ban_legal_check:
                self.info = fail_info
            else:
                self.done = True
                return self.return_res(fail_info)

        disc_name, self.steps_to_reconnect_line, self.count_soft_overflow_steps, new_recover_name, new_disc_name = self.disconnect.get_disc_name(last_obs)

        self.sample_idx += 1
        self.timestep += 1

        """
        NOTE:
            1. C++ read the data by the row number of the csv file;
            2. The first row of the csv file is the header.
        """
        row_idx = self.sample_idx + 1
        # Read the power data of the next step from .csv file
        grid.readdata(row_idx, settings.ld_p_filepath, settings.ld_q_filepath,
                      settings.gen_p_filepath, settings.gen_q_filepath)

        injection_gen_p = self._injection_auto_mapping(injection_gen_p, adjust_gen_p)

        # Update generator running status
        self._update_gen_status(injection_gen_p)

        # print('before', injection_gen_p[settings.balanced_id])
        # 潮流计算
        grid.env_feedback(grid.un_nameindex, injection_gen_p, injection_gen_v, row_idx, disc_name, injection_ld)
        # print('after', grid.prod_p[0][settings.balanced_id])

        flag, info = self.check_done(grid, settings)
        if flag:
            if hasattr(settings, 'ban_legal_check') and settings.ban_legal_check:
                self.info = info
            else:
                # pack next_obs before finish episode
                curstep_ld_p, nextstep_ld_p = self.forecast_reader.read_step_ld_p(self.sample_idx)
                self.obs = Observation(
                    grid=grid, timestep=self.timestep, action_space=self.obs.action_space,
                    steps_to_reconnect_line=self.steps_to_reconnect_line,
                    count_soft_overflow_steps=self.count_soft_overflow_steps, rho=None,
                    gen_status=self.gen_status, steps_to_recover_gen=self.steps_to_recover_gen,
                    steps_to_close_gen=self.steps_to_close_gen,
                    curstep_renewable_gen_p_max=None,
                    nextstep_renewable_gen_p_max=None,
                    rounded_gen_p=self._round_p(grid.prod_p[0]),
                    curstep_ld_p=curstep_ld_p,
                    nextstep_ld_p=nextstep_ld_p,
                    total_adjld=self.total_adjld,
                    total_stoenergy=self.total_stoenergy,
                    last_injection_gen_p=self.last_injection_gen_p,
                    disc_name=new_disc_name, recover_name=new_recover_name,
                )
                self.done = True
                self.obs.flag = grid.flag
                return self.return_res(info)

        rounded_gen_p = self._round_p(grid.prod_p[0])
        rounded_ld_p = self._round_p(grid.load_p[0])
        if hasattr(settings, 'ban_check_gen_status') and settings.ban_check_gen_status:
            pass
        else:
            # print('---------------------- print diff ------------------------------')
            # print('diff', np.sum(np.array(injection_gen_p)) - np.sum(np.array(rounded_gen_p)))
            # for p1, p2 in zip(injection_gen_p, rounded_gen_p):
            #     print('mini diff', p1 - p2)
            self._check_gen_status(injection_gen_p, rounded_gen_p)
        self.last_injection_gen_p = copy.deepcopy(injection_gen_p)

        # 更新下一时刻的预测值
        curstep_renewable_gen_p_max, nextstep_renewable_gen_p_max = \
            self.forecast_reader.read_step_renewable_gen_p_max(self.sample_idx)
        # 检查新能源预测值.csv文件中数据是否有错
        for i in range(len(nextstep_renewable_gen_p_max)):
            assert nextstep_renewable_gen_p_max[i] >= 0, i

        # 更新负荷预测值
        curstep_ld_p, nextstep_ld_p = self.forecast_reader.read_step_ld_p(self.sample_idx)
        # 检查储能设备初始值.csv文件中数据是否有错
        nextstep_sto_p = [nextstep_ld_p[i] for i in self.settings.stoenergy_ids]
        for i in range(len(nextstep_sto_p)):
            assert nextstep_sto_p[i] == 0, str(i) + str(nextstep_sto_p[i])

        # 更新负荷累加值
        self._update_total_load(adjust_adjld_p, adjust_stoenergy_p)

        action_space = self.action_space_cls.update(grid, self.steps_to_recover_gen, self.steps_to_close_gen,
                                                    rounded_gen_p, rounded_ld_p,
                                                    nextstep_renewable_gen_p_max, nextstep_ld_p,
                                                    self.total_adjld, self.total_stoenergy)

        rho = self._calc_rho(grid, settings)

        # pack obs
        self.obs = Observation(
            grid=grid, timestep=self.timestep, action_space=action_space,
            steps_to_reconnect_line=self.steps_to_reconnect_line,
            count_soft_overflow_steps=self.count_soft_overflow_steps, rho=rho,
            gen_status=self.gen_status, steps_to_recover_gen=self.steps_to_recover_gen,
            steps_to_close_gen=self.steps_to_close_gen,
            curstep_renewable_gen_p_max=curstep_renewable_gen_p_max,
            nextstep_renewable_gen_p_max=nextstep_renewable_gen_p_max,
            rounded_gen_p=rounded_gen_p,
            curstep_ld_p=curstep_ld_p,
            nextstep_ld_p=nextstep_ld_p,
            total_adjld=self.total_adjld,
            total_stoenergy=self.total_stoenergy,
            last_injection_gen_p=self.last_injection_gen_p,
            disc_name=new_disc_name, recover_name=new_recover_name,
        )

        self.reward = self.get_reward(self.obs, last_obs, act)
        return self.return_res(info=None)

    def _check_balance_bound(self, grid, settings):
        balanced_id = settings.balanced_id
        min_balanced_bound = settings.min_balanced_gen_bound
        max_balanced_bound = settings.max_balanced_gen_bound
        gen_p_min = settings.gen_p_min
        gen_p_max = settings.gen_p_max
        prod_p = grid.prod_p[0]
        val = prod_p[balanced_id]
        min_val = min_balanced_bound * gen_p_min[balanced_id]
        max_val = max_balanced_bound * gen_p_max[balanced_id]
        return val < min_val or val > max_val
        # return False

    # 计算支路的负载率
    def _calc_rho(self, grid, settings):
        limit = settings.ln_thermal_limit
        ln_num = settings.ln_num
        a_or = grid.a_or
        a_ex = grid.a_ex
        _rho = [None] * ln_num
        for i in range(ln_num):
            _rho[i] = max(a_or[0][i], a_ex[0][i]) / (limit[i] + 1e-3)
        return _rho

    def _injection_auto_mapping(self, injection_gen_p, adjust_gen_p):
        """
        based on the last injection q, map the value of injection_gen_p
        from (0, min_gen_p) to 0/min_gen_p
        """
        for i in self.settings.thermal_ids:
            if 0 < injection_gen_p[i] < self.settings.gen_p_min[i]:
                if self.last_injection_gen_p[i] == self.settings.gen_p_min[i]:
                    # print('!!!!!!!!!!!!!!!!!!!!!!! turn off thermal !!!!!!!!!!!!!!!!!!!!!!!')
                    injection_gen_p[i] = 0.0  # close the generator
                elif self.last_injection_gen_p[i] > self.settings.gen_p_min[i]:
                    injection_gen_p[i] = self.settings.gen_p_min[i]  # mapped to the min_gen_p
                elif self.last_injection_gen_p[i] == 0.0:
                    injection_gen_p[i] = self.settings.gen_p_min[i]  # open the generator
                    # print('!!!!!!!!!!!!!!!!!!!!!!! turn on thermal !!!!!!!!!!!!!!!!!!!!!!!')
                else:
                    assert False  # should never in (0, min_gen_p)

        return injection_gen_p

    def _update_gen_status(self, injection_gen_p):
        settings = self.settings
        for i in settings.thermal_ids:
            if injection_gen_p[i] == 0.0:
                if self.gen_status[i] == 1:  # the generator is open
                    if hasattr(settings, 'ban_check_steps_to_close_gen') and settings.ban_check_steps_to_close_gen:
                        pass
                    else:
                        assert self.steps_to_close_gen[i] == 0
                    self.gen_status[i] = 0  # close the generator
                    self.steps_to_recover_gen[i] = settings.max_steps_to_recover_gen
            elif injection_gen_p[i] == settings.gen_p_min[i]:
                if self.gen_status[i] == 0:  # the generator is shutdown
                    if hasattr(settings, 'ban_check_steps_to_close_gen') and settings.ban_check_steps_to_close_gen:
                        pass
                    else:
                        assert self.steps_to_recover_gen[i] == 0  # action isLegal function should have checked
                    self.gen_status[i] = 1  # open the generator
                    self.steps_to_close_gen[i] = settings.max_steps_to_close_gen

            if self.steps_to_recover_gen[i] > 0:
                self.steps_to_recover_gen[i] -= 1  # update recover timesteps counter
            if self.steps_to_close_gen[i] > 0:
                self.steps_to_close_gen[i] -= 1  # update close timesteps counter

    def _check_gen_status(self, injection_gen_p, rounded_gen_p):
        # check gen_p value of thermal generators after calling grid.env_feedback

        for i in self.settings.thermal_ids:
            if self.gen_status[i] == 0:
                assert rounded_gen_p[i] == 0.0
            else:
                assert rounded_gen_p[i] >= self.settings.gen_p_min[i], (i, rounded_gen_p[i], self.settings.gen_p_min[i])

            assert abs(injection_gen_p[i] - rounded_gen_p[i]) <= self.settings.env_allow_precision, (
                i, injection_gen_p[i], rounded_gen_p[i])

        for i in self.settings.renewable_ids:
            assert abs(injection_gen_p[i] - rounded_gen_p[i]) <= self.settings.env_allow_precision, (
                i, injection_gen_p[i], rounded_gen_p[i])

    def _check_action(self, act):
        assert 'adjust_gen_p' in act
        assert 'adjust_gen_v' in act
        assert 'adjust_adjld_p' in act
        assert 'adjust_stoenergy_p' in act

        adjust_gen_p = act['adjust_gen_p']
        adjust_gen_v = act['adjust_gen_v']
        adjust_space_adjld = act['adjust_adjld_p']
        adjust_space_stoenergy = act['adjust_stoenergy_p']

        assert isinstance(adjust_gen_p, (list, tuple, np.ndarray))
        assert len(adjust_gen_p) == self.settings.gen_num

        assert isinstance(adjust_gen_v, (list, tuple, np.ndarray))
        assert len(adjust_gen_v) == self.settings.gen_num

        assert isinstance(adjust_space_adjld, (list, tuple, np.ndarray))
        assert len(adjust_space_adjld) == self.settings.adjld_num

        assert isinstance(adjust_space_stoenergy, (list, tuple, np.ndarray))
        assert len(adjust_space_stoenergy) == self.settings.stoenergy_num

    def _round_p(self, p):
        dig = self.settings.keep_decimal_digits
        # return [(round(x * 10 ** dig)) / (10 ** dig) for x in p]
        return [round(x, dig) for x in p]
        # return [(int(x*10**dig))/(10**dig) for x in p]

    def _update_total_load(self, adjust_adjld_p, adjust_stoenergy_p):
        self.total_adjld = self.total_adjld + adjust_adjld_p
        for i in range(self.settings.stoenergy_num):
            if adjust_stoenergy_p[i] > 0:
                self.total_stoenergy[i] += self.settings.chargerate_rho * adjust_stoenergy_p[i]
            else:
                self.total_stoenergy[i] += adjust_stoenergy_p[i]

    def get_reward(self, obs, last_obs, act):
        reward_dict = {
            "EPRIReward": EPRIReward,
            "line_over_flow_reward": line_over_flow_reward,
            "renewable_consumption_reward": renewable_consumption_reward,
            "running_cost_reward": running_cost_reward,
            "balanced_gen_reward": balanced_gen_reward,
            "gen_reactive_power_reward": gen_reactive_power_reward,
            "sub_voltage_reward": sub_voltage_reward,
            "adjld_reward": adjld_reward,
            "stoenergy_reward": stoenergy_reward
        }
        reward_func = reward_dict[self.reward_type]
        return reward_func(obs, last_obs, act, self.settings)

    def check_done(self, grid, settings):
        if grid.flag == 1:
            return True, 'grid is not converged'
        if self.sample_idx >= settings.sample_num:
            return True, 'sample idx reach the limit'
        if self._check_balance_bound(grid, settings):
            return True, 'balance gen out of bound'
        return False, None

    def return_res(self, info=None):
        ret_obs = copy.deepcopy(self.obs)
        if self.done:
            if not info:
                return ret_obs, 0, True, {}
            else:
                return ret_obs, 0, True, {'fail_info': info}
        else:
            assert self.reward, "the reward are not calculated yet"
            info = copy.deepcopy(self.info)
            self.info = None
            return ret_obs, self.reward, False, {'fail_info': info}
