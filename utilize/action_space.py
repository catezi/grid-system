import numpy as np
from gym import spaces
import copy

class ActionSpace(object):
    def __init__(self, settings):
        self.settings = copy.deepcopy(settings)
        self.gen_num = settings.gen_num
        self.adjld_num = settings.adjld_num
        self.adjld_ids = settings.adjld_ids
        self.stoenergy_ids = settings.stoenergy_ids
        self.stoenergy_num = settings.stoenergy_num
        self.ramp_rate = settings.ramp_rate
        self.gen_p_min = settings.gen_p_min
        self.gen_p_max = settings.gen_p_max
        self.gen_v_max = settings.gen_v_max
        self.gen_v_min = settings.gen_v_min
        self.thermal_ids = settings.thermal_ids
        self.renewable_ids = settings.renewable_ids
        self.balanced_id = settings.balanced_id
        self.keep_decimal_digits = settings.keep_decimal_digits
        self.adjld_capacity = settings.adjld_capacity
        self.adjld_uprate = settings.adjld_uprate
        self.adjld_dnrate = settings.adjld_dnrate
        self.stoenergy_capacity = settings.stoenergy_capacity
        self.stoenergy_chargerate_max = settings.stoenergy_chargerate_max
        self.stoenergy_dischargerate_max = settings.stoenergy_dischargerate_max
        self.chargerate_rho = settings.chargerate_rho

    # 获得机组有功出力调整值的上/下限
    def get_p_range(self, gen_p, steps_to_recover_gen, steps_to_close_gen, nextstep_renewable_gen_p_max):
        # Initialization
        low = np.zeros([self.gen_num])
        high = np.zeros([self.gen_num])
        low[self.balanced_id] = -float('inf')
        high[self.balanced_id] = float('inf')
        self.update_thermal_p(low, high, gen_p, steps_to_recover_gen, steps_to_close_gen)
        self.update_renewable_p(low, high, gen_p, nextstep_renewable_gen_p_max)
        self.update_balance_p(low, high, gen_p)
        low = np.round(low, self.keep_decimal_digits)
        high = np.round(high, self.keep_decimal_digits)
        return low, high

    def update_thermal_p(self, low, high, gen_p, steps_to_recover_gen, steps_to_close_gen):
        # injection values are less than maximum limit and larger than minimum limit
        max_capa_adjust = [self.gen_p_max[i] - gen_p[i] for i in range(self.gen_num)]
        min_capa_adjust = [self.gen_p_min[i] - gen_p[i] for i in range(self.gen_num)]
        # adjust actions of thermal generators should less than ramp value
        max_ramp_adjust = [self.ramp_rate * ele for ele in self.gen_p_max]

        # default value of min&max adjust is 0
        for idx in self.thermal_ids:
            if gen_p[idx] == 0.0:
                low[idx] = 0.0
                high[idx] = self.gen_p_min[idx]
                if steps_to_recover_gen[idx] != 0:  # cannot turn on
                    high[idx] = 0.0

            elif gen_p[idx] == self.gen_p_min[idx]:
                high[idx] = min(max_capa_adjust[idx], max_ramp_adjust[idx])
                if steps_to_close_gen[idx] == 0:  # can turn off
                    low[idx] = -self.gen_p_min[idx]
                else:  # cannot turn off
                    low[idx] = 0.0

            elif gen_p[idx] > self.gen_p_min[idx]:
                low[idx] = max(min_capa_adjust[idx], -max_ramp_adjust[idx])
                high[idx] = min(max_capa_adjust[idx], max_ramp_adjust[idx])
                if steps_to_close_gen[idx] == 0:  # can turn off
                    low[idx] = max(-gen_p[idx], -max_ramp_adjust[idx])
            # else:
            #     assert False
            # 这里潮流计算黑箱可能存在bug, 导致注入值和实际计值不一致, 可能导致功率介于0-min
            # 因此这里采用柔性策略
            else:
                low[idx] = 0.0
                high[idx] = self.gen_p_min[idx]
                if steps_to_recover_gen[idx] != 0:  # cannot turn on
                    high[idx] = 0.0

    def update_renewable_p(self, low, high, gen_p, nextstep_renewable_gen_p_max):
        for i, idx in enumerate(self.renewable_ids):
            low[idx] = -gen_p[idx]
            high[idx] = nextstep_renewable_gen_p_max[i] - gen_p[idx]

    def update_balance_p(self, low, high, gen_p):
        balanced_id = self.balanced_id
        # low[balanced_id] = self.gen_p_min[balanced_id] * self.settings.min_balanced_gen_bound - gen_p[balanced_id]
        # high[balanced_id] = self.gen_p_max[balanced_id] * self.settings.max_balanced_gen_bound - gen_p[balanced_id]
        low[balanced_id] = 0.0
        high[balanced_id] = 0.0


    # 获得机组电压调整值的上/下限
    def get_v_range(self, gen_v):
        low = np.zeros([self.gen_num])
        high = np.zeros([self.gen_num])
        for i in range(self.gen_num):
            low[i] = self.gen_v_min[i] - gen_v[i]
            high[i] = self.gen_v_max[i] - gen_v[i]
        low = np.round(low, self.keep_decimal_digits)
        high = np.round(high, self.keep_decimal_digits)
        return low, high

    # 获得可调负荷有功调整值上/下限
    def get_adjld_range(self, total_adjld):
        # total_adjld: 各可调负荷的负荷调整值累加（近 288 时间步内）
        low = np.zeros([self.adjld_num])
        high = np.zeros([self.adjld_num])

        # capa_adjust = [self.adjld_capacity[i] - total_adjld[i] for i in range(self.adjld_num)]
        # for i in range(self.adjld_num):
        #     low[i] = max(-self.adjld_dnrate[i], -capa_adjust[i])
        #     high[i] = min(self.adjld_uprate[i], capa_adjust[i])
        max_capa_adjust = [self.adjld_capacity[i] - total_adjld[i] for i in range(self.adjld_num)]
        min_capa_adjust = [-self.adjld_capacity[i] - total_adjld[i] for i in range(self.adjld_num)]
        for i in range(self.adjld_num):
            low[i] = max(-self.adjld_dnrate[i], min_capa_adjust[i])
            high[i] = min(self.adjld_uprate[i], max_capa_adjust[i])
        low = np.round(low, self.keep_decimal_digits)
        high = np.round(high, self.keep_decimal_digits)
        return low, high

    # 获得储能有功调整值
    def get_stoenergy_range(self, total_stoenergy):
        # 各储能设备剩余电量（各储能设备 的充放电实际值累加）
        low = np.zeros([self.stoenergy_num])
        high = np.zeros([self.stoenergy_num])

        min_capa_adjust = [-total_stoenergy[i] for i in range(self.stoenergy_num)]
        max_capa_adjust = [(self.stoenergy_capacity[i]-total_stoenergy[i])/self.chargerate_rho
                           for i in range(self.stoenergy_num)]

        for i in range(self.stoenergy_num):
            low[i] = max(min_capa_adjust[i], -self.stoenergy_dischargerate_max[i])
            high[i] = min(max_capa_adjust[i], self.stoenergy_chargerate_max[i])
        low = np.round(low, self.keep_decimal_digits)
        high = np.round(high, self.keep_decimal_digits)
        return low, high

    # 根据当前状态更新动作空间
    def update(self, grid, steps_to_recover_gen, steps_to_close_gen, rounded_gen_p, rounded_ld_p,
               nextstep_renewable_gen_p_max, nextstep_ld_p, total_adjld, total_stoenergy):

        # 机组有功gen_p
        gen_p = rounded_gen_p
        low_adjust_p, high_adjust_p = self.get_p_range(
            gen_p,
            steps_to_recover_gen,
            steps_to_close_gen,
            nextstep_renewable_gen_p_max
        )
        action_space_p = spaces.Box(low=low_adjust_p, high=high_adjust_p)

        # 机组电压
        gen_v = grid.prod_v[0]
        low_adjust_v, high_adjust_v = self.get_v_range(gen_v)
        action_space_v = spaces.Box(low=low_adjust_v, high=high_adjust_v)

        # 可调负荷
        # adjld_p = [rounded_ld_p[i] for i in self.adjld_ids]
        # nextstep_adjld_p = [nextstep_ld_p[i] for i in self.adjld_ids]
        low_adjust_adjld, high_adjust_adjld = self.get_adjld_range(total_adjld)
        action_space_adjld = spaces.Box(low=low_adjust_adjld, high=high_adjust_adjld)

        # 储能设备
        # stoenergy_p = [rounded_ld_p[i] for i in self.stoenergy_ids]
        # nextstep_sto_p = [nextstep_ld_p[i] for i in self.stoenergy_ids]
        low_adjust_sto, high_adjust_sto = self.get_stoenergy_range(total_stoenergy)
        action_space_stoenergy = spaces.Box(low=low_adjust_sto, high=high_adjust_sto)

        action_space = {
            'adjust_gen_p': action_space_p,
            'adjust_gen_v': action_space_v,
            'adjust_adjld_p': action_space_adjld,
            'adjust_stoenergy_p': action_space_stoenergy
        }
        return action_space
