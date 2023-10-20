import copy
from utilize.settings import settings

class Observation:
    def __init__(self, grid, timestep, action_space, steps_to_reconnect_line, count_soft_overflow_steps,
                 rho, gen_status, steps_to_recover_gen, steps_to_close_gen, curstep_renewable_gen_p_max,
                 nextstep_renewable_gen_p_max, rounded_gen_p, curstep_ld_p, nextstep_ld_p,
                 total_adjld, total_stoenergy, last_injection_gen_p, disc_name, recover_name):
        self.timestep = timestep
        self.vTime = grid.vTime
        self.gen_p = rounded_gen_p
        self.gen_q = grid.prod_q[0]
        self.gen_v = grid.prod_v[0]
        self.target_dispatch = grid.target_dispatch[0]
        self.actual_dispatch = grid.actual_dispatch[0]
        self.ld_p = grid.load_p[0]
        self.adjld_p = [grid.load_p[0][i] for i in settings.adjld_ids]
        self.stoenergy_p = [grid.load_p[0][i] for i in settings.stoenergy_ids]
        self.ld_q = grid.load_q[0]
        self.ld_v = grid.load_v[0]
        self.p_or = grid.p_or[0]
        self.q_or = grid.q_or[0]
        self.v_or = grid.v_or[0]
        self.a_or = grid.a_or[0]
        self.p_ex = grid.p_ex[0]
        self.q_ex = grid.q_ex[0]
        self.v_ex = grid.v_ex[0]
        self.a_ex = grid.a_ex[0]
        self.line_status = grid.line_status[0]
        self.grid_loss = grid.grid_loss
        self.bus_v = grid.bus_v
        # 拓扑结构信息
        self.bus_gen = grid.bus_gen
        self.bus_load = grid.bus_load
        self.bus_branch = grid.bus_branch
        # 潮流计算是否收敛
        self.flag = grid.flag
        self.unnameindex = grid.un_nameindex
        self.action_space = action_space  # 合法动作空间
        self.steps_to_reconnect_line = steps_to_reconnect_line  # 线路断开后恢复连接的剩余时间步数
        self.count_soft_overflow_steps = count_soft_overflow_steps  # 线路软过载的已持续时间步数
        self.rho = rho
        self.gen_status = gen_status  # 机组开关机状态（1为开机，0位关机）
        self.steps_to_recover_gen = steps_to_recover_gen  # 机组关机后可以重新开机的时间步（如果机组状态为开机，则值为0）
        self.steps_to_close_gen = steps_to_close_gen  # 机组开机后可以重新关机的时间步（如果机组状态为关机，则值为0）
        self.curstep_renewable_gen_p_max = curstep_renewable_gen_p_max  # 当前时间步新能源机组的最大有功出力
        self.nextstep_renewable_gen_p_max = nextstep_renewable_gen_p_max  # 下一时间步新能源机组的最大有功出力
        self.curstep_ld_p = curstep_ld_p  # 当前时间步的负荷预测值
        self.nextstep_ld_p = nextstep_ld_p  # 下一时间步的负荷预测值
        self.total_adjld = total_adjld  # 各台可调负荷的负荷调整值累加
        self.total_stoenergy = total_stoenergy  # 各台储能设备的充放电实际值累加
        ###### 魔改部分
        self.last_injection_gen_p = last_injection_gen_p
        self.disc_name = disc_name
        self.recover_name = recover_name
