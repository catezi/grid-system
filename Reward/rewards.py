import math


def line_over_flow_reward(obs, settings):
    r = 1 - sum([min(i, 1) for i in obs.rho]) / settings.ln_num
    return r


def renewable_consumption_reward(obs, settings):
    all_gen_p = 0.0
    all_gen_p_max = 0.0
    for i, j in enumerate(settings.renewable_ids):
        all_gen_p += obs.gen_p[j]
        all_gen_p_max += obs.curstep_renewable_gen_p_max[i]
    r = all_gen_p / all_gen_p_max
    return r


def balanced_gen_reward(obs, settings):
    r = 0.0
    idx = settings.balanced_id
    max_val = settings.gen_p_max[idx]
    min_val = settings.gen_p_min[idx]
    gen_p_val = obs.gen_p[idx]
    if gen_p_val > max_val:
        r += abs((gen_p_val - max_val) / max_val)
    if gen_p_val < min_val:
        r += abs((gen_p_val - min_val) / min_val)
    r = -10 * r  # Ensure the range of r is [-1,0]
    return r


def running_cost_reward(obs, last_obs, settings):

    # case 2：
    # r = 0.0
    # for i, name in enumerate(settings.gen_name_list):
    #     idx = obs.unnameindex[name]
    #     a = settings.gen_p_a
    #     b = settings.gen_p_b
    #     c = settings.gen_p_c
    #     d = settings.gen_p_d
    #     gen_p_max = settings.gen_p_max
    #
    #     ri = a[i] * (obs.gen_p[idx]) ** 2 + b[i] * obs.gen_p[idx] + c[i]
    #     if obs.gen_status[idx] != last_obs.gen_status[idx] and idx in settings.thermal_ids:
    #         ri += d[i]
    #     ri /= a[i] * (gen_p_max[idx]) ** 2 + b[i] * gen_p_max[idx] + c[i] + d[i]
    #     r += ri
    #
    # r = -r / settings.gen_num


    r = 0.0
    for i, name in enumerate(settings.gen_name_list):
        idx = obs.unnameindex[name]
        a = settings.gen_p_a
        b = settings.gen_p_b
        c = settings.gen_p_c
        d = settings.gen_p_d
        r -= a[i] * (obs.gen_p[idx]) ** 2 + b[i] * obs.gen_p[idx] + c[i]
        if obs.gen_status[idx] != last_obs.gen_status[idx] and idx in settings.thermal_ids:
            r -= d[i]
    r = r/50000
    r = math.exp(r) - 1
    return r



def gen_reactive_power_reward(obs, settings):
    r = 0.0
    for i in range(settings.gen_num):
        if obs.gen_q[i] > settings.gen_q_max[i]:
            r -= abs((obs.gen_q[i] - settings.gen_q_max[i]) / settings.gen_q_max[i])
        if obs.gen_q[i] < settings.gen_q_min[i]:
            r -= abs((obs.gen_q[i] - settings.gen_q_min[i]) / settings.gen_q_min[i])
    r = math.exp(r) - 1
    return r


def sub_voltage_reward(obs, settings):
    r = 0.0
    for i in range(len(settings.bus_v_max)):
        if obs.bus_v[i] > settings.bus_v_max[i]:
            r -= abs((obs.bus_v[i] - settings.bus_v_max[i]) / settings.bus_v_max[i])
        if obs.bus_v[i] < settings.bus_v_min[i]:
            r -= abs((obs.bus_v[i] - settings.bus_v_min[i]) / settings.bus_v_min[i])
    r = math.exp(r) - 1
    return r


def adjld_reward(act, settings):
    r = 0.0
    a = settings.adjld_a
    b = settings.adjld_b
    adjust_adjld_p = act['adjust_adjld_p']
    adjld_uprate = settings.adjld_uprate
    adjld_dnrate = settings.adjld_dnrate
    for i in range(settings.adjld_num):
        ri = a[i] * (adjust_adjld_p[i]) ** 2 + b[i] * abs(adjust_adjld_p[i])

        if adjust_adjld_p[i] > 0:
            ri /= a[i] * (adjld_uprate[i]) ** 2 + b[i] * adjld_uprate[i]
        else:
            ri /= a[i] * (adjld_dnrate[i]) ** 2 + b[i] * adjld_dnrate[i]

        r += ri
    r = -r / settings.adjld_num
    return r


def stoenergy_reward(act, settings):
    r = 0.0
    k = settings.stoenergy_k
    b = settings.stoenergy_b
    adjust_adjld_p = act['adjust_stoenergy_p']
    for i in range(settings.stoenergy_num):
        r -= k[i] * abs(adjust_adjld_p[i]) + b[i]
    r = math.exp(r) - 1
    return r


# 电网运行费用
def grid_operation_reward(obs, act, settings):
    a = settings.grid_operation_a
    b = settings.grid_operation_b
    c = settings.grid_operation_c
    sum_gen_p = 0
    for i, name in enumerate(settings.gen_name_list):
        idx = obs.unnameindex[name]
        sum_gen_p += obs.gen_p[idx]

    sum_adjust_adjld_p = sum(abs(i) for i in act['adjust_adjld_p'])
    sum_adjust_stoenergy_p = sum(abs(i) for i in act['adjust_stoenergy_p'])
    sum_ld_p = sum(obs.ld_p)
    r = -(a * sum_gen_p + b * sum_adjust_adjld_p + c * sum_adjust_stoenergy_p)/sum_ld_p
    return r


def EPRIReward(obs, last_obs, act, settings):
    r1 = line_over_flow_reward(obs, settings)
    r2 = renewable_consumption_reward(obs, settings)
    r3 = running_cost_reward(obs, last_obs, settings)
    r4 = balanced_gen_reward(obs, settings)
    r5 = gen_reactive_power_reward(obs, settings)
    r6 = sub_voltage_reward(obs, settings)
    r7 = adjld_reward(act, settings)
    r8 = grid_operation_reward(obs, act, settings)
    # r = settings.coeff_line_over_flow * r1 + \
    #     settings.coeff_renewable_consumption * r2 + \
    #     settings.coeff_running_cost * r3 + \
    #     settings.coeff_balanced_gen * r4 + \
    #     settings.coeff_gen_reactive_power * r5 + \
    #     settings.coeff_sub_voltage * r6 + \
    #     settings.coeff_adjld * r7

    r = settings.coeff_line_over_flow * r1 + \
        settings.coeff_renewable_consumption * r2 + \
        settings.coeff_balanced_gen * r4 + \
        settings.coeff_gen_reactive_power * r5 + \
        settings.coeff_sub_voltage * r6 + \
        settings.coeff_grid_operation * r8

    return r


def fun(x):
    return str(round(x, 2))
