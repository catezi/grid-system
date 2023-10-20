import numpy as np

from utilize.exceptions.action_illegal_exceptions import *


def check_gen_p(adjust_gen_p, action_space_gen_p, gen_ids, eps):
    illegal_gen_ids = [i for i in gen_ids if adjust_gen_p[i] < action_space_gen_p.low[i] - eps or adjust_gen_p[i] > action_space_gen_p.high[i] + eps]
    return illegal_gen_ids


def check_gen_v(adjust_gen_v, action_space_gen_v, gen_ids, eps):
    illegal_gen_ids = [i for i in gen_ids if adjust_gen_v[i] < action_space_gen_v.low[i] - eps or adjust_gen_v[i] > action_space_gen_v.high[i] + eps]
    return illegal_gen_ids


def check_adjld_p(adjust_adjld_p, action_space_adjld_p, adjld_ids, eps):
    illegal_adjld_ids = [i for i in adjld_ids if adjust_adjld_p[i] < action_space_adjld_p.low[i] - eps or adjust_adjld_p[i] > action_space_adjld_p.high[i] + eps]
    return illegal_adjld_ids


def check_stoenergy_p(adjust_stoenergy_p, action_space_stoenergy_p, stoenergy_ids, eps):
    illegal_stoenergy_ids = [i for i in stoenergy_ids if adjust_stoenergy_p[i] < action_space_stoenergy_p.low[i] - eps or adjust_stoenergy_p[i] > action_space_stoenergy_p.high[i] + eps]
    return illegal_stoenergy_ids


def is_legal(act, last_obs, settings):
    """
    Returns:
        illegal_reasons(list): reasons why the action is illegal 
    """
    illegal_reasons = []
    eps = settings.action_allow_precision

    gen_ids = list(range(settings.gen_num))
    adjld_ids = list(range(settings.adjld_num))
    stoenergy_ids = list(range(settings.stoenergy_num))

    action_space_gen_p = last_obs.action_space['adjust_gen_p']
    action_space_gen_v = last_obs.action_space['adjust_gen_v']
    action_space_adjld_p = last_obs.action_space['adjust_adjld_p']
    action_space_stoenergy_p = last_obs.action_space['adjust_stoenergy_p']

    adjust_gen_p = act['adjust_gen_p']
    adjust_gen_v = act['adjust_gen_v']
    adjust_adjld_p = act['adjust_adjld_p']
    adjust_stoenergy_p = act['adjust_stoenergy_p']

    gen_p_illegal_ids = check_gen_p(adjust_gen_p, action_space_gen_p, gen_ids, eps)
    gen_v_illegal_ids = check_gen_v(adjust_gen_v, action_space_gen_v, gen_ids, eps)
    adjld_p_illegal_ids = check_adjld_p(adjust_adjld_p, action_space_adjld_p, adjld_ids, eps)
    stoenergy_p_illegal_ids = check_stoenergy_p(adjust_stoenergy_p, action_space_stoenergy_p, stoenergy_ids, eps)

    if gen_p_illegal_ids:
        illegal_reasons.append(GenPOutOfActionSpace(gen_p_illegal_ids, action_space_gen_p, adjust_gen_p))
    if gen_v_illegal_ids:
        illegal_reasons.append(GenVOutOfActionSpace(gen_v_illegal_ids, action_space_gen_v, adjust_gen_v))
    if adjld_p_illegal_ids:
        illegal_reasons.append(AdjldPOutOfActionSpace(adjld_p_illegal_ids, action_space_adjld_p, adjust_adjld_p))
    if stoenergy_p_illegal_ids:
        illegal_reasons.append(StoenergyPOutOfActionSpace(stoenergy_p_illegal_ids, action_space_stoenergy_p, adjust_stoenergy_p))

    return illegal_reasons == [], illegal_reasons
