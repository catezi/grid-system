from utilize.settings import settings

# Base exception class
class ActionIllegalException(Exception):
    def __init__(self, error_info):
        self.error_info = error_info

    def __str__(self):
        return self.error_info

    def __repr__(self):
        return "`{}: {}`".format(type(self).__name__, self.error_info)


class GenPOutOfActionSpace(ActionIllegalException):
    def __init__(self, illegal_gen_ids, action_space_gen_p, action_gen_p):
        
        error_info = 'Adjustment of gen_p is out of action space: \n{}'.format(
                "\n".join(["gen_id: {}, action_space.low: {}, action_space.high: {}, action: {}".format(
                    gen_id, action_space_gen_p.low[gen_id], action_space_gen_p.high[gen_id], action_gen_p[gen_id])
                    for gen_id in illegal_gen_ids])
                )
        super(GenPOutOfActionSpace, self).__init__(error_info)

class GenVOutOfActionSpace(ActionIllegalException):
    def __init__(self, illegal_gen_ids, action_space_gen_v, action_gen_v):
        
        error_info = 'Adjustment of gen_v is out of action space: \n{}'.format(
                "\n".join(["gen_id: {}, action_space.low: {}, action_space.high: {}, action: {}".format(
                    gen_id, action_space_gen_v.low[gen_id], action_space_gen_v.high[gen_id], action_gen_v[gen_id])
                    for gen_id in illegal_gen_ids])
                )
        super(GenVOutOfActionSpace, self).__init__(error_info)


class AdjldPOutOfActionSpace(ActionIllegalException):
    def __init__(self, illegal_adjld_ids, action_space_adjld_p, action_adjld_p):
        error_info = 'Adjustment of adjld_p is out of action space: \n{}'.format(
            "\n".join(["adjld_id: {}, action_space.low: {}, action_space.high: {}, action: {}".format(
                adjld_id, action_space_adjld_p.low[adjld_id], action_space_adjld_p.high[adjld_id], action_adjld_p[adjld_id])
                for adjld_id in illegal_adjld_ids])
        )
        super(AdjldPOutOfActionSpace, self).__init__(error_info)


class StoenergyPOutOfActionSpace(ActionIllegalException):
    def __init__(self, illegal_stoenergy_ids, action_space_stoenergy_p, action_stoenergy_p):
        error_info = 'Adjustment of stoenergy_p is out of action space: \n{}'.format(
            "\n".join(["stoenergy_id: {}, action_space.low: {}, action_space.high: {}, action: {}".format(
                stoenergy_id, action_space_stoenergy_p.low[stoenergy_id],
                action_space_stoenergy_p.high[stoenergy_id], action_stoenergy_p[stoenergy_id])
                for stoenergy_id in illegal_stoenergy_ids])
        )
        super(StoenergyPOutOfActionSpace, self).__init__(error_info)