import copy
import numpy as np

class Disconnect(object):
    def __init__(self, np_random, settings):
        self.np_random = np_random
        self.ln_name = settings.ln_name

        self.prob_dis = settings.prob_disconnection
        self.white_list = settings.white_list_random_disconnection
        self.hard_bound = settings.hard_overflow_bound
        self.soft_bound = settings.soft_overflow_bound
        self.ln_num = settings.ln_num
        self.max_steps_to_reconnect_line = settings.max_steps_to_reconnect_line
        self.max_steps_soft_overflow = settings.max_steps_soft_overflow

    # Randomly cut one line from white list
    def random_cut(self):
        if self.np_random.rand() < self.prob_dis:
            dis_line_name = self.np_random.choice(self.white_list)
            return [self.ln_name.index(dis_line_name)]
        return []

    # Find lines meetings soft & hard overflow
    def overflow(self, rho):
        hard_overflow_ids = np.where(rho > self.hard_bound)[0] 
        soft_overflow_ids = np.intersect1d(np.where(rho > self.soft_bound),
                                       np.where(rho <= self.hard_bound))
        return hard_overflow_ids, soft_overflow_ids

    # Count soft overflow steps
    def count_soft_steps(self, soft_overflow_ids, count_soft_overflow_steps):
        dis_ids = []
        for i in range(self.ln_num):
            if i in soft_overflow_ids:
                count_soft_overflow_steps[i] += 1 
                if count_soft_overflow_steps[i] >= self.max_steps_soft_overflow:
                    dis_ids.append(i)
            else:
                count_soft_overflow_steps[i] = 0
        return dis_ids, count_soft_overflow_steps

    # If the line is to be & can be cut: cut
    # If the line has been cut before: step -= 1
    def update_reconnect_steps(self, cut_line_ids, steps_to_reconnect_line):
        for i in range(self.ln_num):
            if i in cut_line_ids:
                if steps_to_reconnect_line[i] == 0:
                    steps_to_reconnect_line[i] = self.max_steps_to_reconnect_line
            else:
                if steps_to_reconnect_line[i] <= 0:
                    steps_to_reconnect_line[i] = 0
                else:
                    steps_to_reconnect_line[i] -= 1
        return steps_to_reconnect_line

    def get_disc_name(self, last_obs):
        steps_to_reconnect_line = last_obs.steps_to_reconnect_line
        last_steps_to_reconnect_line = copy.deepcopy(last_obs.steps_to_reconnect_line)
        count_soft_overflow_steps = last_obs.count_soft_overflow_steps
        rho = np.array(last_obs.rho)
    
        dis_line_id = self.random_cut()
        hard_overflow_ids, soft_overflow_ids = self.overflow(rho)
        dis_softoverflow_ids, count_soft_overflow_steps =\
         self.count_soft_steps(soft_overflow_ids, count_soft_overflow_steps)
        
        cut_line_ids = dis_line_id + dis_softoverflow_ids + hard_overflow_ids.tolist()
        cut_line_ids = list(set(cut_line_ids))

        steps_to_reconnect_line = self.update_reconnect_steps(cut_line_ids, steps_to_reconnect_line)
        final_cut_line_ids = np.where(steps_to_reconnect_line > 0)
        disc_name = [self.ln_name[idx] for idx in final_cut_line_ids[0]]
        new_recover_name, new_disc_name = self.get_recover_line_name(steps_to_reconnect_line, last_steps_to_reconnect_line)

        return disc_name, steps_to_reconnect_line, count_soft_overflow_steps, new_recover_name, new_disc_name

    def get_recover_line_name(self, steps_to_reconnect_line, last_steps_to_reconnect_line):
        recover_ids = []
        disconnect_ids = []
        for t1, t2, idx in zip(steps_to_reconnect_line, last_steps_to_reconnect_line, range(len(steps_to_reconnect_line))):
            if t2 > 0 and t1 == 0:
                recover_ids.append(idx)
            elif t2 == 0 and t1 > 0:
                disconnect_ids.append(idx)
        recover_name = [self.ln_name[idx] for idx in recover_ids]
        disconnect_name = [self.ln_name[idx] for idx in disconnect_ids]

        return recover_name, disconnect_name
