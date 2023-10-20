import numpy as np

from Agent.BaseAgent import BaseAgent
from utilize.form_action import *

class DoNothingAgent(BaseAgent):

    def __init__(self, settings):
        BaseAgent.__init__(self, settings)
        self.settings = settings
        self.action = form_action(np.zeros(self.settings.gen_num),
                                  np.zeros(self.settings.gen_num),
                                  np.zeros(self.settings.adjld_num),
                                  np.zeros(self.settings.stoenergy_num))

    def act(self, obs, reward, done=False):
        return self.action

