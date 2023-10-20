import copy
from abc import abstractmethod

class BaseAgent():
    def __init__(self, settings):
        self.settings = settings

    def reset(self, ons):
        pass

    @abstractmethod
    def act(self, obs, reward, done=False):
        pass
