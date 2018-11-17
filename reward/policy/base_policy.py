from abc import ABC, abstractmethod


class BasePolicy:
    def __init__(self, nn):
        self.nn = nn

    @abstractmethod
    def create_dist(self, s):
        pass

    @abstractmethod
    def get_ac(self, s, step):
        pass
