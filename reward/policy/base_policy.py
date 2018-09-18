from abc import ABC, abstractmethod


class BasePolicy:
    def __init__(self, nn):
        self.nn = nn

    @abstractmethod
    def create_dist(self, state):
        pass

    @abstractmethod
    def get_action(self, state, step):
        pass
