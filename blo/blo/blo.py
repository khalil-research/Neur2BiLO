from abc import ABC, abstractmethod

class BLO(ABC):
    """ Class for bilevel optimization problem. """

    @abstractmethod
    def sample_instance(self, cfg):
        """ Samples instance. """
        pass

    @abstractmethod
    def solve_follower(self, opt_model, x):
        """ Solve follower problem. """
        pass
