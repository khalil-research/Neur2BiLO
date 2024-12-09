import os
import pickle as pkl
import time
from multiprocessing import Manager, Pool

import gurobipy as gp
import numpy as np

from blo.blo.cng import CriticalNodeGame
from blo.utils.cng import get_path
from .data_manager import DataManager


class CriticalNodeGameDataManager(DataManager):


    def __init__(self, cfg):
        """ Constructor for knapsack problem.  """
        self.cfg = cfg

        self.problem_path = get_path(self.cfg.data_path, self.cfg, "problem")
        self.ml_data_path = get_path(self.cfg.data_path, self.cfg, "ml_data")

        self.blo = CriticalNodeGame()


    def initialize_problem(self):
        """ Initializes a knapsack problem from Tang et al. 2016.  """
        print("Generating problem...")

        self.prob = self._get_problem_data(self.cfg)

        # save problem 
        print("Saving problem to:", self.problem_path)
        
        pkl.dump(self.prob, open(self.problem_path, 'wb'))


    def _solve_lower_level_mp(self, x, instance, inst_id, mp_time, mp_count, n_samples):
        """ Obtains the cost of the suboptimal first stage solution.  """
        time_ = time.time()
    
        # create pyomo bilevel model
        m = self.blo.create_cng_model(instance)

        # solve follower for fixed x
        solving_res = self.blo.solve_follower(m, x)

        time_ = time.time() - time_

        # store results
        results = {
            'x' : x,
            'instance' : instance,
            'inst_id' : inst_id,
            'follower_obj' : solving_res['follower_obj'],
            'follower_sol' : solving_res['follower_sol'],
            'leader_obj' : solving_res['leader_obj'],
            'leader_sol' : solving_res['leader_sol'],
        }

        self.update_mp_status(mp_count, mp_time, n_samples)

        return results


    def _sample_random_x(self, instance, X_hash=None):
        """ Samples random x decision. """

        # this may be needed for small |V| to avoid infinite looping
        max_attempts = 5 # number of attempts before adding sample if in X_hash
        attempts = 1     # current number of attempts

        # sample until a unique point that fits the budget is found
        while True:

            # sample a probability of 0/1's
            max_prob = min([1, instance['d_ratio'] + 0.3]) # sample up to 30% more than the budget
            p = np.random.uniform(0, max_prob)

            # sample x based on probablility
            x = np.random.choice([0,1], p=(1-p, p), size=instance['v'])
            
            # check if point is feasible
            if np.dot(x, instance['d_budget_coefs']) <= instance['D']:

                # also check that x is not already a decision
                x_str = np.array2string(x)
                if x_str not in X_hash or attempts >= max_attempts:
                    break

                attempts += 1

        X_hash.add(x_str)

        return x


    def _get_problem_data(self, cfg):
        """ Stores generic problem information. """
        prob = {}

        # values for CNG instances
        prob["prob_type"] = cfg.prob_type
        prob["v"] = cfg.v
        prob["gamma"] = cfg.gamma
        prob["eta"] = cfg.eta
        prob["epsilon_ratio"] = cfg.epsilon_ratio
        prob["delta_ratio"] = cfg.delta_ratio
        prob["d_ratio"] = cfg.d_ratio
        prob["a_ratio"] = cfg.a_ratio
        prob["budget_range"] = cfg.budget_range
        prob["profit_range"] = cfg.profit_range

        # number of samples
        prob['n_samples_inst'] = cfg.n_samples_inst
        prob['n_samples_per_inst'] = cfg.n_samples_per_inst
        prob['n_samples'] = cfg.n_samples_inst * cfg.n_samples_per_inst

        prob['time_limit'] = cfg.time_limit
        prob['mip_gap'] = cfg.mip_gap
        prob['verbose'] = cfg.verbose
        prob['threads'] = cfg.threads
        prob['tr_split'] = cfg.tr_split

        # generic parameters
        prob['seed'] = cfg.seed
        prob['data_path'] = cfg.data_path
        
        return prob