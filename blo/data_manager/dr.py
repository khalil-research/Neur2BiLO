import os
import pickle as pkl
import time
from multiprocessing import Manager, Pool

import gurobipy as gp
import numpy as np

from blo.blo.dr import DonorRecipient
from blo.utils.dr import get_path
from .data_manager import DataManager


class DonorRecipientDataManager(DataManager):

    def __init__(self, cfg):
        """ Constructor for knapsack problem.  """
        self.cfg = cfg

        self.problem_path = get_path(self.cfg.data_path, self.cfg, "problem")
        self.ml_data_path = get_path(self.cfg.data_path, self.cfg, "ml_data")

        self.blo = DonorRecipient()


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
        m = self.blo.create_dr_model(instance)

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
        # sample budget between 0 and Br
        budget = instance['Bd']
        sampled_budget = np.random.uniform(0, budget)    

        # sample x unifromly at random
        x = np.random.uniform(0, 1, size=instance['n'])

        # normalize x such that c @ x == budget
        x = sampled_budget * x / (instance['c'] @ x)

        return x


    def _get_problem_data(self, cfg):
        """ Stores generic problem information. """
        prob = {}

        # values for CNG instances
        prob["n"] = cfg.n
        prob["cost_range"] = cfg.cost_range
        prob["cost_ext_range"] = cfg.cost_ext_range
        prob["p_c_ratios"] = cfg.p_c_ratios
        prob["hard"] = cfg.hard
        prob["budget_donor_perc"] = cfg.budget_donor_perc
        prob["budget_rec_perc"] = cfg.budget_rec_perc

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