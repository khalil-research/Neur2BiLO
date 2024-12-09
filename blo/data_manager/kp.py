import os
import pickle as pkl
import time
from multiprocessing import Manager, Pool

import gurobipy as gp
import numpy as np

from blo.blo.kp import Knapsack
from blo.utils.kp import get_path
from .data_manager import DataManager


class KnapsackDataManager(DataManager):


    def __init__(self, cfg):
        """ Constructor for knapsack problem.  """
        self.cfg = cfg

        self.problem_path = get_path(self.cfg.data_path, self.cfg, "problem")
        self.ml_data_path = get_path(self.cfg.data_path, self.cfg, "ml_data")

        self.blo = Knapsack()


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
        m = self.blo.create_knapsack_model(**instance)

        # solve follower for fixed x
        solve_res = self.blo.solve_follower(m, x)
        follower_obj = solve_res["follower_obj"]
        follower_sol = solve_res["follower_sol"]

        # run double greedy
        double_greedy_obj, double_greedy_sol = self.blo.run_double_greedy(instance, m)

        # run greedy
        greedy_obj, greedy_sol = self.blo.run_greedy(x, instance, m)

        # compute some additional statistics
        greedy_y_approx = np.abs(follower_obj - greedy_obj) / follower_obj

        time_ = time.time() - time_

        # store results
        results = {
            'x' : x,
            'instance' : instance,
            'inst_id' : inst_id,
            'follower_obj' : follower_obj,
            'follower_sol' : follower_sol,
            'double_greedy_obj' : double_greedy_obj,
            'double_greedy_sol' : double_greedy_sol,
            'greedy_obj' : greedy_obj,
            'greedy_sol' : greedy_sol,
            'greedy_y_approx' : greedy_y_approx,
            'solve_res' : solve_res,
        }

        self.update_mp_status(mp_count, mp_time, n_samples)

        return results


    def _sample_random_x(self, instance, X_hash=None):
        """ Samples random x decision. """
        # get item probs (can be more efficient by precomputing).
        # however, this is ideal for multiprocessing and modularity
        ratios = instance['p']/instance['a']
        item_prob = ratios/np.sum(ratios)

        # sample k between 1 and k
        # k_cur = np.random.randint(low=1, high=instance['k']+1, size=1)
        
        # sample from list of ratios
        # k_ratio_idx = np.random.choice(range(len(self.prob['k_ratio'])))
        # k_ratio = self.prob['k_ratio'][k_ratio_idx]
        # k_cur = int(np.ceil(instance['n'] * k_ratio[0] / k_ratio[1]))

        k_ratio = np.random.choice(self.prob['k_ratio']) # [k_ratio_idx]
        k_cur = int(np.ceil(instance['n'] * k_ratio))

        x_nnz = np.random.randint(low=0, high=instance['n'], size=k_cur)
        
        x = np.zeros(instance['n'])

        # sample x until unique (new) upper-level decision is found
        while True:
            x_nnz = np.sort(np.random.choice(instance['n'], k_cur, replace=False, p=item_prob))
            x_nnz_str = np.array2string(x_nnz)

            # if no dict is given, simply sample once
            if X_hash is None:
                break

            # if x in dict, then break
            if x_nnz_str not in X_hash:
                break

        X_hash.add(x_nnz_str)
        x[x_nnz] = 1

        return x


    def _get_problem_data(self, cfg):
        """ Stores generic problem information. """
        prob = {}
        prob['prob_type'] = cfg.prob_type
        prob['n'] = cfg.n
        prob['k_ratio'] = cfg.k_ratio

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