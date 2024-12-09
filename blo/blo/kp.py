from multiprocessing import Pool, Manager

import gurobipy as gp
import numpy as np

import pyomo.environ as pe
from pao.pyomo import SubModel
from pyomo.environ import value

from .blo import BLO


class Knapsack(BLO):

    def __init__(self):
        pass


    def sample_instance(self, cfg, scale=True):
        """ Samples instance. """
        n = np.random.choice(cfg.n)
                
        k_ratio = np.random.choice(cfg.k_ratio)
        k = int(np.ceil(n * k_ratio))

        if cfg.prob_type == "tang_2016":
            instance = self.sample_instance_tang_2016(n, k, scale=scale)

        return instance


    def sample_instance_tang_2016(self, n, k, scale, seed=None):
        """ Samples problem of n items and k interdecition from Tang, 2016. """
        if seed is not None:
            np.random.seed(seed)
            
        bad_instance = True
        while bad_instance:
            a = np.random.randint(low=1, high=101, size=n)
            p = np.random.randint(low=1, high=101, size=n)
            b = np.ceil((n-k)*np.sum(a)/(2*n))
            bad_instance = (np.max(a) > b)
        ratio = p/a
        order = np.argsort(ratio)
        a = a[order]
        p = p[order]

        if scale:
            a = a/b
            b = 1
            p_max = np.max(p)
            p = p / p_max

        else:
            p_max = 1

        instance = {
            'a' : a,
            'p' : p,
            'b' : b,
            'p_max' : p_max,
            'n' : n,
            'k' : k,
        }

        return instance


    def read_instance(self, cfg, n, k, i, scale):
        """ Reads instances.  """
        if cfg.prob_type == "tang_2016":
            instance= self.read_instance_tang_2016(cfg, n, k, i, scale=scale)
        return instance



    def read_instance_tang_2016(self, cfg, n, k, i, scale=True):
        """ Reads knapsack instance from Tang et al. 2016.  """
        # specify and read instances file
        instance_file = 'BKPIns_%i_%i_%i.txt' % (n, k, i)
        fp_inst = cfg.data_path + 'kp/BKPIns_ver2/' + instance_file
        f = open(fp_inst, 'r')
        bkp_inst = f.read()

        # bkp inst to array
        bkp_arr = bkp_inst.splitlines()


        # n, k, b = int(bkp_arr[1]), int(bkp_arr[2]), int(bkp_arr[3])
        b = int(bkp_arr[3])
        p = np.array([int(i) for i in bkp_arr[4].split()])
        a = np.array([int(i) for i in bkp_arr[5].split()])

        ratio = p/a
        order = np.argsort(ratio)
        a = a[order]
        p = p[order]

        if scale:
            a = a/b
            b = 1
            p_max = np.max(p)
            p = p / p_max

        else:
            p_max = 1

        return self.create_knapsack_model(a, p, b, p_max, n, k)


    def create_knapsack_model(self, a, p, b, p_max, n, k):
        """ Creates model for knapsack problem. """
        # create model object
        M = pe.ConcreteModel()

        M.k = k
        M.a = a
        M.p = p
        M.p_max = p_max
        M.b = b

        M.I = pe.Set(initialize=list(range(n)))

        # define decision variables
        M.x = pe.Var(M.I, domain=pe.Binary)
        M.y = pe.Var(M.I, domain=pe.Binary)

        # define the upper-level objective
        M.o = pe.Objective(expr=sum(M.p[i]*M.y[i] for i in M.I))

        # upper-level budget constraint
        M.c = pe.Constraint(expr=sum(M.x[i] for i in M.I) <= k)

        # create a SubModel component to declare a lower-level problem
        # the variable M.x is fixed in this lower-level problem
        M.L = SubModel(fixed=M.x)

        # define the lower-level objective
        M.L.o = pe.Objective(expr=sum(-M.p[i]*M.y[i] for i in M.I))

        # define lower-level constraints
        M.L.c_budget = pe.Constraint(expr=sum(M.a[i]*M.y[i] for i in M.I) <= M.b)

        # volume constraint
        M.L.c_interdict = pe.ConstraintList()
        for i in M.I:
          M.L.c_interdict.add(M.y[i] <= 1 - M.x[i])

        return M


    def solve_follower(self, opt_model, x):
        """ Solve follower problem. """
        for i in range(len(x)):
            opt_model.x[i].fix(x[i])
        # pe.SolverFactory('glpk', executable='/usr/bin/glpsol').solve(opt_model.L)
        pe.SolverFactory('glpk').solve(opt_model.L)

        res = {
            'follower_obj' : -1 * value(opt_model.L.o),
            'follower_sol' : [pe.value(opt_model.y[key]) for key in opt_model.y],
            'leader_obj' : -1 * value(opt_model.L.o),
            'leader_sol' : x,
        }

        return res


    def greedy(self, opt_model, x, n, k): 
        """ Double greedy function implementation. """
        budget_left = opt_model.b
        val = 0
        y = np.zeros(n)

        # iterate over all items in knapsack and add greedy item
        for i in range(n-1, -1, -1):
            
            # skip if interdiction
            if x[i] >= 0.5:
                continue

            # otherwise, add if fits in budget
            size_cur = opt_model.a[i]
            profit_cur = opt_model.p[i]
            if size_cur <= budget_left:
                y[i] = 1 
                val += profit_cur
                budget_left -= size_cur

        return val, y


    def run_greedy(self, x, instance, opt_model):
        """ Runs greedy on an instance.  """
        if opt_model is None:
            print("getting model")
            opt_model = self.create_knapsack_model(**instance)

        n = len(opt_model.I) # instance["n"]
        k = opt_model.k # instance["k"]

        greedy_obj, greedy_sol = self.greedy(opt_model, x, n, k)

        return greedy_obj, greedy_sol


    def run_double_greedy(self, instance, opt_model):
        """ Runs double greedy on an instance.  """
        if opt_model is None:
            print("getting model")
            opt_model = self.create_knapsack_model(**instance)

        n = len(opt_model.I) # instance["n"]
        k = opt_model.k # instance["k"]

        # get greedy upper-level decision
        x_greedy = np.zeros(n)
        x_greedy[n - k:] = 1

        # get double greedy objective
        double_greedy_obj, double_greedy_sol = self.greedy(opt_model, x_greedy, n, k)

        return double_greedy_obj, double_greedy_sol

