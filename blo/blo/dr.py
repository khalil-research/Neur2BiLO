from multiprocessing import Pool, Manager

import gurobipy as gp
import pandas as pd
import numpy as np
import copy

import pyomo.environ as pe
from pao.pyomo import SubModel
from pyomo.environ import value

from .blo import BLO


class DonorRecipient(BLO):
    def __init__(self):
        pass

    def sample_instance(self, cfg, scale=True, seed=None):
        """ Samples instance. """
        if cfg.hard:
            return self.sample_instance_hard(cfg, scale, seed)
        else:
            raise Exception("Only sampling for hard instances (dataset 15 from reference) is impleemted.")


    def sample_instance_hard(self, cfg, scale=True, seed=None):
        """ Samples instance. """
        if seed is not None:
            np.random.seed(seed)
        n = np.random.choice(cfg.n)
        n_per_class = int(n / 3)
        # get cost and profit vectors
        if cfg.hard:
            alpha1 = np.random.uniform(0.01, 0.1, n_per_class)
            alpha = np.array([*alpha1, *np.ones(n_per_class*2)])
        else:
            raise Exception("Only sampling for hard instances (dataset 15 from reference) is impleemted.")
            # alpha = np.ones(n)
            
        gamma = np.random.uniform(low=cfg.gamma_range[0], high=cfg.gamma_range[1])
        c = np.random.randint(low=cfg.cost_range[0], high=cfg.cost_range[1], size=n)
        c0 = np.random.randint(low=cfg.cost_ext_range[0], high=cfg.cost_ext_range[1])
        v = np.array(copy.deepcopy(c), dtype=float)
        v0 = copy.deepcopy(c0)
        for i in range(3):
            v[i * n_per_class:(i + 1) * n_per_class] *= cfg.p_c_ratios[i]
            pass
        v0 = v0 * np.mean(cfg.p_c_ratios) * gamma
        w = copy.deepcopy(v)*alpha
        w = w.astype(int)
        v = v.astype(int)
        v0 = int(v0)
        # get budgets
        sum_cost = sum(c)
        Bd = int(sum_cost * cfg.budget_donor_perc)
        Br = int(sum_cost * cfg.budget_rec_perc)
        if scale:
            # scale objectives
            w = w / v0
            v = v / v0
            v0 = 1

            # scale constraints
            c = c / c0
            Bd = Bd / c0
            Br = Br / c0
            c0 = 1

        # set instance dict
        instance_dict = {
            'n': n,
            'w': w,
            'v': v,
            'c': c,
            'v0': v0,
            'c0': c0,
            'Bd': Bd,
            'Br': Br,
            "budget_donor_perc": cfg.budget_donor_perc
        }
        
        return instance_dict


    def read_instance(self, cfg, id_data, i, scale):
        """ Reads instances. """
        # specify and read instances file
        instance_file = 'Instance%i.xlsx' % i
        fp_inst = cfg.data_path + 'dr/DR-BKP-main/DataSet%i/' % id_data + instance_file
        f = pd.read_excel(fp_inst, 'HCProjects', index_col=0)

        w = f.DonorValuation.to_numpy()
        v = f.CountryValuation.to_numpy()
        c = f.ProjectCost.to_numpy()
        n = int(len(w))

        f2 = pd.read_excel(fp_inst, 'ExtProject', index_col=0)
        v0 = f2.iloc[0]["CountryValuation"]
        c0 = f2.iloc[0]["ProjectCost"]

        f3 = pd.read_excel(fp_inst, 'Budget', index_col=0)
        Bd = f3.iloc[0]['Donor']
        Br = f3.iloc[0]['Country']

        if scale:
            # scale objectives
            w = w / v0
            v = v / v0
            v0 = 1

            # scale constraints
            c = c / c0
            Bd = Bd / c0
            Br = Br / c0
            c0 = 1

        instance_dict = {
            'n': n,
            'w': w,
            'v': v,
            'c': c,
            'v0': v0,
            'c0': c0,
            'Bd': Bd,
            'Br': Br
        }

        return self.create_dr_model(instance_dict)


    def create_dr_model(self, instance_dict):
        """ Creates model for DonorReceipient problem. """
        def donor_obj_rule(model):
            """ Defender objective rule. """
            obj = 0
            for i in model.projects:
                obj += model.y[i] * model.w[i]
            return obj

        def recipient_obj_rule(model):
            """ Attacker objective rule. """
            obj = 0
            for i in model.projects:
                obj += model.y[i] * model.v[i]
            obj += model.y0 * model.v0
            return obj

        # create model object
        M = pe.ConcreteModel()

        M.n = instance_dict['n']
        M.w = instance_dict['w']
        M.v = instance_dict['v']
        M.c = instance_dict['c']
        M.v0 = instance_dict['v0']
        M.c0 = instance_dict['c0']
        M.Bd = instance_dict['Bd']
        M.Br = instance_dict['Br']
        M.inst_dict = instance_dict
        M.projects = set(range(M.n))

        # define decision variables
        M.x = pe.Var(M.projects, bounds=(0, 1))
        M.y = pe.Var(M.projects, domain=pe.Binary)
        M.y0 = pe.Var(bounds=(0, 1))

        # upper-level (donor) objective
        M.o = pe.Objective(expr=donor_obj_rule(M), sense=pe.maximize)

        # upper-level budget constraint
        M.d_budget_constr = pe.Constraint(expr=sum(M.x[i] * M.c[i] for i in M.projects) <= M.Bd)

        # create a SubModel component to declare a lower-level problem
        # the variable M.x is fixed in this lower-level problem
        M.L = SubModel(fixed=M.x)

        # set upper-level (attacker) objective
        M.L.o = pe.Objective(expr=recipient_obj_rule(M), sense=pe.maximize)

        M.L.r_budget_constr = pe.Constraint(expr=sum((M.c[i] - M.c[i]*M.x[i])*M.y[i] for i in M.projects) + M.c0*M.y0 <= M.Br)

        return M


    def solve_follower(self, opt_model, x):
        """ Solve follower problem. """
        for i in range(len(x)):
            opt_model.x[i].fix(x[i])
        pe.SolverFactory('gurobi').solve(opt_model.L)

        res = {
            'leader_obj' : value(opt_model.o),
            'follower_obj' : value(opt_model.L.o),
            'leader_sol' : x,
            'follower_sol' : [[pe.value(opt_model.y[key]) for key in opt_model.y], pe.value(opt_model.y0)],
        }

        return res