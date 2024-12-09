from multiprocessing import Pool, Manager

import gurobipy as gp
import numpy as np

import pyomo.environ as pe
from pao.pyomo import SubModel
from pyomo.environ import value

from .blo import BLO


class CriticalNodeGame(BLO):

    def __init__(self):
        pass


    def sample_instance(self, cfg, scale=True):
        """ Samples instance. """
        if cfg.prob_type == "dragotto_2023":
            instance = self.sample_instance_dragotto_2023(cfg, scale=scale)

        return instance


    def sample_instance_dragotto_2023(self, cfg, scale, seed=None):
        """ Samples problem of n items and k interdecition from Dragotto 2023. """
        if seed is not None:
            np.random.seed(seed)

        # sample number of nodes
        v = np.random.choice(cfg.v)

        # sample problem parameters.  Note most of these can only take 1 value
        gamma = np.random.choice(cfg.gamma)
        eta = np.random.choice(cfg.eta)
        epsilon_ratio = np.random.choice(cfg.epsilon_ratio)
        delta_ratio = np.random.choice(cfg.delta_ratio)
        d_ratio = np.random.choice(cfg.d_ratio)
        a_ratio = np.random.choice(cfg.a_ratio)

        # compute epsilon and delta
        epsilon = epsilon_ratio * eta
        delta = delta_ratio * eta

        # sample budgets for defender and attacker
        d_budget_coefs = np.random.randint(low=cfg.budget_range[0], high=cfg.budget_range[1], size=v)
        a_budget_coefs = np.random.randint(low=cfg.budget_range[0], high=cfg.budget_range[1], size=v)

        # sample profits for defender
        d_profit_coefs = np.random.randint(low=cfg.profit_range[0], high=cfg.profit_range[1], size=v) 
        d_profit_coefs += np.random.randint(low=cfg.profit_range[0], high=cfg.profit_range[1], size=v)

        # sample profits for attacker
        a_profit_coefs = np.random.randint(low=cfg.profit_range[0], high=cfg.profit_range[1], size=v)
        a_profit_coefs += np.random.randint(low=cfg.profit_range[0], high=cfg.profit_range[1], size=v)

        # get exact budgets using coefficients sampled
        D = d_ratio * np.sum(d_budget_coefs)
        A = a_ratio * np.sum(a_budget_coefs)

        # scale profits/weights/budgets
        if scale:
            d_budget_coefs = d_budget_coefs / D
            D = 1

            a_budget_coefs = a_budget_coefs / A
            A = 1

            d_p_max = np.max(d_profit_coefs)
            d_profit_coefs = d_profit_coefs / d_p_max

            a_p_max = np.max(a_profit_coefs)
            a_profit_coefs = a_profit_coefs / a_p_max

        else:
            d_p_max = 1
            a_p_max = 1

        # set instance dict
        instance = {
            'v' : v,
            'gamma' : gamma,
            'eta' : eta,
            'epsilon_ratio' : epsilon_ratio,
            'delta_ratio' : delta_ratio,
            'epsilon' : epsilon,
            'delta' : delta,
            'd_ratio' : d_ratio,
            'a_ratio' : a_ratio,
            'd_budget_coefs' : d_budget_coefs,
            'a_budget_coefs' : a_budget_coefs,
            'd_profit_coefs' : d_profit_coefs,
            'a_profit_coefs' : a_profit_coefs,
            'D' : D,
            'A' : A,
            'd_p_max' : d_p_max,
            'a_p_max' : a_p_max,
        }

        return instance


    def read_instance(self, cfg, n, k, i, scale):
        """ Reads instances.  """
        instance = self.read_instance_dragotto_2023(cfg, n, k, i, scale=scale)
        return instance


    def read_instance_dragotto_2023(self, cfg, n, k, i, scale=True):
        """ Reads CNG instance from Dragotto et al. 2013.  """
        # not implemented, only sampling for now
        raise Exception("Instance reading not implemented for CNG!")


    def create_cng_model(self, instance_dict):
        """ Creates model for knapsack problem. """
        def defender_obj_rule(model):
            """ Defender objective rule. """
            obj_total = 0
            for i in range(model.v):
                obj = (1 - model.x[i]) * (1 - model.y[i])               # no defend, no attack
                obj += model.eta * model.x[i] * model.y[i]              # defend, attack 
                obj += model.epsilon * model.x[i] * (1 - model.y[i])    # defend, no attack
                obj += model.delta * (1 - model.x[i]) * model.y[i]      # no defend, attack
                obj = model.d_profit_coefs[i] * obj
                obj_total += obj

            return obj_total

        def attacker_obj_rule(model):
            """ Attacker objective rule. """
            obj_total = 0
            for i in range(model.v):
                obj = - model.gamma * (1 - model.x[i]) * (1 - model.y[i])   # no defend, no attack
                obj += (1 - model.x[i]) * model.y[i]                        # no defend, attack 
                obj += (1 - model.eta) * model.x[i] * model.y[i]            # defend, attack 
                obj = model.a_profit_coefs[i] * obj
                obj_total += obj

            return obj_total

        # create model object
        M = pe.ConcreteModel()

        M.inst_dict = instance_dict
        
        M.v = instance_dict['v']
        M.gamma = instance_dict['gamma']
        M.eta = instance_dict['eta']
        M.epsilon = instance_dict['epsilon']
        M.delta = instance_dict['delta']
        M.d_ratio = instance_dict['d_ratio']
        M.a_ratio = instance_dict['a_ratio']
        M.d_budget_coefs = instance_dict['d_budget_coefs']
        M.a_budget_coefs = instance_dict['a_budget_coefs']
        M.d_profit_coefs = instance_dict['d_profit_coefs']
        M.a_profit_coefs = instance_dict['a_profit_coefs']
        M.D = instance_dict['D']
        M.A = instance_dict['A']
        M.d_p_max = instance_dict['d_p_max']
        M.a_p_max = instance_dict['a_p_max']

        # set for the nodes in the graph
        M.V = pe.Set(initialize=list(range(M.v)))

        # define decision variables
        M.x = pe.Var(M.V, domain=pe.Binary)
        M.y = pe.Var(M.V, domain=pe.Binary)

        # upper-level (defender) objective
        M.o = pe.Objective(expr = defender_obj_rule(M), sense = pe.maximize)

        # upper-level budget constraint
        M.d_budget_constr = pe.Constraint(expr=sum(M.x[i] * M.d_budget_coefs[i] for i in M.V) <= M.D)

        # create a SubModel component to declare a lower-level problem
        # the variable M.x is fixed in this lower-level problem
        M.L = SubModel(fixed=M.x)

        # set upper-level (attacker) objective
        M.L.o = pe.Objective(expr = attacker_obj_rule(M), sense = pe.maximize)

        # define lower-level constraints
        M.L.a_budget_constr = pe.Constraint(expr=sum(M.y[i] * M.a_budget_coefs[i] for i in M.V) <= M.A)

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
            'follower_sol' : [pe.value(opt_model.y[key]) for key in opt_model.y],
        }

        return res


