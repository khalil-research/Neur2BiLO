import time
import copy
import collections
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt

import gurobipy as gp
from gurobi_ml import add_predictor_constr

from gurobipy import quicksum, min_

import torch
import torch.nn as nn

import blo.params as blo_params
from blo.data_preprocessor import factory_dp
from blo.data_preprocessor.kp import KnapsackDataPreprocessor
from blo.utils.kp import get_path

from .approximator import Approximator



class KnapsackApproximator(Approximator):

    def __init__(self, args, cfg, blo, net, instance):
        """ Constructor for Knapsack Aproximator.  """
        super(KnapsackApproximator, self).__init__(args, cfg, blo, net, instance)

        # knapsack specific parameters
        self.n = len(self.instance.I)

        if net is not None or self.args.model_type == "greedy":
            self.use_greedy = net["kp_use_greedy"]

            print(f"ML-Approximator using greedy? {self.use_greedy}")


    def recover_sol(self, grb_model):
        """ Recovers solution information for gurobi model. """
        all_vars = grb_model.getVars()
        var_names = grb_model.getAttr("VarName", all_vars)
        var_values = grb_model.getAttr("X", all_vars)

        x_values = []
        y_values = []
        y_vf = 0
        y_vf_us = 0
        y_vfg = 0
        y_pred = []
        for name, val in zip(var_names, var_values):
            if 'x[' == name[:2]:
                x_values += [round(val)]
            if 'y[' == name[:2]:
                y_values += [round(val)]
            if 'y_vf[' in name:
                y_vf = val
            if 'y_vfg[' in name:
                y_vfg = val
            if 'y_pred[' in name:
                y_pred += [val]
            if 'y_vf_unscaled[' in name:
                y_vf_us = val

        sol = {
            'x' : x_values,
            'y' : y_values,
            'y_vf' : y_vf,
            'y_vfg' : y_vfg,
            'y_vf_us' : y_vf_us,
        }

        if self.args.model_type != "greedy":
            sol["a"] = self.instance.a
            sol["p"] = self.instance.p
            sol["y @ a"] = y_values @ self.instance.a
            sol["y @ p"] = y_values @ self.instance.p
            sol["y_pred @ p"] = y_pred @ self.instance.p
            sol["y_pred"] = y_pred

        return sol


    def get_unscaled_valuefun(self, y_valuefun, grb_model):
        """ Gets unscaled value function.  """
        k = self.instance.k
        y_valuefun_unscaled =  grb_model.addMVar((1,), lb=-gp.GRB.INFINITY, name="y_vf_unscaled")
        grb_model.addConstr(y_valuefun == ((y_valuefun_unscaled - self.label_scaler[k][0]) / (self.label_scaler[k][1] - self.label_scaler[k][0])) + 1) 
        return y_valuefun_unscaled


    def do_warmstart(self, grb_model):
        """ Does warmstart.  Can be initialized to do nothing for some problems. """
        if self.args.warmstart:
            grb_model.NumStart = 1
            grb_model.update()
            # set StartNumber
            grb_model.params.StartNumber = 0
            # now set MIP start values using the Start attribute, e.g.:
            for v in grb_model.getVars():
                name = v.VarName
                if 'x[' == name[:2]:
                    v.Start = int(int(name[2:-1]) >= self.n - self.instance.k)


    def get_grb_features_with_x(self, x, grb_model):
        """ Initialize gurobi variables for problem features (i.e., input to NN).  """
        
        # get greedy
        if self.use_greedy:
            y_valuefun_greedy, yg = self.embed_greedy(x, grb_model)
        
        # get double greedy
        val_double_greedy, y_double_greedy = self.blo.run_double_greedy(None, self.instance)
        x_double_greedy = np.zeros(self.n)
        x_double_greedy[self.n - self.instance.k:] = 1

        # compute l1 distance to double greedy solution
        l1_pre_vars = grb_model.addVars(self.n, name="l1_pre_vars")
        l1_vars = grb_model.addVars(self.n, name="l1_vars")
        for i in range(self.n):
            grb_model.addConstr(l1_pre_vars[i] == x[i] - x_double_greedy[i])
            grb_model.addConstr(l1_vars[i] == gp.abs_(l1_pre_vars[i]))

        l1_double_greedy = gp.quicksum(l1_vars) / self.n

        # max ratio
        # f_max = self.instance.p / self.instance.a
        f_vals = self.instance.p / self.instance.a
        f_max = np.max(f_vals)

        if self.model_type == "greedy":
            x_withfeatures = None

        elif self.model_type == "ff_fixed":
            x_withfeatures = grb_model.addMVar((self.input_dim,), vtype=(self.input_dim)*[gp.GRB.CONTINUOUS], lb=0 , name="x_wf")

            # set (1-x) * p/a
            grb_model.addConstrs((x_withfeatures[i] == (1-x[i])*(self.instance.p[i]/self.instance.a[i]) for i in range(self.n)), name='set_f')
            
            # set p
            grb_model.addConstrs((x_withfeatures[self.n+i] == self.instance.p[i] for i in range(self.n)), name='set_p')
            
            # set greedy/double greedy
            grb_model.addConstr(x_withfeatures[-4] == y_valuefun_greedy, name='set_vf_g')
            grb_model.addConstr(x_withfeatures[-3] == val_double_greedy, name='set_vf_dg')

            # set k and b
            grb_model.addConstr(x_withfeatures[-2] == self.instance.k, name='set_k')
            grb_model.addConstr(x_withfeatures[-1] == self.instance.b, name='set_b')

        elif self.model_type == "ff_invariant":
            x_withfeatures = grb_model.addMVar((self.input_dim,), vtype=(self.input_dim)*[gp.GRB.CONTINUOUS], lb=0 , name="x_wf")

            # features for instance
            grb_model.addConstr(x_withfeatures[0] == self.n, name='set_n')
            grb_model.addConstr(x_withfeatures[1] == self.instance.k, name='set_k')
            grb_model.addConstr(x_withfeatures[2] == self.instance.b, name='set_b')

            # number of active items in upper level solution
            grb_model.addConstr(x_withfeatures[3] == gp.quicksum(x), name='set_sum_x')
           
            # features for (1-x) * p
            p_x = grb_model.addMVar(self.n, vtype=gp.GRB.CONTINUOUS, lb=0 , name="p_x")
            grb_model.addConstrs((p_x[i] == self.instance.p[i] * (1 - x[i]) for i in range(self.n)), name='set_p_x')
            grb_model.addConstr(x_withfeatures[4] == gp.min_([p_x[i] for i in range(self.n)]), name='set_min_p_x')
            grb_model.addConstr(x_withfeatures[5] == gp.max_([p_x[i] for i in range(self.n)]), name='set_max_p_x')
            grb_model.addConstr(x_withfeatures[6] == gp.quicksum(p_x) / self.n, name='set_mean_p_x')
            p_x_var = gp.quicksum([(p_x[i] - x_withfeatures[6]) * (p_x[i] - x_withfeatures[6]) for i in range(self.n)]) / self.n
            grb_model.addConstr(x_withfeatures[7] == p_x_var, name='set_var_p_x')

            # features for (1-x) * a
            a_x = grb_model.addMVar(self.n, vtype=gp.GRB.CONTINUOUS, lb=0 , name="a_x")
            grb_model.addConstrs((a_x[i] == self.instance.a[i] * (1 - x[i]) for i in range(self.n)), name='set_a_x')
            grb_model.addConstr(x_withfeatures[8] == gp.min_([a_x[i] for i in range(self.n)]), name='set_min_a_x')
            grb_model.addConstr(x_withfeatures[9] == gp.max_([a_x[i] for i in range(self.n)]), name='set_max_a_x')
            grb_model.addConstr(x_withfeatures[10] == gp.quicksum(a_x) / self.n, name='set_mean_a_x')
            a_x_var = gp.quicksum([(a_x[i] - x_withfeatures[10]) * (a_x[i] - x_withfeatures[10]) for i in range(self.n)]) / self.n
            grb_model.addConstr(x_withfeatures[11] == a_x_var, name='set_var_a_x')

            # features for (1-x) * a
            pa_x = grb_model.addMVar(self.n, vtype=gp.GRB.CONTINUOUS, lb=0 , name="pa_x")
            grb_model.addConstrs((pa_x[i] == self.instance.p[i] / self.instance.a[i] * (1 - x[i]) for i in range(self.n)), name='set_pa_x')
            grb_model.addConstr(x_withfeatures[12] == gp.min_([pa_x[i] for i in range(self.n)]), name='set_min_pa_x')
            grb_model.addConstr(x_withfeatures[13] == gp.max_([pa_x[i] for i in range(self.n)]), name='set_max_pa_x')
            grb_model.addConstr(x_withfeatures[14] == gp.quicksum(pa_x) / self.n, name='set_mean_pa_x')
            pa_x_var = gp.quicksum([(pa_x[i] - x_withfeatures[14]) * (pa_x[i] - x_withfeatures[14]) for i in range(self.n)]) / self.n
            grb_model.addConstr(x_withfeatures[15] == pa_x_var, name='set_var_pa_x')

            # greedy output features
            grb_model.addConstr(x_withfeatures[-2] == y_valuefun_greedy, name='set_vf_g')
            grb_model.addConstr(x_withfeatures[-1] == val_double_greedy, name='set_vf_dg')
            
        elif self.model_type == "set_invariant":
            x_withfeatures = grb_model.addMVar((self.n, self.input_dim), vtype=gp.GRB.CONTINUOUS, lb=0 , name="x_wf")
            
            for i in range(self.n):

                # features from multipling by (1-x)
                grb_model.addConstr(x_withfeatures[i,0] == (1 - x[i]) * self.instance.p[i] / self.instance.a[i], name=f'set_f[{i}]')
                grb_model.addConstr(x_withfeatures[i,1] == (1 - x[i]) * self.instance.a[i], name=f'set_a[{i}]')
                grb_model.addConstr(x_withfeatures[i,2] == (1 - x[i]) * self.instance.p[i], name=f'set_p[{i}]')
                grb_model.addConstr(x_withfeatures[i,3] == (1 - x[i]) * yg[i], name=f"set_y_g_{i}")
                grb_model.addConstr(x_withfeatures[i,4] == (1 - x[i]) * ydg[i], name=f"set_y_dg_{i}")
                grb_model.addConstr(x_withfeatures[i,5] == (1 - x[i]) * y_valuefun_greedy, name=f"set_g_{i}")
                grb_model.addConstr(x_withfeatures[i,6] == (1 - x[i]) * val_double_greedy, name=f"set_dg_{i}")
                grb_model.addConstr(x_withfeatures[i,7] == (1 - x[i]) * self.instance.k, name=f"set_k_{i}")
                grb_model.addConstr(x_withfeatures[i,8] == (1 - x[i]) * self.instance.b, name=f"set_b_{i}")

                # # features from notebook
                # item_feats = [(1-x) * f_val[i], p_val[i]]
                # grb_model.addConstr(x_withfeatures[i,0] == (1 - x[i]) * self.instance.p[i] / self.instance.a[i], name=f'set_f[{i}]')
                # grb_model.addConstr(x_withfeatures[i,1] == self.instance.p[i], name=f'set_f[{i}]')

        elif self.model_type == "inst_encoder":
            x_withfeatures = grb_model.addMVar((self.n, self.input_dim), vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY , name="x_wf")

            # get instance features
            instance_feats = self.get_instance_features()

            # embed instance information
            x_inst_embedding = self.net.instance_decision_embedder(instance_feats)
            x_inst_embedding = self.net.aggregate(x_inst_embedding, self.net.agg_type)
            x_inst_embedding = self.net.final_instance_embedder(x_inst_embedding)

            x_inst_embedding = x_inst_embedding.detach().cpu().numpy().reshape(-1)
            x_inst_embedding = 0 * x_inst_embedding

            # print(self.net(instance_feats, 0, 0, 0, 0, print_embedding=True))
            for i in range(self.n):

                if self.use_greedy:
                    # features from multipling by (1-x) with greedy constraints
                    grb_model.addConstr(x_withfeatures[i,0] == (1 - x[i]) * x[i], name=f'set_x[{i}]')
                    grb_model.addConstr(x_withfeatures[i,1] == (1 - x[i]) * f_vals[i] / f_max, name=f'set_f[{i}]')
                    grb_model.addConstr(x_withfeatures[i,2] == (1 - x[i]) * self.instance.a[i], name=f'set_a[{i}]')
                    grb_model.addConstr(x_withfeatures[i,3] == (1 - x[i]) * self.instance.p[i], name=f'set_p[{i}]')
                    grb_model.addConstr(x_withfeatures[i,4] == (1 - x[i]) * yg[i], name=f"set_y_g_{i}")
                    grb_model.addConstr(x_withfeatures[i,5] == (1 - x[i]) * y_valuefun_greedy / self.n, name=f"set_g_{i}")
                    grb_model.addConstr(x_withfeatures[i,6] == (1 - x[i]) * x_double_greedy[i], name=f"set_x_dg_{i}")
                    grb_model.addConstr(x_withfeatures[i,7] == (1 - x[i]) * y_double_greedy[i], name=f"set_y_dg_{i}")
                    grb_model.addConstr(x_withfeatures[i,8] == (1 - x[i]) * val_double_greedy / self.n, name=f"set_dg_{i}")
                    grb_model.addConstr(x_withfeatures[i,9] == (1 - x[i]) * l1_double_greedy, name=f"set_l1_dg_{i}")
                    grb_model.addConstr(x_withfeatures[i,10] == (1 - x[i]) * self.instance.k / self.n, name=f"set_k_ratio_{i}")
                    grb_model.addConstr(x_withfeatures[i,11] == (1 - x[i]) * self.instance.b, name=f"set_b_{i}")

                    n_feats = 12
    
                else:
                    # features from multipling by (1-x) without greedy constraints
                    grb_model.addConstr(x_withfeatures[i,0] == (1 - x[i]) * x[i], name=f'set_x[{i}]')
                    grb_model.addConstr(x_withfeatures[i,1] == (1 - x[i]) * f_vals[i] / f_max, name=f'set_f[{i}]')
                    grb_model.addConstr(x_withfeatures[i,2] == (1 - x[i]) * self.instance.a[i], name=f'set_a[{i}]')
                    grb_model.addConstr(x_withfeatures[i,3] == (1 - x[i]) * self.instance.p[i], name=f'set_p[{i}]')
                    grb_model.addConstr(x_withfeatures[i,4] == (1 - x[i]) * x_double_greedy[i], name=f'set_x_dg_[{i}]')
                    grb_model.addConstr(x_withfeatures[i,5] == (1 - x[i]) * y_double_greedy[i], name=f"set_y_dg_{i}")
                    grb_model.addConstr(x_withfeatures[i,6] == (1 - x[i]) * val_double_greedy / self.n, name=f"set_vf_dg_{i}")
                    grb_model.addConstr(x_withfeatures[i,7] == (1 - x[i]) * l1_double_greedy, name=f"set_l1_dg_{i}")
                    grb_model.addConstr(x_withfeatures[i,8] == (1 - x[i]) * self.instance.k / self.n, name=f"set_k_ratio_{i}")
                    grb_model.addConstr(x_withfeatures[i,9] == (1 - x[i]) * self.instance.b, name=f"set_b_{i}")

                    n_feats = 10

                # set inst features
                for j in range(self.input_dim - n_feats):
                   grb_model.addConstr(x_withfeatures[i,j + n_feats] == (1 - x[i]) * x_inst_embedding[j], name=f"set_inst_embed_{j}_{i}")

        return x_withfeatures


    def embed_greedy(self, x, grb_model):
        """ Embed greedy value function as variables/constraints. """
        y_valuefun_greedy = grb_model.addMVar((1,), lb=-gp.GRB.INFINITY, name="y_vfg")
        yg = grb_model.addMVar((self.n,), vtype=gp.GRB.BINARY, lb=0, ub=1, name="yg")

        # yg = grb_model.addMVar((n,), vtype=gp.GRB.CONTINUOUS, lb=0, ub=1, name="yg")
        grb_model.addConstr(yg @ self.instance.a <= self.instance.b, name="greedy_budget")
        grb_model.addConstrs((yg[i] <= 1 - x[i] for i in range(self.n)), name="greedy_interdict")
        grb_model.addConstr(y_valuefun_greedy == yg @ self.instance.p, name="greedy_vf")
        
        # note: gurobi 11.0.0 is needed for this
        grb_model.addConstrs(((x[i] == 0) >> (yg[i] >= 1e-9 + self.instance.b - self.instance.a[i] - 
            gp.quicksum(self.instance.a[j]*yg[j] for j in range(i+1, self.n))) for i in range(self.n)), name="greedy_bigM")

        return y_valuefun_greedy, yg


    def get_approx_model_lower_level(self):
        """ Function for using NN to approximate lower-level value function. """
        grb_model = gp.Model()

        # initlaize variables
        x = grb_model.addMVar((self.n,), vtype=gp.GRB.BINARY, name="x")
        y = grb_model.addMVar((self.n,), vtype=gp.GRB.BINARY, name="y")

        grb_model._x = x
        grb_model._y = y

        # initialize value function estimation + slack
        y_valuefun = grb_model.addMVar((1,), lb=-gp.GRB.INFINITY, name="y_vf")

        # interdiction constraints
        # grb_model.addConstr(gp.quicksum(x) <= self.instance.k, name="interdict_budget")
        grb_model.addConstr(gp.quicksum(x) == self.instance.k, name="interdict_budget")

        # lower-level constraints
        grb_model.addConstr(y @ self.instance.a <= self.instance.b, name="lower_budget")
        grb_model.addConstrs((y[i] + x[i] <= 1  for i in range(self.n)), name="interdict")

        # objective
        if self.args.vf_constr_type == "slack":
            slack = grb_model.addMVar((1,), name="slack")
            grb_model.setObjective(y @ self.instance.p + self.args.slack_obj_coef * slack, gp.GRB.MINIMIZE)
        else:
            slack = None
            grb_model.setObjective(y @ self.instance.p, gp.GRB.MINIMIZE)

        # ml-based value function
        if self.model_type != "greedy":
            x_withfeatures = self.get_grb_features_with_x(x, grb_model)
            self.embed_net(x_withfeatures, y_valuefun, grb_model)

        # greedy value function
        if self.model_type == "greedy":
            y_valuefun_greedy, yg = self.embed_greedy(x, grb_model)
            grb_model.addConstr(y_valuefun == y_valuefun_greedy)

        # unscale value function is nn has scaled predictions
        # this needs to be done after y_valuefun is the NN output, but before the prediction 
        # constraint the value function constraint is added
        if self.model_type == "greedy" or self.label_scaler is None:
            # add constraint for value function
            self.add_value_function_constraint(grb_model, y @ self.instance.p, y_valuefun, slack)

        else:
            y_valuefun_unscaled = self.get_unscaled_valuefun(y_valuefun, grb_model)

            # add constraint for value function
            self.add_value_function_constraint(grb_model, y @ self.instance.p, y_valuefun_unscaled, slack)

        # add warmstart (if specified in args)
        self.do_warmstart(grb_model)

        return grb_model


    def get_approx_model_upper_level(self, ):
        """ Function for using NN to approximate upper-level value function. """
        grb_model = gp.Model()

        # initlaize variables
        x = grb_model.addMVar((self.n,), vtype=gp.GRB.BINARY, name="x")
        y = grb_model.addMVar((self.n,), vtype=gp.GRB.BINARY, name="y")

        grb_model._x = x
        grb_model._y = y

        # initialize value function estimation + slack
        y_valuefun = grb_model.addMVar((1,), lb=-gp.GRB.INFINITY, name="y_vf")

        # interdiction constraints
        # grb_model.addConstr(gp.quicksum(x) <= self.instance.k, name="interdict_budget")
        grb_model.addConstr(gp.quicksum(x) == self.instance.k, name="interdict_budget") # prevents bad extrapolation

        # add value function
        if self.model_type != "greedy":
            # get features and add ml-based value function
            x_withfeatures = self.get_grb_features_with_x(x, grb_model)
            self.embed_net(x_withfeatures, y_valuefun, grb_model)

        else:
            # add greedy value function 
            y_valuefun_greedy, yg = self.embed_greedy(x, grb_model)
            grb_model.addConstr(y_valuefun == y_valuefun_greedy)

        # Note: removed, this is not needed!
        # infact, this may have led to bad performance
        # # add constraint for value function
        # self.add_value_function_constraint(grb_model, y @ self.instance.p, y_valuefun, slack=None)

        # unscale value function is nn has scaled predictions
        if self.label_scaler is not None:
            y_valuefun_scaled = y_valuefun
            y_valuefun = self.get_unscaled_valuefun(y_valuefun_scaled, grb_model)

        # objective
        grb_model.setObjective(y_valuefun, gp.GRB.MINIMIZE)

        # add warmstart (if specified in args)
        self.do_warmstart(grb_model)

        return grb_model


    def get_instance_features(self,):
        """ Computes instances based features for problem. """
        # get instance as dict
        instance = {
                'a' : self.instance.a,
                'p' : self.instance.p,
                'b' : self.instance.b,
                'p_max' : self.instance.p_max,
                'n' : len(self.instance.I),
                'k' : self.instance.k,
            }

        # compute additional labels/features for instance
        double_greedy_obj, double_greedy_sol = self.blo.run_double_greedy(None, self.instance)

        # store all features
        raw_features = {
            'x' : np.zeros(self.n),
            'instance' : instance,
            'inst_id' : 0,
            'follower_obj' : 0,
            'follower_sol' : np.zeros(self.n),
            'double_greedy_obj' : double_greedy_obj,
            'double_greedy_sol' : double_greedy_sol,
            'greedy_obj' : 0,
            'greedy_sol' : np.zeros(self.n),
            'greedy_y_approx' : 0,
        }

        # get features as tensor
        self.args.kp_use_greedy = self.use_greedy # set param in args
        data_preprocessor = factory_dp(self.args, self.args.model_type, self.args.approx_type, self.args.problem, self.device)
        tr_dataset, val_dataset = data_preprocessor.preprocess_data([raw_features], [raw_features])
        x_feats = tr_dataset[0][0]

        # reshape
        x_feats = x_feats.reshape(1, x_feats.shape[0], x_feats.shape[1])

        return x_feats