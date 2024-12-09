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
from blo.data_preprocessor.cng import CriticalNodeGameDataPreprocessor
from blo.utils.cng import get_path

from .approximator import Approximator



class CriticalNodeGameApproximator(Approximator):

    def __init__(self, args, cfg, blo, net, instance):
        """ Constructor for Knapsack Aproximator.  """
        super(CriticalNodeGameApproximator, self).__init__(args, cfg, blo, net, instance)

        # knapsack specific parameters
        self.v = self.instance.v


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

        return sol


    def do_warmstart(self, grb_model):
        """ Does warmstart.  Can be initialized to do nothing for some problems. """
        pass


    def get_grb_features_with_x(self, x, grb_model):
        """ Initialize gurobi variables for problem features (i.e., input to NN).  """
        # get greedy/double greedy

        if self.model_type == "greedy":
            x_withfeatures = None

        elif self.model_type == "ff_fixed":
            pass

        elif self.model_type == "ff_invariant":
            pass
            
        elif self.model_type == "set_invariant":
            pass

        elif self.model_type == "inst_encoder":
            x_withfeatures = grb_model.addMVar((self.v, self.input_dim), vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY , name="x_wf")

            # get instance features
            instance_feats = self.get_instance_features()

            # embed instance information
            x_inst_embedding = self.net.instance_decision_embedder(instance_feats)
            x_inst_embedding = self.net.aggregate(x_inst_embedding, self.net.agg_type)
            x_inst_embedding = self.net.final_instance_embedder(x_inst_embedding)

            x_inst_embedding = x_inst_embedding.detach().cpu().numpy().reshape(-1)

            d_p_a_ratio = self.instance.d_profit_coefs / self.instance.d_budget_coefs 
            a_p_a_ratio = self.instance.a_profit_coefs / self.instance.a_budget_coefs 

            d_p_a_ratio /= np.max(d_p_a_ratio)
            a_p_a_ratio /= np.max(a_p_a_ratio)            

            for i in range(self.v):

                if self.args.approx_type == "lower":

                    # features from multipling by 
                    grb_model.addConstr(x_withfeatures[i,0] ==  x[i], name=f'set_x[{i},0]')

                    p1 = self.instance.a_profit_coefs[i] * (- self.instance.gamma) * (1 - x[i])
                    p2 = self.instance.a_profit_coefs[i] * (1 - x[i])
                    p3 = self.instance.a_profit_coefs[i] * (1 - self.instance.eta) * x[i]

                    grb_model.addConstr(x_withfeatures[i,1] ==  p1, name=f'set_p1[{i},1]')
                    grb_model.addConstr(x_withfeatures[i,2] ==  p2, name=f'set_p2[{i},2]')
                    grb_model.addConstr(x_withfeatures[i,3] ==  p3, name=f'set_p3[{i},3]')

                    grb_model.addConstr(x_withfeatures[i,4] ==  d_p_a_ratio[i], name=f'set_d_pa_ratio[{i},4]')
                    grb_model.addConstr(x_withfeatures[i,5] ==  a_p_a_ratio[i], name=f'set_a_pa_ratio[{i},5]')

                    grb_model.addConstr(x_withfeatures[i,6] ==  self.instance.d_budget_coefs[i], name=f'set_d_budget[{i},6]')
                    grb_model.addConstr(x_withfeatures[i,7] ==  self.instance.a_budget_coefs[i], name=f'set_a_budget[{i},7]')

                    grb_model.addConstr(x_withfeatures[i,8] ==  self.instance.d_profit_coefs[i], name=f'set_d_profit[{i},8]')
                    grb_model.addConstr(x_withfeatures[i,9] ==  self.instance.a_profit_coefs[i], name=f'set_a_profit[{i},9]')

                    grb_model.addConstr(x_withfeatures[i,10] ==  self.instance.gamma, name=f'set_gamma[{i},10]')
                    grb_model.addConstr(x_withfeatures[i,11] ==  self.instance.eta, name=f'set_eta[{i},11]')
                    grb_model.addConstr(x_withfeatures[i,12] ==  self.instance.epsilon, name=f'set_epsilon[{i},12]')
                    grb_model.addConstr(x_withfeatures[i,13] ==  self.instance.delta, name=f'set_delta[{i},13]')
                    grb_model.addConstr(x_withfeatures[i,14] ==  self.instance.D, name=f'set_D[{i},14]')
                    grb_model.addConstr(x_withfeatures[i,15] ==  self.instance.A, name=f'set_A[{i},15]')

                    n_feats = 16

                elif self.args.approx_type == "upper":

                    # features from multipling by 
                    grb_model.addConstr(x_withfeatures[i,0] ==  x[i], name=f'set_x[{i},0]')
     
                    p1 = self.instance.d_profit_coefs[i] * (1 - x[i])
                    p2 = self.instance.d_profit_coefs[i] * self.instance.eta * x[i]
                    p3 = self.instance.d_profit_coefs[i] * self.instance.epsilon * x[i]
                    p4 = self.instance.d_profit_coefs[i] * self.instance.delta * (1 - x[i])

                        # p_1 += self.instance.d_profit_coefs[i] * (1 - grb_model._x[i]) * (1 - y_pred[i,0])
                        # p_2 += self.instance.d_profit_coefs[i] * self.instance.eta * grb_model._x[i] * y_pred[i,0]
                        # p_3 += self.instance.d_profit_coefs[i] * self.instance.epsilon * grb_model._x[i] * (1 - y_pred[i,0])
                        # p_4 += self.instance.d_profit_coefs[i] * self.instance.delta * (1 - grb_model._x[i]) * y_pred[i,0]


                    grb_model.addConstr(x_withfeatures[i,1] ==  p1, name=f'set_p1[{i},1]')
                    grb_model.addConstr(x_withfeatures[i,2] ==  p2, name=f'set_p2[{i},2]')
                    grb_model.addConstr(x_withfeatures[i,3] ==  p3, name=f'set_p3[{i},3]')
                    grb_model.addConstr(x_withfeatures[i,4] ==  p4, name=f'set_p4[{i},4]')

                    grb_model.addConstr(x_withfeatures[i,5] ==  d_p_a_ratio[i], name=f'set_d_pa_ratio[{i},5]')
                    grb_model.addConstr(x_withfeatures[i,6] ==  a_p_a_ratio[i], name=f'set_a_pa_ratio[{i},6]')

                    grb_model.addConstr(x_withfeatures[i,7] ==  self.instance.d_budget_coefs[i], name=f'set_d_budget[{i},7]')
                    grb_model.addConstr(x_withfeatures[i,8] ==  self.instance.a_budget_coefs[i], name=f'set_a_budget[{i},8]')

                    grb_model.addConstr(x_withfeatures[i,9] ==  self.instance.d_profit_coefs[i], name=f'set_d_profit[{i},9]')
                    grb_model.addConstr(x_withfeatures[i,10] ==  self.instance.a_profit_coefs[i], name=f'set_a_profit[{i},10]')

                    grb_model.addConstr(x_withfeatures[i,11] ==  self.instance.gamma, name=f'set_gamma[{i},11]')
                    grb_model.addConstr(x_withfeatures[i,12] ==  self.instance.eta, name=f'set_eta[{i},12]')
                    grb_model.addConstr(x_withfeatures[i,13] ==  self.instance.epsilon, name=f'set_epsilon[{i},13]')
                    grb_model.addConstr(x_withfeatures[i,14] ==  self.instance.delta, name=f'set_delta[{i},14]')
                    grb_model.addConstr(x_withfeatures[i,15] ==  self.instance.D, name=f'set_D[{i},15]')
                    grb_model.addConstr(x_withfeatures[i,16] ==  self.instance.A, name=f'set_A[{i},16]')

                    n_feats = 17

                # print(self.input_dim, n_feats)
                # set inst features
                for j in range(self.input_dim - n_feats):
                   grb_model.addConstr(x_withfeatures[i,j + n_feats] == x_inst_embedding[j], name=f"set_inst_embed[{i},{j}]")
    
        return x_withfeatures


    def get_defender_obj(self, x, y):
        """ Gets objective for defender.  """
        obj_total = 0
        for i in range(self.v):
            obj = (1 - x[i]) * (1 - y[i])                       # no defend, no attack
            obj += self.instance.eta * x[i] * y[i]              # defend, attack 
            obj += self.instance.epsilon * x[i] * (1 - y[i])    # defend, no attack
            obj += self.instance.delta * (1 - x[i]) * y[i]      # no defend, attack
            obj = self.instance.d_profit_coefs[i] * obj
            obj_total += obj
        return obj_total


    def get_attacker_obj(self, x, y):
        """ Gets objective for attacker.  """
        obj_total = 0
        for i in range(self.v):
            obj = - self.instance.gamma * (1 - x[i]) * (1 - y[i])   # no defend, no attack
            obj += (1 - x[i]) * y[i]                                # no defend, attack 
            obj += (1 - self.instance.eta) * x[i] * y[i]            # defend, attack 
            obj = self.instance.a_profit_coefs[i] * obj
            obj_total += obj
        return obj_total


    def get_approx_model_lower_level(self):
        """ Function for using NN to approximate lower-level value function. """
        grb_model = gp.Model()

        # initlaize variables
        x = grb_model.addMVar((self.v,), vtype=gp.GRB.BINARY, name="x")
        y = grb_model.addMVar((self.v,), vtype=gp.GRB.BINARY, name="y")

        grb_model._x = x
        grb_model._y = y

        # initialize value function estimation
        y_valuefun = grb_model.addMVar((1,), lb=-gp.GRB.INFINITY, name="y_vf")

        # upper-level (defender) budget constraints
        grb_model.addConstr(x @ self.instance.d_budget_coefs <= self.instance.D, name="d_budget")

        # lower-level (attacker) budget constraints
        grb_model.addConstr(y @ self.instance.a_budget_coefs <= self.instance.A, name="a_budget")

        # get upper/lower objectives
        defender_obj = self.get_defender_obj(x, y)
        attacker_obj = self.get_attacker_obj(x, y)

        # objective
        if self.args.vf_constr_type == "slack":
            slack = grb_model.addMVar((1,), name="slack")
            # should be subtracting slack in this case since we are maximizing
            grb_model.setObjective(defender_obj - self.args.slack_obj_coef * slack, gp.GRB.MAXIMIZE)
        else:
            slack = None
            grb_model.setObjective(defender_obj, gp.GRB.MAXIMIZE)

        # ml-based value function
        x_withfeatures = self.get_grb_features_with_x(x, grb_model)
        self.embed_net(x_withfeatures, y_valuefun, grb_model)

        # add constraint for value function
        self.add_value_function_constraint(grb_model, attacker_obj, y_valuefun, slack)

        return grb_model


    def get_approx_model_upper_level(self, ):
        """ Function for using NN to approximate upper-level value function. """
        grb_model = gp.Model()

        # initlaize variables
        x = grb_model.addMVar((self.v,), vtype=gp.GRB.BINARY, name="x")
        grb_model._x = x

        # initialize value function estimation
        y_valuefun = grb_model.addMVar((1,), lb=-gp.GRB.INFINITY, name="y_vf")

        # upper-level (defender) budget constraints
        grb_model.addConstr(x @ self.instance.d_budget_coefs <= self.instance.D, name="d_budget")

        # ml-based value function
        x_withfeatures = self.get_grb_features_with_x(x, grb_model)
        self.embed_net(x_withfeatures, y_valuefun, grb_model)

        # set objective
        grb_model.setObjective(y_valuefun, gp.GRB.MAXIMIZE)

        return grb_model

    def get_instance_features(self,):
        """ Computes instances based features for problem. """
        # store all features
        raw_features = {
            'x' : np.zeros(self.v),
            'instance' : self.instance.inst_dict,
            'inst_id' : 0,
            'follower_obj' : 0,
            'follower_sol' : np.zeros(self.v),
            'leader_obj' : 0,
            'leader_sol' : np.zeros(self.v),
        }

        # get features as tensor
        data_preprocessor = factory_dp(self.args, self.args.model_type, self.args.approx_type, self.args.problem, self.device)
        tr_dataset, val_dataset = data_preprocessor.preprocess_data([raw_features], [raw_features])
        x_feats = tr_dataset[0][0]

        # reshape
        x_feats = x_feats.reshape(1, x_feats.shape[0], x_feats.shape[1])

        return x_feats