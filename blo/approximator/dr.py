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
from blo.data_preprocessor.dr import DonorRecipientDataPreprocessor
from blo.utils.cng import get_path

from .approximator import Approximator



class DonorRecipientApproximator(Approximator):

    def __init__(self, args, cfg, blo, net, instance):
        """ Constructor for Knapsack Aproximator.  """
        super(DonorRecipientApproximator, self).__init__(args, cfg, blo, net, instance)

        # knapsack specific parameters
        self.n = self.instance.n


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
        y0 = 0
        y_pred = []
        for name, val in zip(var_names, var_values):
            if 'x[' == name[:2]:
                x_values += [val]
            if 'y[' == name[:2]:
                y_values += [val]
            if 'y_vf[' in name:
                y_vf = val
            if 'y_vfg[' in name:
                y_vfg = val
            if 'y_pred[' in name:
                y_pred += [val]
            if 'y_vf_unscaled[' in name:
                y_vf_us = val
            if 'y0' in name:
                y0 = val

        sol = {
            'x' : x_values,
            'y' : y_values,
            'y0' : y0,
            'y_vf' : y_vf,
            'y_vfg' : y_vfg,
            'y_vf_us' : y_vf_us,
        }

        return sol


    def get_unscaled_valuefun(self, y_valuefun, grb_model):
        """ Gets unscaled value function.  """
        y_valuefun_unscaled =  grb_model.addMVar((1,), lb=-gp.GRB.INFINITY, name="y_vf_unscaled")
        grb_model.addConstr(y_valuefun == ((y_valuefun_unscaled - self.label_scaler[0]) / (self.label_scaler[1] - self.label_scaler[0])) + 1) 
        return y_valuefun_unscaled


    def do_warmstart(self, grb_model):
        """ Does warmstart.  Can be initialized to do nothing for some problems. """
        pass


    def get_grb_features_with_x(self, x, grb_model):
        """ Initialize gurobi variables for problem features (i.e., input to NN).  """
        # get greedy/double greedy
        if self.model_type == "ff_fixed":
            pass

        elif self.model_type == "ff_invariant":
            pass
            
        elif self.model_type == "set_invariant":
            pass

        elif self.model_type == "inst_encoder":
            x_withfeatures = grb_model.addMVar((self.n, self.input_dim), vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY , name="x_wf")

            # get instance features
            instance_feats = self.get_instance_features()

            # embed instance information
            x_inst_embedding = self.net.instance_decision_embedder(instance_feats)
            x_inst_embedding = self.net.aggregate(x_inst_embedding, self.net.agg_type)
            x_inst_embedding = self.net.final_instance_embedder(x_inst_embedding)

            x_inst_embedding = x_inst_embedding.detach().cpu().numpy().reshape(-1)

            # get ratios
            w_c = self.instance.w / self.instance.c 
            v_c = self.instance.v / self.instance.c 

            # normalize ratios
            w_c /= np.max(w_c)
            v_c /= np.max(v_c)

            for i in range(self.n):

                # features 
                grb_model.addConstr(x_withfeatures[i,0] ==  x[i], name=f'set_x[{i},0]')
                grb_model.addConstr(x_withfeatures[i,1] ==  w_c[i], name=f'set_w_c[{i},1]')
                grb_model.addConstr(x_withfeatures[i,2] ==  v_c[i], name=f'set_v_c[{i},2]')
                grb_model.addConstr(x_withfeatures[i,3] ==  self.instance.w[i], name=f'set_w[{i},3]')
                grb_model.addConstr(x_withfeatures[i,4] ==  self.instance.v[i], name=f'set_w[{i},4]')
                grb_model.addConstr(x_withfeatures[i,5] ==  self.instance.c[i], name=f'set_w[{i},5]')
                grb_model.addConstr(x_withfeatures[i,6] ==  self.instance.Bd, name=f'set_Bd[{i},8]')
                grb_model.addConstr(x_withfeatures[i,7] ==  self.instance.Br, name=f'set_Br[{i},9]')

                n_feats = 8
        
                # set inst features
                for j in range(self.input_dim - n_feats):
                   grb_model.addConstr(x_withfeatures[i,j + n_feats] == x_inst_embedding[j], name=f"set_inst_embed[{i},{j}]")
    
        return x_withfeatures


    def get_donor_obj(self, x, y):
        """ Gets objective for donor.  """
        w = copy.deepcopy(self.instance.w)
        adjust = False
        if adjust:
            w = w * (np.mean(self.instance.v)/np.mean(self.instance.w))
        obj = 0
        for i in range(self.n):
            obj += y[i] * w[i]
        return obj


    def get_recipient_obj(self, x, y, y0):
        """ Gets objective for attacker.  """
        obj = 0
        for i in range(self.n):
            obj += y[i] * self.instance.v[i]
        obj += y0 * self.instance.v0
        return obj


    def get_approx_model_lower_level(self):
        """ Function for using NN to approximate lower-level value function. """
        grb_model = gp.Model()

        # initlaize variables
        x = grb_model.addMVar((self.n,), vtype=gp.GRB.CONTINUOUS, lb=0, ub=1, name="x")
        y = grb_model.addMVar((self.n,), vtype=gp.GRB.BINARY, lb=0, ub=1, name="y")
        y0 = grb_model.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, ub=1, name="y0")

        grb_model._x = x
        grb_model._y = y

        # initialize value function estimation
        y_valuefun = grb_model.addMVar((1,), lb=-gp.GRB.INFINITY, name="y_vf")

        # upper-level (donor) budget constraints
        grb_model.addConstr(x @ self.instance.c <= self.instance.Bd, name="donor_budget")

        # lower-level (receipient) budget constraints
        r_constr = 0
        for i in range(self.n):
            r_constr += (self.instance.c[i] - self.instance.c[i] * x[i]) * y[i]
        r_constr += self.instance.c0 * y0
        grb_model.addConstr(r_constr <= self.instance.Br, name="receipient_budget")

        # get upper/lower objectives
        donor_obj = self.get_donor_obj(x, y)
        recipient_obj = self.get_recipient_obj(x, y, y0)

        # objective
        if self.args.vf_constr_type == "slack":
            slack = grb_model.addMVar((1,), name="slack")
            # should be subtracting slack in this case since we are maximizing
            grb_model.setObjective(donor_obj - self.args.slack_obj_coef * slack, gp.GRB.MAXIMIZE)
        else:
            slack = None
            grb_model.setObjective(donor_obj, gp.GRB.MAXIMIZE)

        # ml-based value function
        x_withfeatures = self.get_grb_features_with_x(x, grb_model)
        self.x_features = x_withfeatures
        self.embed_net(x_withfeatures, y_valuefun, grb_model)

        # add constraint for value function
        if self.label_scaler is None:
            self.add_value_function_constraint(grb_model, recipient_obj, y_valuefun, slack)
        else:
            y_valuefun_unscaled = self.get_unscaled_valuefun(y_valuefun, grb_model)
            self.add_value_function_constraint(grb_model, recipient_obj, y_valuefun_unscaled, slack)

        # self.add_value_function_constraint(grb_model, recipient_obj, y_valuefun, slack)

        return grb_model


    def get_approx_model_upper_level(self, ):
        """ Function for using NN to approximate upper-level value function. """
        grb_model = gp.Model()

        # initlaize variables
        x = grb_model.addMVar((self.n,), vtype=gp.GRB.CONTINUOUS, name="x", lb=0, ub=1)
        grb_model._x = x

        # initialize value function estimation
        y_valuefun = grb_model.addMVar((1,), lb=-gp.GRB.INFINITY, name="y_vf")

        # upper-level (donor) budget constraints
        grb_model.addConstr(x @ self.instance.c <= self.instance.Bd, name="donor_budget")

        # ml-based value function
        x_withfeatures = self.get_grb_features_with_x(x, grb_model)
        self.x_features = x_withfeatures

        self.embed_net(x_withfeatures, y_valuefun, grb_model)

        # set objective
        grb_model.setObjective(y_valuefun, gp.GRB.MAXIMIZE)

        return grb_model


    def get_instance_features(self,):
        """ Computes instances based features for problem. """
        # store all features
        raw_features = {
            'x' : np.zeros(self.n),
            'instance' : self.instance.inst_dict,
            'inst_id' : 0,
            'follower_obj' : 0,
            'follower_sol' : np.zeros(self.n),
            'leader_obj' : 0,
            'leader_sol' : np.zeros(self.n),
        }

        # get features as tensor
        data_preprocessor = factory_dp(self.args, self.args.model_type, self.args.approx_type, self.args.problem, self.device)
        tr_dataset, val_dataset = data_preprocessor.preprocess_data([raw_features], [raw_features])
        x_feats = tr_dataset[0][0]

        # reshape
        x_feats = x_feats.reshape(1, x_feats.shape[0], x_feats.shape[1])

        return x_feats