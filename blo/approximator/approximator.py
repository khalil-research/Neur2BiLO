from abc import ABC, abstractmethod

import copy
import torch
import numpy as np

import gurobipy as gp
from gurobi_ml import add_predictor_constr

from blo.models import *


from gurobipy import quicksum, min_


class Approximator(ABC):

    @abstractmethod
    def recover_sol(self, grb_model):
        """ Recovers solution information for gurobi model. """
        pass

    @abstractmethod
    def do_warmstart(self, grb_model):
        """ Does warmstart.  Can be initialized to do nothing for some problems. """
        pass

    @abstractmethod
    def get_approx_model_lower_level(self, ):
        """ Function for using NN to approximate lower-level value function. """
        pass

    @abstractmethod
    def get_approx_model_upper_level(self, ):
        """ Function for using NN to approximate upper-level value function. """
        pass

    @abstractmethod
    def get_grb_features_with_x(self, grb_model):
        """ Initialize gurobi variables for problem features (i.e., input to NN).  """
        pass


    def __init__(self, args, cfg, blo, net, instance):
        """ Constructor for Knapsack Aproximator.  """
        self.args = args
        self.cfg = cfg
        self.blo = blo

        # initialize instance
        self.instance = instance
        
        # initialize approximator parameters
        self.model_type = args.model_type

        # initialize nn
        self.initialize_nn(net)
        # self.device = torch.device("cuda") if torch.cuda.is_available() else 
        self.device = torch.device("cpu")


    def embed_net(self, x_withfeatures, y_valuefun, grb_model):
        """ Embeds network. """
        if "ff" in self.model_type:
            self.embed_feedforward(x_withfeatures, y_valuefun, grb_model)
        elif "set" in self.model_type:
            self.embed_setbased(x_withfeatures, y_valuefun, grb_model)
        elif "inst" in self.model_type:
            self.embed_inst_encoder(x_withfeatures, y_valuefun, grb_model)


    def add_value_function_constraint(self, grb_model, lhs, y_valuefun, slack=None):
        """ Adds estimated value function constraint. """
        if self.args.vf_constr_type == "dampening":
            dampening_factor = (1 - np.min([self.args.dampening_ub, np.max([self.args.dampening_lb, self.err_max_over])]))
            grb_model.addConstr(lhs >= dampening_factor * y_valuefun, name="lower_vf_pred")

        elif self.args.vf_constr_type == "slack":
            grb_model.addConstr(lhs >=  y_valuefun - slack, name="lower_vf_pred")

        elif self.args.vf_constr_type == "none":
            grb_model.addConstr(lhs >= y_valuefun, name="lower_vf_pred")

        else:
            raise Exception(f"vf_constr_type {vf_constr_type} is not implemented")


    def initialize_nn(self, net_):
        """ Initializes neural network for optimization.  """
        if net_ is None:
            return

        net = copy.deepcopy(net_)

        if "ff" in self.model_type:
            # initialize net
            self.net = FeedForwardNetwork( 
                feedforward_net = net["feedforward_net"].get_grb_net(), 
                use_coef = net["use_coef"])

            # dimensions
            self.input_dim = net_["feedforward_net"].input_dim
            self.y_pred_dim = net_["feedforward_net"].output_dim

        elif "set" in self.model_type:
            # initialize net
            self.net = SetBasedNetwork( 
                decision_embedder = net["decision_embedder"].get_grb_net(),
                value_predictor = net["value_predictor"].get_grb_net(),
                agg_type = net["params"]["set_agg_type"],  
                use_coef = net["use_coef"])

            # dimensions
            self.input_dim = net_["decision_embedder"].input_dim
            self.embed_dim = net_["decision_embedder"].output_dim
            self.y_pred_dim = net_["value_predictor"].output_dim

        elif "inst" in self.model_type:
            # initialize net
            self.net = SetInstanceEncodingNetwork(
                instance_decision_embedder = net["instance_decision_embedder"].get_grb_net(),
                final_instance_embedder = net["final_instance_embedder"].get_grb_net(),
                value_predictor = net["value_predictor"].get_grb_net(),
                agg_type = net["params"]["inst_agg_type"],
                use_coef = net["use_coef"],
                problem = self.args.problem,
                approx_type = self.args.approx_type,)

            # dimensions
            self.inst_embed_dim = net_["final_instance_embedder"].output_dim
            self.input_dim = net_["value_predictor"].input_dim
            self.y_pred_dim = net_["value_predictor"].output_dim # should be 1

        # error
        self.err_max_over = net_["err_max_over"]
        self.err_max_under = net_["err_max_under"]

        print("err_max_over: ", self.err_max_over)
        # use coef
        self.use_coef = net_["use_coef"]

        # get scalers for the problem
        # self.feat_scaler = net_["feat_scaler"]   # not currently implemented
        self.label_scaler = net_["label_scaler"]


    def embed_feedforward(self, x_withfeatures, y_valuefun, grb_model):
        """ Embed FeedforwardNetwork. """
        y_pred = grb_model.addMVar((self.y_pred_dim,), vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY, name="y_pred")
        pred_constr = add_predictor_constr(grb_model, self.net.feedforward_net, x_withfeatures, y_pred)

        # set value function
        if self.use_coef:
            grb_model.addConstr(y_valuefun == y_pred @ self.instance.p, name="set_vf")
        else:
            grb_model.addConstr(y_valuefun == y_pred, name="set_vf")


    def embed_setbased(self, x_withfeatures, y_valuefun, grb_model):
        """ Embed SetBasedNetwork model. """
        n_embed = x_withfeatures.shape[0]

        # embed_dim = self.net["value_predictor"].input_dim                
        embedding = grb_model.addMVar((n_embed, self.embed_dim), vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY, name="embedding")
        embedding_aggregate = grb_model.addMVar((self.embed_dim,), vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY, name="embedding_agg")

        # add predictive constraints for each model
        pred_constr = []
        for i in range(n_embed):
            pred_constr += [add_predictor_constr(grb_model, self.net.decision_embedder, x_withfeatures[i,:], embedding[i,:])]

        # aggregate embeddings
        if self.net.agg_type == "sum":
            grb_model.addConstr(embedding_aggregate == embedding.sum(axis=0))

        elif self.net.agg_type == "mean":
            grb_model.addConstr(embedding_aggregate == embedding.sum(axis=0) / (n_embed * 1.0))

        # value function prediction
        y_pred = grb_model.addMVar((self.y_pred_dim,), vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY, name="y_pred")
        pred_constr = add_predictor_constr(grb_model, self.net.value_predictor, embedding_aggregate, y_pred)

        # set value function
        if self.use_coef:
            grb_model.addConstr(y_valuefun == y_pred @ self.instance.p, name="set_vf")
        else:
            grb_model.addConstr(y_valuefun == y_pred, name="set_vf")


    def embed_inst_encoder(self, x_withfeatures, y_valuefun, grb_model):
        """ Embeds instance encoder model.  x_withfeatures should already contain precomputed instance features. """
        n_embed = x_withfeatures.shape[0]

        # value function prediction
        y_pred = grb_model.addMVar((self.y_pred_dim * n_embed,1), vtype=gp.GRB.CONTINUOUS, lb=-gp.GRB.INFINITY, name="y_pred")

        # add predictive constraints for each decision variable
        pred_constr = []
        for i in range(n_embed):
            pred_constr += [add_predictor_constr(grb_model, self.net.value_predictor, x_withfeatures[i,:], y_pred[i])]

        # set value function 
        if self.use_coef:

            if "kp" in self.args.problem:
                grb_model.addConstr(y_valuefun == y_pred[:,0] @ self.instance.p, name="set_vf")
        
            elif "cng" in self.args.problem:
                if self.args.approx_type == "lower":
                    p_1, p_2, p_3 = 0, 0, 0
                    for i in range(self.v):
                        p_1 += self.instance.a_profit_coefs[i] * (- self.instance.gamma * (1 - grb_model._x[i]) * (1 - y_pred[i,0]))
                        p_2 += self.instance.a_profit_coefs[i] * (1 - grb_model._x[i]) * y_pred[i,0]
                        p_3 += self.instance.a_profit_coefs[i] * (1 - self.instance.eta) * grb_model._x[i] * y_pred[i,0]
     
                    grb_model.addConstr(y_valuefun == p_1 + p_2 + p_3, name="set_vf")

                elif self.args.approx_type == "upper":
                    p_1, p_2, p_3, p_4 = 0, 0, 0, 0
                    for i in range(self.v):
                        p_1 += self.instance.d_profit_coefs[i] * (1 - grb_model._x[i]) * (1 - y_pred[i,0])
                        p_2 += self.instance.d_profit_coefs[i] * self.instance.eta * grb_model._x[i] * y_pred[i,0]
                        p_3 += self.instance.d_profit_coefs[i] * self.instance.epsilon * grb_model._x[i] * (1 - y_pred[i,0])
                        p_4 += self.instance.d_profit_coefs[i] * self.instance.delta * (1 - grb_model._x[i]) * y_pred[i,0]
     
                    grb_model.addConstr(y_valuefun == p_1 + p_2 + p_3 + p_4, name="set_vf")

                else:
                    raise Exception(f"approx_type={self.args.approx_type} is not implemented for {self.args.problem}")

            elif "dr" in self.args.problem:
                if self.args.approx_type == "lower":
                    # compute pred @ v + v0 * (Br - pred @ c)
                    rhs = y_pred[:,0] @ self.instance.v 
                    rhs += self.instance.v0 * (self.instance.Br - y_pred[:,0] @ self.instance.c)
                    grb_model.addConstr(y_valuefun == rhs, name="set_vf")

                elif self.args.approx_type == "upper":
                    grb_model.addConstr(y_valuefun == y_pred[:,0] @ self.instance.w, name="set_vf")

                else:
                    raise Exception(f"approx_type={self.args.approx_type} is not implemented for {self.args.problem}")

        else:
            raise Exception("Not implemented!  use_coef=0 not embedding for embed_inst_encoder")
            grb_model.addConstr(y_valuefun == y_pred, name="set_vf")


