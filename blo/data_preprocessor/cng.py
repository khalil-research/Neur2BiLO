import os
import pickle as pkl
import time
from multiprocessing import Manager, Pool

import gurobipy as gp
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from blo.blo.cng import CriticalNodeGame
from blo.utils.cng import get_path
from .data_preprocessor import DataPreprocessor


class CriticalNodeGameDataPreprocessor(DataPreprocessor):


    def __init__(self, model_type, approx_type, device):
        """ Constructor for data preprocessor. """
        self.approx_type = approx_type
        self.model_type = model_type
        self.device = device

        self.label_scaler = None


    def get_label_scalers(self, data):
        """ Gets scalers for labels. For knapsack each k (interdiction budget) is scaled separately. """
        pass

    def get_ff_fixed_dataset(self, data):
        """ Gets fixed dataset for FeedForwardNetwork.  """     
        pass

    def get_ff_invariant_dataset(self, data):
        """ Gets invariant dataset for FeedForwardNetwork.  """   
        pass

    def get_set_invariant_dataset(self, data):
        """ Gets invariant dataset for SetBasedNetwork.  """        
        pass


    def get_inst_encoder_dataset(self, data):
        """ Gets features for instance encoder model.  """ 
        inst_features, decision_features, decisions, n_decisions, p, labels = [], [], [], [], [], []

        for sample in data:

            # decisions
            x_val = sample['x']

            # static features (i.e., features for each sample)
            gamma = sample['instance']['gamma']
            eta = sample['instance']['eta']
            epsilon = sample['instance']['epsilon']
            delta = sample['instance']['delta']
            D = sample['instance']['D']
            A = sample['instance']['A']

            # features for every node
            d_budget_coefs = sample['instance']['d_budget_coefs']
            a_budget_coefs = sample['instance']['a_budget_coefs']
            d_profit_coefs = sample['instance']['d_profit_coefs']
            a_profit_coefs = sample['instance']['a_profit_coefs']
            
            # ratio of profit to cost for each node
            d_p_a_ratio  = d_profit_coefs / d_budget_coefs
            a_p_a_ratio  = a_profit_coefs / a_budget_coefs

            d_p_a_ratio /= np.max(d_p_a_ratio)
            a_p_a_ratio /= np.max(a_p_a_ratio)
            
            # cost specific lower-level features
            if self.approx_type == "lower":
                p_1 = a_profit_coefs * (- np.multiply(gamma, (1 - x_val)))      # needs to be multiplied by (1-y)
                p_2 = a_profit_coefs * (1 - x_val)                              # needs to be multiplied by y    
                p_3 = a_profit_coefs * np.multiply(1 - eta,  x_val)             # needs to be multiplied by y
                p_val = [p_1, p_2, p_3]

            # cost specific upper-level features
            else:
                p_1 = d_profit_coefs * (1 - x_val)                              # needs to be multiplied by (1-y)
                p_2 = d_profit_coefs * x_val * eta                              # needs to be multiplied by y    
                p_3 = d_profit_coefs * epsilon * x_val                          # needs to be multiplied by (1-y)
                p_4 = d_profit_coefs * delta * (1-x_val)                        # needs to be multiplied by y
                p_val = [p_1, p_2, p_3, p_4]

            # instance features (one for each decision)
            inst_feats = []
            decision_feats = []
            for i in range(sample['instance']['v']):
                                
                # features from multipling by (1-x)
                item_feats_no_decision = [
                    # node-level ratio features
                    d_p_a_ratio[i],
                    a_p_a_ratio[i],

                    # node-level features
                    d_budget_coefs[i],
                    a_budget_coefs[i],
                    d_profit_coefs[i],       
                    a_profit_coefs[i],

                    # static features
                    gamma,
                    eta,
                    epsilon,
                    delta,
                    D,
                    A,
                ]

                # todo: consider adding more later if needed
                # currently only using features not based on the decisions
                # item_feats_decision = item_feats_no_decision
                item_feats_decision = [
                    x_val[i],
                    p_1[i],
                    p_2[i],
                    p_3[i],
                ]
                if self.approx_type == "upper":
                    item_feats_decision.append(p_4[i])

                item_feats_decision += item_feats_no_decision

                inst_feats.append(item_feats_no_decision)
                decision_feats.append(item_feats_decision)

            # feats = self.pad_for_set_invariant(feats, pad_size)

            # set label for either upper or lower approximation
            if self.approx_type == "lower":
                label = sample['follower_obj']

            elif self.approx_type == "upper":
                label = sample['leader_obj']
            
            else:
                raise Exception("CNG is not interdiction, set approx_type to either 'upper' or 'lower'.")

            inst_features.append(inst_feats)
            decision_features.append(decision_feats)
            decisions.append(x_val)
            n_decisions.append(sample['instance']['v'])
            p.append(p_val)
            labels.append(label)

        # to tensors
        inst_features = self.to_tensor(np.array(inst_features)).to(self.device)
        decision_features = self.to_tensor(np.array(decision_features)).to(self.device)
        decisions = self.to_tensor(np.array(decisions)).to(self.device)
        n_decisions = self.to_tensor(np.array(n_decisions)).to(self.device)
        p = self.to_tensor(np.array(p)).to(self.device)
        labels = self.to_tensor(np.array(labels)).to(self.device)

        # to pytorch dataset
        dataset = TensorDataset(inst_features, decision_features, decisions, n_decisions, p, labels)
        
        return dataset
