import os
import pickle as pkl
import time
from multiprocessing import Manager, Pool

import gurobipy as gp
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from blo.blo.dr import DonorRecipient
from blo.utils.dr import get_path
from .data_preprocessor import DataPreprocessor


class DonorRecipientDataPreprocessor(DataPreprocessor):


    def __init__(self, model_type, approx_type, device):
        """ Constructor for data preprocessor. """
        self.approx_type = approx_type
        self.model_type = model_type
        self.device = device

        self.label_scaler = None


    def get_label_scalers(self, data):
        """ Gets scalers for labels. For knapsack each k (interdiction budget) is scaled separately. """
        if self.approx_type == "lower":
            labels = list(map(lambda x: x['follower_obj'], data))

        elif self.approx_type == "upper":
            labels = list(map(lambda x: x['leader_obj'], data))

        # compute min/max values for scaling
        self.label_scaler = (np.min(labels), np.max(labels))
        print("Label scaler set to:", self.label_scaler)


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
            n = sample['instance']['n']

            v0 = sample['instance']['v0']
            c0 = sample['instance']['c0']
            Bd = sample['instance']['Bd']
            Br = sample['instance']['Br']

            # features for every node
            w = sample['instance']['w']
            v = sample['instance']['v']
            c = sample['instance']['c']
            
            # ratio of profit to cost for each node
            w_c =  w / c    # donor (upper) 
            v_c =  v / c    # receipient (lower) 

            # normalize ratio
            w_c /= np.max(w_c)
            v_c /= np.max(v_c)
                
            # instance features (one for each decision)
            p_val = []
            inst_feats = []
            decision_feats = []
            for i in range(n):
                                
                # features from multipling by (1-x)
                item_feats_no_decision = [
                    # node-level ratio features
                    w_c[i],
                    v_c[i],

                    # node-level features
                    w[i],
                    v[i],
                    c[i],

                    # static features
                    Bd,
                    Br,
                ]

                # features based on decisions (in this case we only add x)
                item_feats_decision = [
                    x_val[i],
                ]
                item_feats_decision += item_feats_no_decision

                inst_feats.append(item_feats_no_decision)
                decision_feats.append(item_feats_decision)

                # cost specific lower-level features
                if self.approx_type == "upper":
                    p_val.append([w[i]])
                elif self.approx_type == "lower":
                    p_val.append([v[i], v0, c[i], Br])
                else:
                    raise Exception("DR is not interdiction, set approx_type to either 'upper' or 'lower'.")

            # feats = self.pad_for_set_invariant(feats, pad_size)

            # set label for either upper or lower approximation
            if self.approx_type == "lower":
                label = sample['follower_obj']

            elif self.approx_type == "upper":
                label = sample['leader_obj']

            else:
                raise Exception("DR is not interdiction, set approx_type to either 'upper' or 'lower'.")

            # scale labels
            if self.label_scaler is not None:
                label = (label - self.label_scaler[0]) / (self.label_scaler[1] - self.label_scaler[0]) + 1

            inst_features.append(inst_feats)
            decision_features.append(decision_feats)
            decisions.append(x_val)
            n_decisions.append(sample['instance']['n'])
            p.append(p_val)
            labels.append(label)

        print("  Label min:", np.min(labels))
        print("  Label max:", np.max(labels))

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

