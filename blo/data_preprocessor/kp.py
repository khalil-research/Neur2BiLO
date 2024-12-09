import os
import pickle as pkl
import time
from multiprocessing import Manager, Pool

import gurobipy as gp
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from blo.blo.kp import Knapsack
from blo.utils.kp import get_path
from .data_preprocessor import DataPreprocessor


class KnapsackDataPreprocessor(DataPreprocessor):


    def __init__(self, model_type, approx_type, device, use_greedy):
        """ Constructor for data preprocessor. """
        self.use_greedy = use_greedy
        self.model_type = model_type
        self.approx_type = approx_type
        self.device = device

        self.label_scaler = None


    def get_label_scalers(self, data):
        """ Gets scalers for labels. For knapsack each k (interdiction budget) is scaled separately. """
        label_per_k = {}
        for sample in data:
            k = sample['instance']['k']
            label = sample['follower_obj']

            if k in label_per_k:
                label_per_k[k].append(label)
            else:
                label_per_k[k] = [label]

        # compute min/max values for scaling
        self.label_scaler = {}
        for k, v in label_per_k.items():
            self.label_scaler[k] = (np.min(v), np.max(v))


    def get_ff_fixed_dataset(self, data):
        """ Gets fixed dataset for FeedForwardNetwork.  """     
        x, p, y = [], [], []

        for sample in data:
            f = np.multiply(1 - sample['x'], sample['instance']['p']/sample['instance']['a'])
            p_val = sample['instance']['p']
            val_greedy = sample['greedy_obj']
            val_double_greedy = sample['double_greedy_obj']
            k_val = sample['instance']['k']
            b_val = sample['instance']['b']

            feats = np.concatenate([f, p_val, [val_greedy, val_double_greedy, k_val, b_val]])
            label = sample['follower_obj']

            if self.label_scaler is not None:
                label = ((label - self.label_scaler[k_val][0]) / (self.label_scaler[k_val][1] - self.label_scaler[k_val][0])) + 1

            x.append(feats)
            p.append(p_val)
            y.append(label)

        x = self.to_tensor(np.array(x)).to(self.device)
        p = self.to_tensor(np.array(p)).to(self.device)
        y = self.to_tensor(np.array(y)).to(self.device)

        dataset = TensorDataset(x, p, y)
        
        return dataset


    def get_ff_invariant_dataset(self, data):
        """ Gets invariant dataset for FeedForwardNetwork.  """   
        x, p, y = [], [], []

        for sample in data:
            # problem/instance parameters
            n = sample['instance']['n']
            k_x = np.sum(sample['x'])
            k = sample['instance']['k']
            b = sample['instance']['b']

            # statistics of (1-x) * p
            p_x = np.multiply(1 - sample['x'], sample['instance']['p'])
            p_x_min = np.min(p_x[np.nonzero(p_x)])
            p_x_max = np.max(p_x[np.nonzero(p_x)])
            p_x_mean = np.mean(p_x[np.nonzero(p_x)])
            p_x_var = np.var(p_x[np.nonzero(p_x)]) * k_x / n # scale to be divided by n to make easier in grb_model

            # statistics of (1-x) * a
            a_x = np.multiply(1 - sample['x'], sample['instance']['a'])
            a_x_min = np.min(a_x[np.nonzero(a_x)])
            a_x_max = np.max(a_x[np.nonzero(a_x)])
            a_x_mean = np.mean(a_x[np.nonzero(a_x)])
            a_x_var = np.var(a_x[np.nonzero(a_x)]) * k_x / n # scale to be divided by n to make easier in grb_model

            # statistics of (1-x) * p/a
            pa_x = np.multiply(1 - sample['x'], sample['instance']['p']/sample['instance']['a'])
            pa_x_min = np.min(pa_x[np.nonzero(pa_x)])
            pa_x_max = np.max(pa_x[np.nonzero(pa_x)])
            pa_x_mean = np.mean(pa_x[np.nonzero(pa_x)])
            pa_x_var = np.var(pa_x[np.nonzero(pa_x)]) * k_x / n # scale to be divided by n to make easier in grb_model

            # greedy solution values
            val_greedy = sample['greedy_obj']
            val_double_greedy = sample['double_greedy_obj']

            feats = np.array([
                n, k, b,
                k_x,
                p_x_min, p_x_max, p_x_mean, p_x_var,
                a_x_min, a_x_max, a_x_mean, a_x_var,
                pa_x_min, pa_x_max, pa_x_mean, pa_x_var,
                val_greedy, val_double_greedy
            ])

            # p (objective)
            p_val = sample['instance']['p']

            # label
            label = sample['follower_obj']

            x.append(feats)
            p.append(p_val)
            y.append(label)

        x = self.to_tensor(np.array(x)).to(self.device)
        p = self.to_tensor(np.array(p)).to(self.device)
        y = self.to_tensor(np.array(y)).to(self.device)

        dataset = TensorDataset(x, p, y)
        
        return dataset


    def get_set_invariant_dataset(self, data):
        """ Gets invariant dataset for SetBasedNetwork.  """        
        features, n_decisions, p, labels = [], [], [], []

        pad_size = self.get_max_n_items(data)

        for sample in data:

            x_val = sample['x']

            f_val = sample['instance']['p'] / sample['instance']['a']
            a_val = sample['instance']['a']
            p_val = sample['instance']['p']

            sol_greedy = sample['greedy_sol']
            sol_double_greedy = sample['double_greedy_sol']

            val_greedy = sample['greedy_obj']
            val_double_greedy = sample['double_greedy_obj']

            k_val = sample['instance']['k']
            b_val = sample['instance']['b']

            # collect datapoint for each sample
            feats = []
            for i in range(sample['instance']['n']):
                
                # features from multipling by (1-x)
                item_feats = [
                    (1 - x_val[i]) * f_val[i],
                    (1 - x_val[i]) * a_val[i],
                    (1 - x_val[i]) * p_val[i],
                    (1 - x_val[i]) * sol_greedy[i],
                    (1 - x_val[i]) * sol_double_greedy[i],
                    (1 - x_val[i]) * val_greedy,
                    (1 - x_val[i]) * val_double_greedy,                 
                    (1 - x_val[i]) * k_val,
                    (1 - x_val[i]) * b_val]

                # # original features from notebook
                # item_feats = [(1-x) * f_val[i], p_val[i]]

                feats.append(item_feats)

            feats = self.pad_for_set_invariant(feats, pad_size)

            label = sample['follower_obj']

            features.append(feats)
            n_decisions.append(sample['instance']['n'])
            p.append(p_val)
            labels.append(label)

        features = self.to_tensor(np.array(features)).to(self.device)
        n_decisions = self.to_tensor(np.array(n_decisions)).to(self.device)
        p = self.to_tensor(np.array(p)).to(self.device)
        labels = self.to_tensor(np.array(labels)).to(self.device)

        # to pytorch dataset
        dataset = TensorDataset(features, n_decisions, p, labels)
        
        return dataset


    def get_inst_encoder_dataset(self, data):
        """ Gets features for instance encoder model.  """ 
        inst_features, decision_features, decisions, n_decisions, p, labels = [], [], [], [], [], []

        for sample in data:

            # decisions
            x_val = sample['x']

            # get features from sample
            f_val = sample['instance']['p'] / sample['instance']['a']
            f_val = f_val / np.max(f_val)
            a_val = sample['instance']['a']
            p_val = sample['instance']['p']

            y_greedy = sample['greedy_sol']
            val_greedy = sample['greedy_obj']

            x_double_greedy = np.zeros(sample['instance']['n'])
            x_double_greedy[sample['instance']['n'] - sample['instance']['k']:] = 1
            y_double_greedy = sample['double_greedy_sol']
            val_double_greedy = sample['double_greedy_obj']

            l1_greedy = np.sum(np.abs(x_val - x_double_greedy)) / sample['instance']['n']

            k_ratio_val = sample['instance']['k'] / sample['instance']['n']
            b_val = sample['instance']['b']

            # instance features (one for each decision)
            inst_feats = []
            decision_feats = []
            for i in range(sample['instance']['n']):
                                
                # features from multipling by (1-x)
                item_feats_no_decision = [
                    f_val[i],
                    a_val[i],
                    p_val[i],
                    x_double_greedy[i],
                    y_double_greedy[i],
                    val_double_greedy / sample['instance']['n'],
                    k_ratio_val,
                    b_val]

                if self.use_greedy:
                    item_feats_decision = [
                        x_val[i],                                       # depends on x
                        f_val[i],
                        a_val[i],
                        p_val[i],
                        y_greedy[i],                                    # depends on x (greedy)
                        val_greedy / sample['instance']['n'],           # depends on x (greedy)
                        x_double_greedy[i],
                        y_double_greedy[i],
                        val_double_greedy / sample['instance']['n'],
                        l1_greedy,                                      # depends on x
                        k_ratio_val,
                        b_val]
                else:
                    item_feats_decision = [
                        x_val[i],                                       # depends on x
                        f_val[i],
                        a_val[i],
                        p_val[i],
                        x_double_greedy[i],
                        y_double_greedy[i],
                        val_double_greedy / sample['instance']['n'],
                        l1_greedy,                                      # depends on x
                        k_ratio_val,
                        b_val]

                inst_feats.append(item_feats_no_decision)
                decision_feats.append(item_feats_decision)

            # pad features, not currently used
            # feats = self.pad_for_set_invariant(feats, pad_size)
 
            # feed forward instance encoder features
            # inst_feats = np.concatenate([f_val, a_val, p_val, [val_double_greedy]]) # temp

            label = sample['follower_obj']

            inst_features.append(inst_feats)
            decision_features.append(decision_feats)
            decisions.append(x_val)
            n_decisions.append(sample['instance']['n'])
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


    def get_max_n_items(self, data):
        """ Gets maximum number of items in knapsack over dataset.  """ 
        n_vals = list(map(lambda x: x['instance']['n'], data))
        return max(n_vals)


    def pad_for_set_invariant(self, feats, pad_size):
        """ Pads features for items to make them the same size.  Required for batching. """
        feat_dim = len(feats[0])
        n_items_to_add = pad_size - len(feats)
        
        # do not pad if full dim
        if n_items_to_add == 0:
            return feats

        # pad
        padding = [[0] * feat_dim] * n_items_to_add
        feats += padding

        return feats


