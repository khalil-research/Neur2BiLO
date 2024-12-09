from abc import ABC, abstractmethod

import time
import numpy as np
import pickle as pkl

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class DataPreprocessor(ABC):
    
    @abstractmethod
    def preprocess_data(self, tr_data, val_data):
        """ Function to preprocess training/validation data. """
        pass

    @abstractmethod
    def get_ff_fixed_dataset(self, data):
        """ Gets fixed dataset for FeedForwardNetwork.  """        
        pass

    @abstractmethod
    def get_ff_invariant_dataset(self, data):
        """ Gets invariant dataset for FeedForwardNetwork.  """        
        pass

    @abstractmethod
    def get_set_invariant_dataset(self, data):
        """ Gets invariant dataset for SetBasedNetwork.  """        
        pass


    def to_tensor(self, x):
        """ Returns input (x) as tensor. """
        return torch.from_numpy(x).float()


    def preprocess_data(self, tr_data, val_data):
        """ Function to preprocess training/validation data. """
        if self.model_type == "ff_fixed":
            tr_dataset = self.get_ff_fixed_dataset(tr_data)
            val_dataset = self.get_ff_fixed_dataset(val_data)

        elif self.model_type == "ff_invariant":
            tr_dataset = self.get_ff_invariant_dataset(tr_data)
            val_dataset = self.get_ff_invariant_dataset(val_data)

        elif self.model_type == "set_invariant":
            tr_dataset = self.get_set_invariant_dataset(tr_data)
            val_dataset = self.get_set_invariant_dataset(val_data)

        elif self.model_type == "inst_encoder":
            tr_dataset = self.get_inst_encoder_dataset(tr_data)
            val_dataset = self.get_inst_encoder_dataset(val_data)

        return tr_dataset, val_dataset 
