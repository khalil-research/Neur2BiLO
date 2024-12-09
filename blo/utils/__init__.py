import pickle

import numpy as np
import torch
import torch.nn as nn

import blo.params as params


def factory_get_path(args):
    
    if 'kp' in args.problem:
        from .kp import get_path
        return get_path
    
    elif 'cng' in args.problem:
        from .cng import get_path
        return get_path

    elif 'dr' in args.problem:
        from .dr import get_path
        return get_path

    else:
        raise Exception(f"blo.utils not defined for problem class {args.problem}")

    return get_path


def load_problem(args, cfg):
    """ Loads instance file from the cfg. """
    get_path = factory_get_path(args)
    prob_fp = get_path(cfg.data_path, cfg, "problem")
    inst = pickle.load(open(prob_fp, 'rb'))
    return inst
