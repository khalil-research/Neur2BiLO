import os
import time
import copy
import collections
import argparse
import numpy as np

import pickle as pkl
import matplotlib.pyplot as plt

import blo.params as blo_params
from blo.utils import factory_get_path
from blo.blo import factory_blo

# xlsx writer imports
import pandas as pd
from pandas import ExcelWriter
import xlsxwriter


#-----------------------------------------------------------------------#
#                                                                       #
#          File to write instances to xlsx for ml baseline              #
#                                                                       #
#-----------------------------------------------------------------------#


# -------------------------------------------------------#
#                   Instance/Path Info                   #
# -------------------------------------------------------#

def get_instance(cfg, args, blo):
    """ Gets instance based on args. Will likely need to be done with problem specific args.  """
    if "kp" in args.problem:

        # read instances from tang paper
        if args.kp_n in [15, 18, 20, 22, 25, 28, 30] and args.inst_idx <= 10:
            instance_scaled = blo.read_instance(cfg, args.kp_n, args.kp_k, args.inst_idx, scale=True)
            instance_unscaled = blo.read_instance(cfg, args.kp_n, args.kp_k, args.inst_idx, scale=False)

        # sample instances if not in tang paper
        else:
            instance_scaled = blo.sample_instance_tang_2016(args.kp_n, args.kp_k, scale=True, seed=args.inst_idx)
            instance_scaled = blo.create_knapsack_model(**instance_scaled)

            instance_unscaled = blo.sample_instance_tang_2016(args.kp_n, args.kp_k, scale=False, seed=args.inst_idx)
            instance_unscaled = blo.create_knapsack_model(**instance_unscaled)

    elif "cng" in args.problem:
        # modify cng to control sampling.  
        # todo: make this cleaner later
        cfg.v = [args.cng_v]
        cfg.gamma = [args.cng_gamma]
        cfg.epsilon_ratio = [args.cng_epsilon_ratio]
        cfg.delta_ratio = [args.cng_delta_ratio]
        cfg.d_ratio = [args.cng_d_ratio]
        cfg.a_ratio = [args.cng_a_ratio]

        # sample
        instance_scaled = blo.sample_instance_dragotto_2023(cfg, scale=True, seed=args.inst_idx)
        instance_scaled = blo.create_cng_model(instance_scaled)

        instance_unscaled = blo.sample_instance_dragotto_2023(cfg, scale=False, seed=args.inst_idx)
        instance_unscaled = blo.create_cng_model(instance_unscaled)

    if "dr" in args.problem:
        # read instances from paper
        assert(args.inst_idx <= 10)
        instance_scaled = blo.read_instance(cfg, args.dr_dataset, args.inst_idx, scale=True)
        instance_unscaled = blo.read_instance(cfg, args.dr_dataset, args.inst_idx, scale=False)

    return instance_scaled, instance_unscaled


def get_problem_str(args):
    """ Gets path to specific instances. """
    problem_str = ""
    # add problem specific parameters to str
    if "kp" in args.problem:
        problem_str += f"n-{args.kp_n}_k-{args.kp_k}_i-{args.inst_idx}"
    
    elif "cng" in args.problem:
        problem_str += f"v-{args.cng_v}_"
        problem_str += f"g-{args.cng_gamma}_"
        problem_str += f"ep-{args.cng_epsilon_ratio}_"
        problem_str += f"de-{args.cng_delta_ratio}_"
        problem_str += f"dr-{args.cng_d_ratio}_"
        problem_str += f"ar-{args.cng_a_ratio}_"
        problem_str += f"_i-{args.inst_idx}"

    elif "dr" in args.problem:
        problem_str += f"n-{args.dr_n}_"
        problem_str += f"dr-{args.dr_dataset}_"
        problem_str += f"i-{args.inst_idx}"

    else: 
        raise Exception(f"get_problem_str not implemented for {args.problem}")

    return problem_str


def get_base_fp(args, cfg):
    """ Base file path for Zhou et. al, 2024. """
    get_path = factory_get_path(args)
    fp_base = get_path(cfg.data_path, cfg, "")
    fp_base = str(fp_base).split("/")[:-1]
    fp_base = "/".join(fp_base)
    fp_base = fp_base + "/zhou_2024/"
    return fp_base



# -------------------------------------------------------#
#                         Main                           #
# -------------------------------------------------------#

def main(args):

    global cfg

    # initialize get_path
    get_path = factory_get_path(args)

    # load config and paths
    cfg = getattr(blo_params, args.problem)
    
    # initialize BLO class
    blo = factory_blo(args.problem)

    # paths for saving results/data
    problem_str = get_problem_str(args)
    
    fp_base = get_base_fp(args, cfg)

    fp_xlsx = fp_base + 'instances/inst_' + problem_str + '.xlsx'
    fp_inst = fp_base + 'instances/inst_' + problem_str + '.pkl'

    print("ML-based optimization for")

    if "kp" in args.problem:
        print(f"   problem:              {args.problem}")
        print(f"   n:                    {args.kp_n}")
        print(f"   k:                    {args.kp_k}")
        print(f"   idx:                  {args.inst_idx}")

    else:
        raise Exception(f"Bilevel write not implemented for problem: {args.problem}")

    # load instance
    _, instance_unscaled = get_instance(cfg, args, blo)

    # objective
    c = np.zeros(args.kp_n)
    d1 = instance_unscaled.p
    d2 = instance_unscaled.p

    # upper-level constraints
    A1 = np.ones((args.kp_n,1))
    B1 = np.zeros((args.kp_n,1))
    h1 = [instance_unscaled.k]
    
    # lower-level constraints
    A2 = np.zeros((args.kp_n, 1+args.kp_n))
    B2 = np.zeros((args.kp_n, 1+args.kp_n))

    # lower-level: budget constraint
    B2[:,0] = instance_unscaled.a

    # lower-level: interdiction constraints
    for i in range(args.kp_n):
        A2[i,i+1] = 1
        B2[i,i+1] = 1

    # lower-level: RHS
    h2 = [instance_unscaled.b] + args.kp_n * [1]

    # variable type
    y = [1, 0]

    # convert to appropriate dataframe
    c_df = pd.DataFrame(c).transpose()
    d1_df = pd.DataFrame(d1).transpose()
    d2_df = pd.DataFrame(d2).transpose()

    A1_df = pd.DataFrame(A1).transpose()
    B1_df = pd.DataFrame(B1).transpose()
    h1_df = pd.DataFrame(h1)

    A2_df = pd.DataFrame(A2).transpose()
    B2_df = pd.DataFrame(B2).transpose()
    h2_df = pd.DataFrame(h2)

    y_df = pd.DataFrame(y)

    # write to xlsx worksheet
    with pd.ExcelWriter(fp_xlsx) as writer:
        c_df.to_excel(writer, sheet_name='c', index=False, header=False)
        d1_df.to_excel(writer, sheet_name='d1', index=False, header=False)
        d2_df.to_excel(writer, sheet_name='d2', index=False, header=False)

        A1_df.to_excel(writer, sheet_name='A1', index=False, header=False)
        B1_df.to_excel(writer, sheet_name='B1', index=False, header=False)
        h1_df.to_excel(writer, sheet_name='h1', index=False, header=False)

        A2_df.to_excel(writer, sheet_name='A2', index=False, header=False)
        B2_df.to_excel(writer, sheet_name='B2', index=False, header=False)
        h2_df.to_excel(writer, sheet_name='h2', index=False, header=False)

        y_df.to_excel(writer, sheet_name='y', index=False, header=False)


    with open(fp_inst, 'wb') as p:
        pkl.dump(instance_unscaled, p)
        

    print(f"\nXLSX instance saved to: {fp_xlsx}")
    print(f"pkl instance saved to: {fp_inst}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluates row generation with tiny set network for knapsack problem.')

    # problem/model parameters
    parser.add_argument('--problem', type=str, default='kp', help='Type of problem.')
    parser.add_argument('--model_type', type=str, default='inst_encoder', choices=['ff_fixed', 'ff_invariant', 'set_invariant', 'greedy', 'inst_encoder'], help='Type of ML model.')

    # surrogate model parameters
    parser.add_argument('--approx_type', type=str, default='lower', choices=['lower', 'upper'], help='Type of approximation')

    # slack/dampening parameters
    parser.add_argument('--vf_constr_type', type=str, default='slack', choices=['dampening', 'slack', 'none'], help='Type of constraint for vf')
    parser.add_argument('--slack_obj_coef', type=float, default=1.0, help='Objective coef for slack')
    parser.add_argument('--dampening_lb', type=float, default=0.1, help='Objective coef for slack')
    parser.add_argument('--dampening_ub', type=float, default=0.3, help='Objective coef for slack')

    # KP specific parameters
    parser.add_argument('--kp_n', type=int, default=20, help='Number of items.')
    parser.add_argument('--kp_k', type=int, default=15, help='Interdiction budget type.')

    # CNG specific parameters
    parser.add_argument('--cng_v', type=int, default=10, choices=[10, 25, 50, 100, 300, 500], help='Number of nodes.')
    parser.add_argument('--cng_gamma', type=float, default=0.0, choices=[0.0, 0.1], help='Gamma.')
    parser.add_argument('--cng_epsilon_ratio', type=float, default=1.25, choices=[1.25], help='Epsilon ratio.')
    parser.add_argument('--cng_delta_ratio', type=float, default=0.80, choices=[0.80], help='Delta ratio.')
    parser.add_argument('--cng_d_ratio', type=float, default=0.30, choices=[0.30, 0.75], help='Defender budget ratio.')
    parser.add_argument('--cng_a_ratio', type=float, default=0.03, choices=[0.03, 0.10, 0.30], help='Attacker budget ratio.')

    # DR specific parameters
    parser.add_argument('--dr_n', type=int, default=30, choices=[30], help='Number of items.')
    parser.add_argument('--dr_dataset', type=float, default=15, choices=[1,14,15], help='Gamma.')

    # general problem parameters
    parser.add_argument('--inst_idx', type=int, default=1,  help='Index for all problem instance.')

    # gurobi solver paramters
    parser.add_argument('--mip_gap', type=float, default=1e-4, help='gap for solving surrogate model')
    parser.add_argument('--time_limit', type=float, default=3600, help='time for solving surroaget model')
    parser.add_argument('--inc_time', type=float, default=3600, help='termination if no new incumebnts are found')
    parser.add_argument('--mip_focus', type=int, default=0, help='MIPFocus')
    parser.add_argument('--warmstart', type=int, default=1, help='Warmstart based on greedy')

    # save gurobi model
    parser.add_argument('--save_model', type=int, default=1, help='Indicator for saving model')
    parser.add_argument('--save_log', type=int, default=1, help='Indicator for gurobi log')

    # debugging mode
    parser.add_argument('--debug', type=int, default=0, help='Debugging mode to print and compute extra information at the end.')

    # output
    parser.add_argument('--verbose', type=int, default=0, help='Verbose param for optimization model')

    args = parser.parse_args()

    main(args)

