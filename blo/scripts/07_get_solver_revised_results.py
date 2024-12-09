import time
import copy
import collections
import argparse
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt

import gurobipy as gp
from gurobi_ml import add_predictor_constr

import torch
import torch.nn as nn

import blo.params as blo_params
from blo.utils import factory_get_path
from blo.blo import factory_blo
from blo.baselines.blo_solver import factory_solver

# some imports from 05_run_ml_blo
import importlib
blo_functions = importlib.import_module('blo.scripts.05_run_ml_blo')

#-----------------------------------------------------------------------#
#                                                                       #
#           File to run bilevel optimization baseline solver            #
#                                                                       #
#-----------------------------------------------------------------------#

# -------------------------------------------------------#
#                   Instance/Path Info                   #
# -------------------------------------------------------#

def get_problem_str(cfg, args, blo):
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

    else: 
        raise Exception(f"get_problem_str not implemented for {args.problem}")

    return problem_str



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
    problem_str = get_problem_str(cfg, args, blo)
    # problem_str = blo_functions.get_problem_str(cfg, args, blo)

    fp_mps = get_path(cfg.data_path, cfg, "solver_instances/" + problem_str, suffix='mps')
    fp_aux = get_path(cfg.data_path, cfg, "solver_instances/" + problem_str, suffix='aux')
    fp_raw_res = get_path(cfg.data_path, cfg, "solver_results/raw_res_" + problem_str, suffix='log')
    fp_res_original = get_path(cfg.data_path, cfg, "solver_results/res_" + problem_str, suffix='pkl')
    fp_res = get_path(cfg.data_path, cfg, "solver_results_revised/res_" + problem_str, suffix='pkl')

    print("Exact bilevel optimization for")

    # load original results
    with open(fp_res_original, "rb") as p:
        res_original = pkl.load(p)

    # load instance
    # instance_scaled, instance_unscaled = get_instance(cfg, args, blo)
    instance_scaled, instance_unscaled = blo_functions.get_instance(cfg, args, blo)

    # initialize solver
    solver = factory_solver(args, cfg, instance_unscaled)

    # get results from solver output file
    print("Reading solution from:", fp_raw_res)
    leader_obj, follower_obj, leader_sol, follower_sol = solver.read_solution(fp_sol=fp_raw_res)

    # get incumbent objectives and times
    # inc_objs, inc_times = solver.get_incumbent_times(fp_sol=fp_raw_res)
    inc_objs, inc_times = solver.get_incumbent_times_revised(fp_sol=fp_raw_res)

    # verify follower solution is consistent
    blo_solve_res = blo.solve_follower(instance_unscaled, leader_sol)
    follower_obj_blo = blo_solve_res["follower_obj"]

    if np.abs(follower_obj_blo - follower_obj)/np.abs(follower_obj) > 1e-2:
        raise Exception(f"Follower objectives are not equal!\n\
               Obj from solver:    {follower_obj}\n\
               Obj from blo class: {follower_obj_blo}")

    # print some basic info
    print("Done solving surrogate model.")
    print("     Time:         ", res_original["time"])
    print("     Leader obj:   ", leader_obj)
    print("     Follower obj: ", follower_obj)
    print("     x:            ", leader_sol)
    print("     y:            ", follower_sol)

    print("     Incumebnts:")
    for j in range(len(inc_objs)):
        print(f"         {inc_objs[j]} @ {inc_times[j]} seconds")

    # save results
    results = {
        "time" : res_original["time"],
        "leader_obj" : leader_obj,
        "follower_obj" : follower_obj,
        "x" : leader_sol,
        "y" : follower_sol,
        "inc_objs" : inc_objs,   
        "inc_times" : inc_times,
        "blo_solve_res" : blo_solve_res,
    }
   
    with open(fp_res, 'wb') as p:
        pkl.dump(results, p)

    print(f"\nResults saved to: {fp_res}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluates row generation with tiny set network for knapsack problem.')

    # problem/model parameters
    parser.add_argument('--problem', type=str, default='kp', help='Type of problem.')
    parser.add_argument('--solver_dir', type=str, default='./blo/baselines/blo_solver/solver/', help='Type of problem.')

    # knapsack specific parameters
    parser.add_argument('--kp_n', type=int, default=20, help='Number of items.')
    parser.add_argument('--kp_k', type=int, default=15, help='Interdiction budget type.')
    
    # CNG specific parameters
    parser.add_argument('--cng_v', type=int, default=10, choices=[10, 25, 50, 100, 300, 500], help='Number of nodes.')
    parser.add_argument('--cng_gamma', type=float, default=0.0, choices=[0.0, 0.1], help='Gamma.')
    parser.add_argument('--cng_epsilon_ratio', type=float, default=1.25, choices=[1.25], help='Epsilon ratio.')
    parser.add_argument('--cng_delta_ratio', type=float, default=0.80, choices=[0.80], help='Delta ratio.')
    parser.add_argument('--cng_d_ratio', type=float, default=0.30, choices=[0.30, 0.75], help='Defender budget ratio.')
    parser.add_argument('--cng_a_ratio', type=float, default=0.03, choices=[0.03, 0.10, 0.30], help='Attacker budget ratio.')

    # general problem parameters
    parser.add_argument('--inst_idx', type=int, default=1,  help='Index (or seed) for instance.')

    # solver paramters
    parser.add_argument('--setting', type=int, default=4, choices=[4], help='Setting for BILO solver')
    parser.add_argument('--time_limit', type=int, default=3600, help='time for solving surroaget model')

    # output
    parser.add_argument('--debug', type=int, default=0, help='Will print solver to std out and exit (does not collect solution)')
    parser.add_argument('--verbose', type=int, default=0, help='Verbose param for optimization model')

    args = parser.parse_args()

    main(args)

