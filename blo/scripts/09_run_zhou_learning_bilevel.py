
import openpyxl

import pickle as pkl
import argparse

import sys

from blo.blo.kp import Knapsack

# imports from Zhou et. al, 2024
from blo.baselines.zhou_2024.Instance import *

# imports from scripts
import importlib
zhou_functions = importlib.import_module('blo.scripts.08_init_zhou_learning_bilevel')

# blo imports
import blo.params as blo_params
from blo.utils import factory_get_path
from blo.blo import factory_blo

#-----------------------------------------------------------------------#
#                                                                       #
#                   Run Zhou et. al, 2024 baselines                     #
#                                                                       #
#-----------------------------------------------------------------------#


# -------------------------------------------------------#
#                         Main                           #
# -------------------------------------------------------#

def main(args):

    global cfg

    # initialize get_path
    get_path = factory_get_path(args)

    # load config and paths
    cfg = getattr(blo_params, args.problem)

    # get path to instance
    problem_str = zhou_functions.get_problem_str(args)
    print(f"problem_str: {problem_str}")
  
    # initialize BLO class
    blo = factory_blo(args.problem)

    # base file path
    fp_base = zhou_functions.get_base_fp(args, cfg)

    # load instance
    _, blo_inst = zhou_functions.get_instance(cfg, args, blo)

    # read instance
    instance = InstanceMILP(problem_str, fp_base, args.n_iterations)

    instance.solve(args.baseline)

    res = instance.solutionHistory

    # convert results to dict
    results = {
        'x' : res[0],
        'y' : res[1],
        'obj' : res[2],
        'not_sure' : res[3],
        'time' : res[4],
        'time_sampling' : res[5],
        'time_training' : res[6],
        'time_solving' : res[7],
        'flag' : res[8],
    }

    # evaluate using BLO solver
    blo_solve_res = blo.solve_follower(blo_inst, results['x'])
    follower_obj_blo = blo_solve_res["follower_obj"]
    results['blo_solve_res'] = blo_solve_res

    # store results as .pkl file
    fp_results = f'{fp_base}results/results_MILP_{problem_str}.pkl'

    with open(fp_results, 'wb') as p:
        pkl.dump(results, p)

    print(f"Results saved to: {fp_results}")




if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluates row generation with tiny set network for knapsack problem.')

    # problem/model parameters
    parser.add_argument('--problem', type=str, default='kp_20', help='Type of problem.')
    # parser.add_argument('--solver_dir', type=str, default='./blo/baselines/blo_solver/solver/', help='Type of problem.')
    parser.add_argument('--baseline', type=str, default='ISNN', choices=['GNN', 'ISNN'], help='Type of ML baseline.')
    parser.add_argument('--n_iterations', type=int, default=1, choices=[1,2,3], help='Number of iterations for baseline')

    # knapsack specific parameters
    parser.add_argument('--kp_n', type=int, default=20, help='Number of items.')
    parser.add_argument('--kp_k', type=int, default=15, help='Interdiction budget type.')

    # general problem parameters
    parser.add_argument('--inst_idx', type=int, default=1,  help='Index (or seed) for instance.')


    # solver paramters
    # parser.add_argument('--setting', type=int, default=4, choices=[4], help='Setting for BILO solver')
    parser.add_argument('--time_limit', type=int, default=3600, help='time for solving surroaget model')

    args = parser.parse_args()

    main(args)

