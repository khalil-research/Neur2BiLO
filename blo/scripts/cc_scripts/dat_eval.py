import argparse
import hashlib

import numpy as np


#-----------------------------------------------------------------------#
#                                                                       #
#               Compute Canada Meta script generation                   #
#                              Evaluation                               #
#                                                                       #
#-----------------------------------------------------------------------#



def get_kp_cmds(args):
    """ Get list of commands to run for knapsack problem. """
    n_vals = [18, 20, 22, 25, 28, 30, 100]
    k_ratios = [1/4, 1/2, 3/4]

    idx_vals = {
        18 : 10,
        20 : 10,
        22 : 10,
        25 : 10,
        28 : 10,
        30 : 10,
        100 : 100,
    }
    
    prob_cmds = []
    for n in n_vals:
        for k_ratio in k_ratios:
            for i in range(1, idx_vals[n] + 1):
                k = int(np.ceil(n * k_ratio))

                if args.model_per_size:
                    prob_cmd = f"--problem kp_{n} "
                else:
                    prob_cmd = f"--problem kp "

                prob_cmd += f"--kp_n {n} --kp_k {k} --inst_idx {i}"

                prob_cmds += [prob_cmd]

    return prob_cmds


def get_cng_cmds(args):
    """ Get list of commands to run for cng problem. """
    v_vals = [10, 25, 50, 100, 300, 500]
    gamma_vals = [0.0, 0.1]
    d_ratio_vals = [0.30, 0.75]
    a_ratio_vals = [0.03, 0.10, 0.30]

    epsilon_ratio = 1.25
    delta_ratio = 0.80

    n_iters = 25
    idx_vals = {
        10 : n_iters,
        25 : n_iters,
        50 : n_iters,
        100 : n_iters,
        300 : n_iters,
        500 : n_iters,
    }
    
    prob_cmds = []
    for v in v_vals:
        for gamma in gamma_vals:
            for d_ratio in d_ratio_vals:
                for a_ratio in a_ratio_vals:
                    for i in range(1, idx_vals[v] + 1):

                        if args.model_per_size:
                            prob_cmd = f"--problem cng_{v} "
                        else:
                            prob_cmd = f"--problem cng "

                        prob_cmd += f"--cng_v {v} "
                        prob_cmd += f"--cng_gamma {gamma} "
                        prob_cmd += f"--cng_epsilon_ratio {epsilon_ratio} "
                        prob_cmd += f"--cng_delta_ratio {delta_ratio} "
                        prob_cmd += f"--cng_d_ratio {d_ratio} "
                        prob_cmd += f"--cng_a_ratio {a_ratio} "
                        prob_cmd += f"--inst_idx {i} "

                        prob_cmds += [prob_cmd]

    return prob_cmds


def get_dr_cmds(args):
    """ Get list of commands to run for dr problem. """
    n = 30
    datasets = [15]
    idx_vals = list(range(1,11)) # test instances from paper
    
    prob_cmds = []

    for ds in datasets:
        for i in idx_vals:

            prob_cmd = f"--problem dr_hard_{n} "

            prob_cmd += f"--dr_n {n} "
            prob_cmd += f"--dr_dataset {ds} "
            prob_cmd += f"--inst_idx {i} "

            prob_cmds += [prob_cmd]

    return prob_cmds


def get_cmds(args):
    """ Gets all commands.  """
    if "kp" in args.problem:
        prob_cmds = get_kp_cmds(args)
    elif "cng" in args.problem:
        prob_cmds = get_cng_cmds(args)
    elif "dr" in args.problem:
        prob_cmds = get_dr_cmds(args)
    else:
        raise Exception(f"get_{args.problem}_cmds not yet implemented.")

    # type of command to run
    if args.eval_type == "ml" or args.eval_type == "greedy":
        cmd_type = "05_run_ml_blo"
    elif args.eval_type == "solver":
        cmd_type = "06_run_blo_solver"
    elif args.eval_type == "re_solver":
        cmd_type = "07_get_solver_revised_results"

    # start command prefix
    cmd_prefix = f"python -m blo.scripts.{cmd_type} "

    # ml/greedy parameters
    if args.eval_type == "ml" or args.eval_type == "greedy":
        cmd_prefix += f"--model_type {args.model_type} "
        cmd_prefix += f"--approx_type {args.approx_type} "
        cmd_prefix += f"--vf_constr_type {args.vf_constr_type} "
        cmd_prefix += f"--slack_obj_coef {args.slack_obj_coef} "

    # todo. add more stuff here later if needed. 

    # combine commands
    cmds = list(map(lambda x: cmd_prefix + x, prob_cmds))

    return cmds



def main(args):
    """ main. """
    cmds = get_cmds(args)

    # write to text file
    textfile = open(args.file_name, "w")
    for i, cmd in enumerate(cmds[:-1]):
        if args.use_idx:
            textfile.write(f"{1 + i + args.start_idx} {cmd}\n")
        else:
            textfile.write(f"{cmd}\n")

    if args.use_idx:
        textfile.write(f"{i + 2 + args.start_idx} {cmds[-1]}\n")
    else:
        textfile.write(f"{cmds[-1]}\n")

    textfile.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generates a list of configs to run for random search.')
    parser.add_argument('--problem', type=str, default='kp')
    parser.add_argument('--eval_type', type=str, default='ml', choices=['ml', 'solver', 'greedy', 're_solver'])
    parser.add_argument('--model_type', type=str, default='inst_encoder')

    # ml/greedy parameters
    parser.add_argument('--model_per_size', type=int, default=1, choices=[0, 1], help='Use a different model for each instance size')
    parser.add_argument('--approx_type', type=str, default='lower', choices=['lower', 'upper'], help='Type of approximation')
    parser.add_argument('--vf_constr_type', type=str, default='slack', choices=['dampening', 'slack', 'none'], help='Type of constraint for vf')
    parser.add_argument('--slack_obj_coef', type=float, default=1.0, help='Objective coef for slack')

    parser.add_argument('--file_name', type=str, default='table.dat')
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--use_problem_for_rng', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1234)

    parser.add_argument('--use_idx', type=int, default=1)

    args = parser.parse_args()

    main(args)