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
    n_vals = [18, 20, 22, 25, 28, 30]
    k_ratios = [1/4, 1/2, 3/4]

    idx_vals = {
        18 : 10,
        20 : 10,
        22 : 10,
        25 : 10,
        28 : 10,
        30 : 10,
    }
    
    prob_cmds = []
    for n in n_vals:
        for k_ratio in k_ratios:
            for i in range(1, idx_vals[n] + 1):
                k = int(np.ceil(n * k_ratio))

                # if args.model_per_size:
                prob_cmd = f"--problem kp_{n} "
                # else:
                #     prob_cmd = f"--problem kp "

                prob_cmd += f"--kp_n {n} --kp_k {k} --inst_idx {i}"

                prob_cmds += [prob_cmd]

    return prob_cmds



def get_cmds(args):
    """ Gets all commands.  """
    if "kp" in args.problem:
        prob_cmds = get_kp_cmds(args)
    # elif "cng" in args.problem:
    #     prob_cmds = get_cng_cmds(args)
    # elif "dr" in args.problem:
    #     prob_cmds = get_dr_cmds(args)
    else:
        raise Exception(f"get_{args.problem}_cmds not yet implemented.")

    # type of command to run
    if args.eval_type == "init":
        cmd_prefix = "python -m blo.scripts.08_init_zhou_learning_bilevel "
    elif args.eval_type == "run":
        cmd_prefix = "python -m blo.scripts.09_run_zhou_learning_bilevel "

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
    parser.add_argument('--eval_type', type=str, default='run', choices=['init', 'run'])
    # parser.add_argument('--model_type', type=str, default='inst_encoder')

    # # ml/greedy parameters
    # parser.add_argument('--model_per_size', type=int, default=1, choices=[0, 1], help='Use a different model for each instance size')
    # parser.add_argument('--approx_type', type=str, default='lower', choices=['lower', 'upper'], help='Type of approximation')
    # parser.add_argument('--vf_constr_type', type=str, default='slack', choices=['dampening', 'slack', 'none'], help='Type of constraint for vf')
    # parser.add_argument('--slack_obj_coef', type=float, default=1.0, help='Objective coef for slack')

    parser.add_argument('--file_name', type=str, default='table.dat')
    parser.add_argument('--start_idx', type=int, default=0)
    parser.add_argument('--use_problem_for_rng', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1234)

    parser.add_argument('--use_idx', type=int, default=1)

    args = parser.parse_args()

    main(args)