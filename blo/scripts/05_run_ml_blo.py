import os
import time
import copy
import collections
import argparse
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt

from multiprocessing import Manager, Pool

import gurobipy as gp
from gurobi_ml import add_predictor_constr

import torch
import torch.nn as nn

import blo.params as blo_params
from blo.utils import factory_get_path
from blo.blo import factory_blo
from blo.data_manager import factory_dm
from blo.approximator import factory_approximator


#-----------------------------------------------------------------------#
#                                                                       #
#                   File to run ml-based approximation                  #
#                                                                       #
#-----------------------------------------------------------------------#


# -------------------------------------------------------#
#                Debugging for Knapsack                  #
# -------------------------------------------------------#

def debug_knapsack_embedding(args, instance_scaled, approximator, blo, sol):
    """ Function to debug results from soving knapsack surrogate model.  """
    # compute all raw features for instance
    instance = {
            'a' : instance_scaled.a,
            'p' : instance_scaled.p,
            'b' : instance_scaled.b,
            'p_max' : instance_scaled.p_max,
            'n' : len(instance_scaled.I),
            'k' : instance_scaled.k,
        }

    # solve follower for fixed x
    solve_res = blo.solve_follower(instance_scaled, sol["x"])
    follower_obj = solve_res["follower_obj"]
    follower_sol = solve_res["follower_sol"]

    # run double greedy
    double_greedy_obj, double_greedy_sol = blo.run_double_greedy(None, instance_scaled)

    # run greedy
    greedy_obj, greedy_sol = blo.run_greedy(sol["x"], None, instance_scaled)

    # compute some additional statistics
    greedy_y_approx = np.abs(follower_obj - greedy_obj) / follower_obj

    # store results
    raw_features = {
        'x' : np.array(sol["x"]),
        'instance' : instance,
        'inst_id' : 0,
        'follower_obj' : follower_obj,
        'follower_sol' : follower_sol,
        'double_greedy_obj' : double_greedy_obj,
        'double_greedy_sol' : double_greedy_sol,
        'greedy_obj' : greedy_obj,
        'greedy_sol' : greedy_sol,
        'greedy_y_approx' : greedy_y_approx,
    }

    # get features as tensor
    from blo.data_preprocessor import factory_dp
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    args.kp_use_greedy = approximator.use_greedy
    data_preprocessor = factory_dp(args, args.model_type, args.approx_type, args.problem, device=device)
    tr_dataset, val_dataset = data_preprocessor.preprocess_data([raw_features], [raw_features])

    net = approximator.net
    net.to(device)

    if "ff" in args.model_type:
        features, p, labels = tr_dataset[0]
        features = features.reshape(1,-1)
        p = p.reshape(1,-1)

        pred = net(features, p)

    elif "set" in args.model_type:
        features, n_decisions, p, labels = tr_dataset[0]
        features = features.reshape(1, features.shape[0], features.shape[1])
        n_decisions = n_decisions.reshape(1,-1)
        p = p.reshape(1,-1)

        pred = net(features, p, n_decisions)

    elif "inst" in args.model_type:
        inst_features, features, x, n_decisions, p, labels = tr_dataset[0]
        inst_features = inst_features.reshape(1, inst_features.shape[0], inst_features.shape[1])
        features = features.reshape(1, features.shape[0], features.shape[1])
        x = x.reshape(1,-1)
        n_decisions = n_decisions.reshape(1,-1)
        p = p.reshape(1,-1)

        pred = net(inst_features, features, x, p, n_decisions)

    pred = pred.detach().cpu().numpy()[0]

    print("Net prediction: ", pred)

    np.set_printoptions(suppress=True)
    print("Features as input tensor:", features.cpu().numpy())

    print("\nAre these approximately equal?")
    print("  network forward pass:          ", pred)
    print("  y_pred @ p from surrogate:     ", sol["y_pred @ p"])
    print("  y_vf scaled:                   ", sol["y_vf"])

    # do the same if using a label scaler!
    label_scaler = approximator.label_scaler
    if label_scaler is not None:
        k = raw_features['instance']['k']

        # scale both predictions
        pred_unscaled = (pred - 1) * (label_scaler[k][1] - label_scaler[k][0]) + label_scaler[k][0]
        y_pred_from_sur_unscaled = (sol["y_pred @ p"] - 1) * (label_scaler[k][1] - label_scaler[k][0]) + label_scaler[k][0]

        print("\nAre these approximately equal? (unscaled outputs)")
        print("  network forward pass unscaled:        ", pred_unscaled)
        print("  y_pred @ p from surrogate unscaled:   ", y_pred_from_sur_unscaled)
        print("  y_vf unscaled:                        ", sol["y_vf_us"])

    print("\nIf yes, then we are done debugging, if not, then features are different!")





# -------------------------------------------------------#
#                   Debugging for CNG                    #
# -------------------------------------------------------#


def debug_cng_embedding(args, instance_scaled, approximator, blo, sol):
    """ Function to debug results from soving CNG surrogate model.  """
    # get raw features
    raw_features = {
        'x' : np.array(sol["x"]),
        'instance' : instance_scaled.inst_dict,
        'inst_id' : 0,
        'follower_obj' : 0,
        'follower_sol' : np.zeros(instance_scaled.v),
        'leader_obj' : 0,
        'leader_sol' : np.zeros(instance_scaled.v),
    }

    # get features as tensor
    from blo.data_preprocessor import factory_dp
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    data_preprocessor = factory_dp(args, args.model_type, args.approx_type, args.problem, device)

    tr_dataset, val_dataset = data_preprocessor.preprocess_data([raw_features], [raw_features])

    net = approximator.net
    net.to(device)

    if "inst" in args.model_type:
        inst_features, features, x, n_decisions, p, labels = tr_dataset[0]
        inst_features = inst_features.reshape(1, inst_features.shape[0], inst_features.shape[1])
        features = features.reshape(1, features.shape[0], features.shape[1])
        x = x.reshape(1,-1)
        n_decisions = n_decisions.reshape(1,-1)
        p = p.reshape(1,p.shape[0], p.shape[1])

        pred = net(inst_features, features, x, p, n_decisions)

    else:
        raise Exception("Debugging only implemetned with inst_encoder")

    pred = pred.detach().cpu().numpy()[0]

    print("Net prediction: ", pred)

    np.set_printoptions(suppress=True)
    print("Features as input tensor:", features.cpu().numpy())

    print("\nAre these approximately equal?")
    print("  network forward pass:          ", pred)
    print("  y_vf scaled:                   ", sol["y_vf"])

    print("\nIf yes, then we are done debugging, if not, then features are different!")





# -------------------------------------------------------#
#                   Debugging for DR                     #
# -------------------------------------------------------#


def debug_dr_embedding(args, instance_scaled, approximator, blo, sol):
    """ Function to debug results from soving DR surrogate model.  """
    # get raw features
    raw_features = {
        'x' : np.array(sol["x"]),
        'instance' : instance_scaled.inst_dict,
        'inst_id' : 0,
        'follower_obj' : 0,
        'follower_sol' : np.zeros(instance_scaled.n),
        'leader_obj' : 0,
        'leader_sol' : np.zeros(instance_scaled.n),
    }

    # get features as tensor
    from blo.data_preprocessor import factory_dp
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    data_preprocessor = factory_dp(args, args.model_type, args.approx_type, args.problem, device)

    tr_dataset, val_dataset = data_preprocessor.preprocess_data([raw_features], [raw_features])

    net = approximator.net
    net.to(device)

    if "inst" in args.model_type:
        inst_features, features, x, n_decisions, p, labels = tr_dataset[0]
        inst_features = inst_features.reshape(1, inst_features.shape[0], inst_features.shape[1])
        features = features.reshape(1, features.shape[0], features.shape[1])
        x = x.reshape(1,-1)
        n_decisions = n_decisions.reshape(1,-1)
        p = p.reshape(1,-1)
        pred = net(inst_features, features, x, p, n_decisions)

    else:
        raise Exception("Debugging only implemetned with inst_encoder")

    # check all the features are the same
    feats = features[0].detach().cpu().numpy()
    for i in range(feats.shape[0]):
        for j in range(feats.shape[1]):
            feat = feats[i][j]
            app = approximator.x_features[i][j].x
            assert(np.abs(feat - app) < 1e-5)

    # check instance embedding is the same
    embed_start = approximator.x_features[i].shape[0] - feats.shape[0]
    embed_end = approximator.x_features[i].shape[0]

    grb_embedding = approximator.x_features[i][embed_start:embed_end]
    grb_embedding = list(map(lambda x: x.x, grb_embedding))
    print("Instance embedding:", grb_embedding)

    pred = pred.detach().cpu().numpy()[0]

    print("Net prediction: ", pred)

    np.set_printoptions(suppress=True)
    print("Features as input tensor:", features.cpu().numpy())

    print("\nAre these approximately equal?")
    print("  network forward pass:          ", pred)
    print("  y_vf scaled:                   ", sol["y_vf"])

    print("\nIf yes, then we are done debugging, if not, then features are different!")

    # do the same if using a label scaler
    label_scaler = approximator.label_scaler
    if label_scaler is not None:

        # scale both predictions
        pred_unscaled = (pred - 1) * (label_scaler[1] - label_scaler[0]) + label_scaler[0]
        y_pred_from_sur_unscaled = (sol["y_vf"] - 1) * (label_scaler[1] - label_scaler[0]) + label_scaler[0]

        print("\nAre these approximately equal? (unscaled outputs)")
        print("  network forward pass unscaled:        ", pred_unscaled)
        print("  y_pred @ p from surrogate unscaled:   ", y_pred_from_sur_unscaled)
        print("  y_vf unscaled:                        ", sol["y_vf_us"])

    print("\nIf yes, then we are done debugging, if not, then features are different!")





def debug_dr_sampled_best_sol(args, cfg, instance, n_samples):
    """ Sample a bunch of upper-level decisions, compute leader/follower objectives.  """
    print("Comparing against sampled-based solution")

    mp_count = Manager().Value('i', 0)
    mp_time = time.time()

    data_manager = factory_dm(args.problem)

    # sample n_sample leader decisions
    X_hash = set()
    leader_decisions = []
    for i in range(n_samples):
        x = data_manager._sample_random_x(instance.inst_dict, X_hash)
        leader_decisions.append(x)

    # compute objectives instances
    dataset = []
    for i, x  in enumerate(leader_decisions):
        if (i+1) % 100 == 0:
            print(f"   iter {i+1}/{len(leader_decisions)}")
        res = data_manager._solve_lower_level_mp(x, instance.inst_dict, i, mp_time, mp_count, n_samples)
        dataset.append(res)

    leader_objs = list(map(lambda x: x['leader_obj'], dataset))
    follower_objs = list(map(lambda x: x['follower_obj'], dataset))

    best_leader_idx = np.argmax(leader_objs)

    best_leader_obj = leader_objs[best_leader_idx]
    best_follower_obj = follower_objs[best_leader_idx]

    print("     Sampled Leader Obj:     ", best_leader_obj)
    print("     Sampeld Follower Obj:   ", best_follower_obj)

    exit()


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


def get_problem_str(cfg, args, blo):
    """ Gets path to specific instances. """
    # type of model for approximation (or greedy)
    problem_str = f"m-{args.model_type}_"

    # type of approximation/parameters
    problem_str += f"a-{args.approx_type}_"
    problem_str += f"v-{args.vf_constr_type}_"

    # add slack specific 
    if args.vf_constr_type == "slack":
        problem_str += f"s-{args.slack_obj_coef}_"

    elif args.vf_constr_type == "dampening":
        problem_str += f"dlb-{args.dampening_lb}_"
        problem_str += f"dub-{args.dampening_ub}_"

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


def get_dr_baseline_solution(cfg, id_data, i):
    """ Gets baseline solution. """
    fp_sols = cfg.data_path + "dr/DR-BKP-main/Results_Detailed.xlsx"
    f = pd.read_excel(fp_sols)
    slice_data = f.loc[(f['DataSetNumber'] == id_data) & (f["InstanceNumber"] == i) & (f["Epsilon"] == 0.01)].iloc[0]
    obj, runtime = slice_data[["Unnamed: 6", "Unnamed: 13"]]
    return {"obj": obj, "runtime": runtime}

# -------------------------------------------------------#
#                         Main                           #
# -------------------------------------------------------#

def main(args):

    global cfg

    # initialize get_path
    get_path = factory_get_path(args)

    # load config and paths
    cfg = getattr(blo_params, args.problem)
    
    # load pytorch model
    if args.model_type == "greedy":
        net = None
    else:
        # first try to load the network that works for both upper/lower
        fp_nn = get_path(cfg.data_path, cfg, f"nn_{args.model_type}_both", suffix='pt')

        # if both does not exist, then load the upper/lower specific network
        if not os.path.exists(fp_nn):
            fp_nn = get_path(cfg.data_path, cfg, f"nn_{args.model_type}_{args.approx_type}", suffix='pt')

        # load network
        net = torch.load(fp_nn)

    # initialize BLO class
    blo = factory_blo(args.problem)

    # paths for saving results/data
    problem_str = get_problem_str(cfg, args, blo)
    fp_gp_model = get_path(cfg.data_path, cfg, "gp_data/model_" + problem_str, suffix='lp')
    fp_gp_log = get_path(cfg.data_path, cfg, "gp_data/log_" + problem_str, suffix='log')
    fp_res = get_path(cfg.data_path, cfg, "results/res_" + problem_str, suffix='pkl')

    print("ML-based optimization for")

    if "kp" in args.problem:
        print(f"   problem:              {args.problem}")
        print(f"   n:                    {args.kp_n}")
        print(f"   k:                    {args.kp_k}")
        print(f"   idx:                  {args.inst_idx}")

    # load instance
    instance_scaled, instance_unscaled = get_instance(cfg, args, blo)

    # initialize approximator
    approximator = factory_approximator(args, cfg, blo, net, instance_scaled)

    # build surrogate optimization model
    if args.approx_type == "lower":
        m = approximator.get_approx_model_lower_level()
    elif args.approx_type == "upper":
        m = approximator.get_approx_model_upper_level()

    # save model (optional)
    if args.save_model:
        m.write(str(fp_gp_model))
        print("Model saved to:", fp_gp_model)

    # set gurobi parameters
    m.setParam("TimeLimit", args.time_limit)
    m.setParam("MIPGap", args.mip_gap)
    m.setParam("MIPFocus", args.mip_focus)
    if args.save_log:
        m.setParam("LogFile", str(fp_gp_log))

    # optimize
    m.optimize()
    
    if m.Status == gp.GRB.INFEASIBLE and args.save_model:
        fp_inf_gp_model = get_path(cfg.data_path, cfg, "gp_data/inf_model_" + problem_str, suffix='ilp')
        m.computeIIS()
        m.write(str(fp_inf_gp_model))
        print(f"Model infeasible exiting.  Saving infeasible model to {fp_inf_gp_model}")
        exit()

    opt_time = m.RunTime

    # recover decisions/objectives
    sol = approximator.recover_sol(m)

    # solve true follower problem(s)
    res_unscaled = blo.solve_follower(instance_unscaled, sol["x"])
    res_scaled = blo.solve_follower(instance_scaled, sol["x"])

    # check that solutions agree
    if "dr" in args.problem:
        sol_scaled = np.array(res_scaled['follower_sol'][0])
        sol_unscaled = np.array(res_unscaled['follower_sol'][0])

    else:
        sol_scaled = np.array(res_scaled['follower_sol'])
        sol_unscaled = np.array(res_unscaled['follower_sol'])

    diff = np.abs(sol_scaled - sol_unscaled)
    diff_sum = np.sum(diff)
    if diff_sum > 1e-1:
        print("Solutions equal!")
        print(f"  Diff sum:      {diff_sum}")
        print(f"  Diff:          {diff}")
        print(f"  sol_scaled:    {res_scaled['follower_sol']}")
        print(f"  sol_unscaled:  {res_unscaled['follower_sol']}\n")

        # unscale scaled objective
        if "kp" in args.problem:
            f_obj_unscaled = res_scaled['follower_obj'] * instance_scaled.p_max
        elif "cng" in args.problem:
            f_obj_unscaled = res_scaled['follower_obj'] * instance_scaled.a_p_max
        elif "dr" in args.problem:
            f_obj_unscaled = res_scaled['follower_obj'] * instance_unscaled.v0

        print(f"  obj_unscaled:  {res_unscaled['follower_obj']}")
        print(f"  obj_rescaled:  {f_obj_unscaled}")
        print(f"  obj_scaled:    {res_scaled['follower_obj']}")

        # if solutions different solutions, then check objectives as well
        obj_diff = np.abs(res_unscaled['follower_obj'] - f_obj_unscaled)
        obj_gap = np.abs(res_unscaled['follower_obj'] - f_obj_unscaled) / np.abs(f_obj_unscaled)
        print(f"  obj_gap:       {obj_gap}")
        assert(obj_gap < 1e-3)

    # print some basic info
    print("\nDone solving surrogate model.")
    print("     Time:           ", opt_time, "\n")

    print("  Leader:")
    print("     Obj:            ", res_unscaled['leader_obj'])
    print("     Obj scaled:     ", res_scaled['leader_obj'])
    print("     Obj surrogate:  ", m.objVal, "\n")

    print("  Follower:")
    print("     Obj:            ", res_unscaled['follower_obj'])
    print("     Obj scaled:     ", res_scaled['follower_obj'])
    print("     Valuefun Pred:  ", sol["y_vf"], "\n")

    # print baseline results for DRP
    if "dr" in args.problem:
        # checks for feasibility 
        assert(np.dot(sol["x"], instance_unscaled.c) <= instance_unscaled.Bd + 1e-5)
        x = sol["x"]
        y = list(map(lambda x: round(x), res_unscaled['follower_sol'][0]))
        y0 = res_unscaled['follower_sol'][1]
        lhs = 0
        rhs = instance_unscaled.Br
        for i in range(len(y)):
            lhs += (instance_unscaled.c[i] - instance_unscaled.c[i] * x[i]) * y[i]
        lhs += y0 * instance_unscaled.c[i]
        obj = sum([instance_unscaled.w[i] * y[i] for i in range(len(y))])
        assert(lhs <= rhs + 1e-5)

        baseline_results = get_dr_baseline_solution(cfg, args.dr_dataset, args.inst_idx)
        print("  Baseline Results:")
        print("     Baseline Obj:   ", baseline_results['obj'])
        print("     Baseline Time:  ", baseline_results['runtime'], "\n")

    print("  Solutions:")
    print("     x:              ", sol["x"])
    
    if "dr" in args.problem:
        print("     y (surrogate):  ", sol["y"])
        print("     y0 (surrogate): ", sol["y0"])
        print("     y (actual):     ", list(map(lambda x: round(x), res_unscaled['follower_sol'][0])))
        print("     y0 (actual):     ", res_unscaled['follower_sol'][1])

    else:
        print("     y (surrogate):  ", sol["y"])
        print("     y (actual):     ", list(map(lambda x: round(x), res_unscaled['follower_sol'])))

    if args.debug:
        if "kp" in args.problem:
            debug_knapsack_embedding(args, instance_scaled, approximator, blo, sol)
        elif "cng" in args.problem:
            debug_cng_embedding(args, instance_scaled, approximator, blo, sol)
        elif "dr" in args.problem:
            debug_dr_embedding(args, instance_scaled, approximator, blo, sol)
        else: 
            raise Expcetion(f"debugging not implemented for problem {args.problem}")

        # exit and do not save if debugging
        print("  exiting and not saving anything.  Set --debug 0 to save results")
        exit()

    # save results
    results = {
        "time" : opt_time,
        "leader_obj" : res_unscaled["leader_obj"],
        "follower_obj" : res_unscaled["follower_obj"],
        "x" : sol["x"],
        "sol_stats" : sol,
        "res_scaled" : res_scaled,
        "res_unscaled" : res_unscaled,
    }

    # add baselines results if DR problem
    if "dr" in args.problem:
        results["baseline_results"] = baseline_results

    with open(fp_res, 'wb') as p:
        pkl.dump(results, p)

    print(f"\nResults saved to: {fp_res}")



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

