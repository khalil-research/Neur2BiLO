# general
import math
import time
import copy
import argparse
import collections
import numpy as np
import pandas as pd
import pickle as pkl
from scipy import stats 

# gurobi
import gurobipy as gp
from gurobi_ml import add_predictor_constr
import matplotlib.pyplot as plt

# torch
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# sklearn
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE

# blo
import blo.params as blo_params
from blo.utils import load_problem, factory_get_path
from blo.data_preprocessor import factory_dp

from blo.models import *




#-----------------------------------------------------------------------#
#                                                                       #
#           File to train nn to approximation lower-level obj           # 
#                                                                       #
#-----------------------------------------------------------------------#


#------------------------------------------------#
#           Functions for evaluation             #
#------------------------------------------------#

def forward_pass_all_data(net, loader, model_type):
    """ Get predictions on entire dataset (as list).  Respects batch size.  """
    labels_all, preds_all = [], []

    with torch.no_grad():

        for batch_data in loader:

            if "set" in model_type:
                features, n_decisions, p, labels = batch_data
                preds = net(features, p, n_decisions)

            elif "ff" in model_type:
                features, p, labels = batch_data
                preds = net(features, p)

            elif "inst" in model_type:
                inst_features, decisions_features, x, n_decisions, p, labels = batch_data
                preds = net(inst_features, decisions_features, x, p, n_decisions)
            
            labels = labels.cpu().numpy().reshape(-1)
            preds = preds.detach().cpu().numpy().reshape(-1)
            
            labels_all += labels.tolist()
            preds_all += preds.tolist()
        
    return labels_all, preds_all




def test_model_predictions(cfg, net, loader, model_type, print_predictions=False, get_ranking=False, verbose=True):
    """ Evaluations model. """
    err = 0
    err_max_over = -1
    err_max_under = -1
    counter = 0

    labels_instance = []
    outputs_instance = []

    res_kendall_all = []

    labels_all, preds_all = forward_pass_all_data(net, loader, model_type)


    for i in range(len(labels_all)):

        pred = preds_all[i]
        label = labels_all[i]

        mae_cur = np.abs(label - pred)
        err_cur = np.abs(label - pred) / label
        err += err_cur
        err_max_over = np.max([err_cur, err_max_over]) if pred - label > 0 else err_max_over
        err_max_under = np.max([err_cur, err_max_under]) if pred - label < 0 else err_max_under

        labels_instance += [label]
        outputs_instance += [pred]

        if get_ranking and (i % cfg.n_samples_per_inst) == 0 and i > 0:
            res_kendall = stats.kendalltau(labels_instance, outputs_instance)
            res_kendall_all += [res_kendall.statistic]

        labels_instance = []
        outputs_instance = []

        counter += 1

    try:
        mae = mae_cur/counter
        mape = err/counter

    except:
        mae = mae_cur/counter
        mape = err[0,0].item()/counter

    if verbose:
        print("Mean Percentage Error =", mape)
        print("max over/underestimate \%:", err_max_over, err_max_under)
        if get_ranking:
            print("Kendall:", np.min(res_kendall_all), np.median(res_kendall_all), np.max(res_kendall_all))

    res = {
        "mae" : mae,
        "mape" : mape,
        "err_max_over" : err_max_over,
        "err_max_under" : err_max_under,
        "kendall_all" : res_kendall_all,
        "kendall_min" : np.min(res_kendall_all),
        "kendall_max" : np.max(res_kendall_all),
        "kendall_median" : np.median(res_kendall_all),
        "kendall_mean" : np.mean(res_kendall_all),
    }

    return res


def get_nn_param_str(args, params):
    """ Gets parameter string for use in random search. Will need to be changed as more params are added. """

    def lst_to_str(lst):
        """ Converts list of str/int to single string """
        lst_str = list(map(lambda x: str(x), lst))
        return "-".join(lst_str)

    nn_param_str = ""

    nn_param_str += f"bs-{params['batch_size']}_"
    nn_param_str += f"lr-{params['lr']}_"
    nn_param_str += f"o-{params['optimizer']}_"
    nn_param_str += f"ep-{params['n_epochs']}_"
    nn_param_str += f"do-{params['dropout']}_"

    if "ff" in args.model_type:
        nn_param_str += f"ff-h-{lst_to_str(params['ff_hidden_dim'])}_"
        nn_param_str += f"ff-ro-{params['ff_relu_output']}"

    elif "set" in args.model_type:
        nn_param_str += f"set-eh-{lst_to_str(params['set_embed_hidden_dim'])}_"
        nn_param_str += f"set-eo-{params['set_embed_output_dim']}_"
        nn_param_str += f"set-vh-{lst_to_str(params['set_value_hidden_dim'])}_"
        nn_param_str += f"set-ero-{params['set_embed_relu_output']}_"
        nn_param_str += f"set-vro-{params['set_value_relu_output']}_"
        nn_param_str += f"set-a-{params['set_agg_type']}"

    elif "inst" in args.model_type:
        nn_param_str += f"in-eh-{lst_to_str(params['inst_embed_hidden_dim'])}_"
        nn_param_str += f"in-eo-{params['inst_embed_output_dim']}_"
        nn_param_str += f"in-ph-{lst_to_str(params['inst_post_agg_hidden_dim'])}_"
        nn_param_str += f"in-po-{params['inst_post_agg_output_dim']}_"
        nn_param_str += f"in-vh-{lst_to_str(params['inst_value_hidden_dim'])}_"
        nn_param_str += f"in-ero-{params['inst_embed_relu_output']}_"
        nn_param_str += f"in-pro-{params['inst_post_agg_relu_output']}_"
        nn_param_str += f"in-vro-{params['inst_value_relu_output']}_"
        nn_param_str += f"in-a-{params['inst_agg_type']}"

    return nn_param_str



#------------------------------------------------#
#                     Main                       #
#------------------------------------------------#

def main(args):
    
    torch.manual_seed(args.seed)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print(f"Getting instance/path info for {args.problem} ...")
    
    # Get cfg, paths, functions
    cfg = getattr(blo_params, args.problem)

    # get all paths
    get_path = factory_get_path(args)
    fp_data = get_path(cfg.data_path, cfg, "ml_data")

    # load data
    print("Loading data for machine learning ... ")
    with open(fp_data, 'rb') as pf:
        dataset = pkl.load(pf)

    # preprocess data
    print("Preprocessing data  ... ")
    data_preprocessor = factory_dp(args, args.model_type, args.approx_type, args.problem, device)

    if args.scale_labels:
        print("  Scaling labels ...")
        data_preprocessor.get_label_scalers(dataset['tr_data'])

    tr_dataset, val_dataset = data_preprocessor.preprocess_data(dataset['tr_data'], dataset['val_data'])

    # create train/validation loaders
    tr_loader = DataLoader(tr_dataset, shuffle=True, batch_size=args.batch_size)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=args.batch_size_eval)

    # initializing model
    print("Initializing Model  ... ")

    if "set" in args.model_type:

        # get output dimension (may need to be done case-by-case)
        if "kp" in args.problem:
            value_output_dim = cfg.n[0]

        feat_size = tr_dataset[0][0].shape[-1]
        
        # decision embedding network
        decision_embedder = FeedForwardBase(
            input_dim = feat_size, 
            hidden_dims = args.set_embed_hidden_dim, 
            output_dim = args.set_embed_output_dim, 
            output_relu = args.set_embed_relu_output, 
            dropout = args.dropout, 
            bias = False,
            name="decision_embedder")

        # force value dimension to be 1 if not prediction second-stage coef dimension
        if not args.use_coef:
            value_output_dim = 1

        # value predictor
        value_predictor = FeedForwardBase(
            input_dim = args.set_embed_output_dim,  
            hidden_dims = args.set_value_hidden_dim, 
            output_dim = value_output_dim, 
            output_relu = args.set_value_relu_output, 
            dropout = args.dropout, 
            bias = True,
            name = "value")

        decision_embedder.to(device)
        value_predictor.to(device)

        net = SetBasedNetwork(
            decision_embedder = decision_embedder, 
            value_predictor = value_predictor, 
            agg_type = args.set_agg_type,
            use_coef = args.use_coef)

    elif "ff" in args.model_type:

        if "fixed" in args.model_type:
            assert(len(cfg.n) == 1) # ff_fixed   only works for fixed n

        input_size = len(tr_dataset[0][0])

        # get output dimension (may need to be done case-by-case)
        if "kp" in args.problem:
            value_output_dim = cfg.n[0]

        # force value dimension to be 1 if not prediction second-stage coef dimension
        if not args.use_coef:
            value_output_dim = 1
        
        # initialize ff_net
        ff_net = FeedForwardBase(
            input_dim = input_size,  
            hidden_dims = args.ff_hidden_dim, 
            output_dim = value_output_dim, 
            output_relu = args.ff_relu_output, 
            dropout = args.dropout, 
            bias = True,
            name = "ff")

        ff_net.to(device)

        net = FeedForwardNetwork(ff_net, use_coef = args.use_coef)

    elif "inst" in args.model_type:
 
        inst_feat_size = tr_dataset[0][0].shape[-1]
        decision_feat_size = tr_dataset[0][1].shape[-1]

        # instance embedding networks
        instance_decision_embedder = FeedForwardBase(
            input_dim = inst_feat_size, 
            hidden_dims = args.inst_embed_hidden_dim, 
            output_dim = args.inst_embed_output_dim, 
            output_relu = args.inst_embed_relu_output, 
            dropout = args.dropout, 
            bias = False,
            name="instance_decision_embedder")

        final_instance_embedder = FeedForwardBase(
            input_dim = args.inst_embed_output_dim, 
            hidden_dims = args.inst_post_agg_hidden_dim, 
            output_dim = args.inst_post_agg_output_dim, 
            output_relu = args.inst_post_agg_relu_output, 
            dropout = args.dropout, 
            bias = True,
            name="final_instance_embedder")

        # value predictor
        value_input_dim = decision_feat_size + args.inst_post_agg_output_dim
        value_predictor = FeedForwardBase(
            input_dim = value_input_dim,  
            hidden_dims = args.inst_value_hidden_dim, 
            output_dim = 1, 
            output_relu = args.inst_value_relu_output, 
            dropout = args.dropout, 
            bias = True,
            name = "value")

        instance_decision_embedder.to(device)
        final_instance_embedder.to(device)
        value_predictor.to(device)

        net = SetInstanceEncodingNetwork(
            instance_decision_embedder = instance_decision_embedder, 
            final_instance_embedder = final_instance_embedder, 
            value_predictor = value_predictor,
            agg_type = args.set_agg_type,
            use_coef = args.use_coef,
            problem = args.problem,
            approx_type = args.approx_type)

    else:
        raise Exception("No other model_types implemented")

    # loss
    weighted_loss = False
    criterion = nn.MSELoss()

    # optimizer
    Opt = getattr(torch.optim, args.optimizer)
    optimizer = Opt(net.parameters(), lr=args.lr)

    # scdheuler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.9, cooldown=100, verbose=True)

    # training model
    print("Training Model  ... ")

    val_mapes = []
    val_results = []

    loss_epoch = []
    # val_mape_min = math.inf
    val_metric_min = math.inf
    loss_epoch_min_idx = 0

    total_size = len(tr_loader)
    train_time = time.time()

    for epoch in range(args.n_epochs):
        
        loss_epoch += [0]
        for i, batch_data in enumerate(tr_loader, 0):

            # forward pass for invariant models
            if "set" in args.model_type:
                features, n_decisions, p, labels = batch_data
                preds = net(features, p, n_decisions)

            # forward pass for feed-forward models
            elif "ff" in args.model_type:
                features, p, labels = batch_data
                preds = net(features, p)

            elif "inst" in args.model_type:
                inst_features, decisions_features, x, n_decisions, p, labels = batch_data
                preds = net(inst_features, decisions_features, x, p, n_decisions)

            # compute loss
            loss = criterion(preds, labels)
            loss_epoch[-1] += loss.item() / len(tr_dataset)

            optimizer.zero_grad()

            # backpropagation
            loss.backward()
            optimizer.step()

        # get validation results
        val_res = test_model_predictions(
            cfg = cfg,
            net = net,
            loader = val_loader,
            model_type = args.model_type, 
            get_ranking = True, 
            verbose = False)

        # add results from iteration
        val_results.append(val_res)

        # update learning-rate scheduler
        scheduler.step(loss_epoch[-1])

        # print/update best model
        print(f'  Epoch: {epoch}: ')
        print(f'          val_mae:        {val_res["mae"]:.6f}')
        print(f'          val_mape:       {val_res["mape"]:.6f}')
        print(f'          tr_loss:        {loss_epoch[-1]:.6f}')
        print(f'          err_max_over:   {val_res["err_max_over"]:.6f}')
        print(f'          err_max_under:  {val_res["err_max_under"]:.6f}')

        val_metric = val_res[args.metric]
        loss_epoch_min_idx = epoch if val_metric < val_metric_min else loss_epoch_min_idx
        if loss_epoch_min_idx == epoch:
            print("    new best model")
            val_metric_min = val_metric
            best_model = copy.deepcopy(net) # copy.deepcopy(net.state_dict())

            # if model found within last 200 epochs, then increase # of epochs
            if args.n_epochs - epoch < 200:
                print('    doubling epochs!!!')
                args.n_epochs *= 2

        # if True and epoch >= 10 and np.abs(loss_epoch[-1]-np.mean(loss_epoch[-11:-1]))/np.mean(loss_epoch[-11:-1]) <= 1e-3:
        if True and epoch - loss_epoch_min_idx >= 200:
            print('  Early termination!', loss_epoch[-1], loss_epoch[-6])
            break

        if (epoch+1) % 10 == 0:
            print("    Epoch {}/{} Step {}/{} : Epoch {} {:.6f}, Loss {:.6f}".format(epoch+1, args.n_epochs,i+1, total_size, args.metric, val_metric, loss_epoch[-1]))
            print("    Best Epoch {} : Best {}, {:.6f}, Best Loss {:.6f}".format(loss_epoch_min_idx+1, args.metric, val_metric_min, loss_epoch[loss_epoch_min_idx]))

    print("Done training")

    best_model.eval()
    eval_res = test_model_predictions(
        cfg = cfg,
        net = best_model,
        loader = val_loader,
        model_type = args.model_type, 
        get_ranking = True,
        verbose = False)

    print(f'\n  Final model validation results: ')
    print(f'          val_mape:       {eval_res["mape"]:.6f}')
    print(f'          val_mae:       {eval_res["mae"]:.6f}')
    print(f'          err_max_over:   {eval_res["err_max_over"]:.6f}')
    print(f'          err_max_under:  {eval_res["err_max_under"]:.6f}\n')

    # collect parameters and results
    params = {
        "batch_size" : args.batch_size,
        "lr" : args.lr,
        "optimizer" : args.optimizer,
        "n_epochs" : args.n_epochs,
        "dropout" : args.dropout,
        "use_coef" : args.use_coef,
    }

    if "set" in args.model_type:
        params["set_embed_hidden_dim"] = args.set_embed_hidden_dim
        params["set_embed_output_dim"] = args.set_embed_output_dim
        params["set_value_hidden_dim"] = args.set_value_hidden_dim

        params["set_embed_relu_output"] = args.set_embed_relu_output
        params["set_value_relu_output"] = args.set_value_relu_output

        params["set_agg_type"] = args.set_agg_type

    elif "ff" in args.model_type:
        params["ff_hidden_dim"] = args.ff_hidden_dim
        params["ff_relu_output"] = args.ff_relu_output

    elif "inst" in args.model_type:
        params["inst_embed_hidden_dim"] = args.inst_embed_hidden_dim
        params["inst_embed_output_dim"] = args.inst_embed_output_dim
        params["inst_post_agg_hidden_dim"] = args.inst_post_agg_hidden_dim
        params["inst_post_agg_output_dim"] = args.inst_post_agg_output_dim
        params["inst_value_hidden_dim"] = args.inst_value_hidden_dim

        params["inst_embed_relu_output"] = args.inst_embed_relu_output
        params["inst_post_agg_relu_output"] = args.inst_post_agg_relu_output
        params["inst_value_relu_output"] = args.inst_value_relu_output

        params["inst_agg_type"] = args.inst_agg_type

    if "kp" in args.problem:
        params["kp_use_greedy"] = args.kp_use_greedy

    train_time = time.time() - train_time
    
    # collect results
    results = {
        'val_metric' : args.metric,
        'val_metric_min' : val_metric_min,
        'val_results' : val_results,
        'eval_res' : eval_res,
        'term_epoch' : epoch,
        'tr_losses' : loss_epoch,
        'params' : params,
        'train_time' : train_time,
    }
    
    # get parameter string
    param_str = get_nn_param_str(args, params)

    # save results
    fp_res = get_path(cfg.data_path, cfg, f"random_search/nn_res_{args.model_type}_{args.approx_type}")
    fp_res = str(fp_res).replace(".pkl", f"__{param_str}__.pkl")

    with open(fp_res, 'wb') as p:
        pkl.dump(results, p)

    print('  Saved training results to:', fp_res)

    # save model
    fp_net = get_path(cfg.data_path, cfg, f"random_search/nn_{args.model_type}_{args.approx_type}", suffix="pt")
    fp_net = str(fp_net).replace(".pt", f"__{param_str}__.pt")

    
    # model data
    save_data = {
        'model_type' : args.model_type,
        'use_coef' : args.use_coef,
        'err_max_over' : eval_res['err_max_over'],
        'err_max_under' : eval_res['err_max_under'],
        'label_scaler' : data_preprocessor.label_scaler,
        # 'feat_scaler' : data_preprocessor.feat_scaler, # Not implemented
        'params': params,
        'train_time' : train_time,
    }

    if "kp" in args.problem:
        save_data["kp_use_greedy"] = args.kp_use_greedy

    if "set" in args.model_type:
        save_data["decision_embedder"] = net.decision_embedder
        save_data["value_predictor"] = net.value_predictor

    elif "ff" in args.model_type:
        save_data["feedforward_net"] = net.feedforward_net

    elif "inst" in args.model_type:
        save_data["instance_decision_embedder"] = net.instance_decision_embedder
        save_data["final_instance_embedder"] = net.final_instance_embedder
        save_data["value_predictor"] = net.value_predictor

    torch.save(save_data, fp_net)

    print('  Saved model to:', fp_net)

    return
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains network for predicting lower/upper level decisions.')

    parser.add_argument('--problem', type=str, default="kp")

    # data/model type, these may not need to be separate
    parser.add_argument('--model_type', type=str, default='inst_encoder', choices=['ff_fixed', 'ff_invariant', 'set_invariant', 'inst_encoder'])

    # approximation type (lower, upper, both [for interdiction])
    parser.add_argument('--approx_type', type=str, default='both', choices=['lower', 'upper', 'both'])

    # metric to track best model over
    parser.add_argument('--metric', type=str, default='mae', choices=['mape', 'mae'])

    # Knapsack specific
    parser.add_argument('--kp_use_greedy', type=int, default=0, help='Use greedy features.')

    # Scaling arguments
    # parser.add_argument('--scale_features', type=int, default=0, help='Boolean to scale features. ') 
    parser.add_argument('--scale_labels', type=int, default=0, help='Boolean to scale labels. Must be implemented for each problem in data_preprocessor.')

    # General NN parameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--batch_size_eval', type=int, default=32, help='Batch size.')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate.')
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--n_epochs', type=int, default=1000, help='Number of training epochs.')
    parser.add_argument('--dropout', type=float, default=0, help='Dropout rate.')

    # general BLO model specific params
    parser.add_argument('--use_coef', type=int, default=1, help='Use dot product with coefficients of objectives (if 1, predicts dot prod with n-dimensional output.)')

    # FeedForwardNetwork parameters
    parser.add_argument('--ff_hidden_dim', nargs="+", type=int, default=[64], help='Hidden dimensions for feed-forward network')
    parser.add_argument('--ff_relu_output',  type=int, default=0, help='Indicator for using ReLU on output for feed-forward network.')

    # SetBasedNetwork parameters
    parser.add_argument('--set_embed_hidden_dim', type=int, nargs="+", default=[64], help='Hidden dimensions for decision embedder.')
    parser.add_argument('--set_embed_output_dim', type=int, default=16, help='Output dimension for decision embedder.')
    parser.add_argument('--set_value_hidden_dim', type=int, nargs="+", default=[64], help='Hidden dimensions for value network.')
    parser.add_argument('--set_embed_relu_output', type=int, default=0, help='Indicator for using ReLU on output of decision embedder.')
    parser.add_argument('--set_value_relu_output', type=int, default=0, help='Indicator for using ReLU on output of value network.')
    parser.add_argument('--set_agg_type', type=str, default="sum", help='Type of aggregation (sum, mean).')

    # SetInstanceEncodingNetwork parameters
    parser.add_argument('--inst_embed_hidden_dim', type=int, nargs="+", default=[128], help='Hidden dimensions for decision embedder.')
    parser.add_argument('--inst_embed_output_dim', type=int, default=64, help='Output dimension for decision embedder.')
    parser.add_argument('--inst_post_agg_hidden_dim', type=int, nargs="+", default=[128], help='Hidden dimensions for decision embedder.')
    parser.add_argument('--inst_post_agg_output_dim', type=int, default=32, help='Output dimension for decision embedder.')
    parser.add_argument('--inst_value_hidden_dim', type=int, nargs="+", default=[128], help='Hidden dimensions for value network.')

    parser.add_argument('--inst_embed_relu_output', type=int, default=0, help='Indicator for using ReLU on output of decision embedder.')
    parser.add_argument('--inst_post_agg_relu_output', type=int, default=0, help='Indicator for using ReLU on output of decision embedder.')
    parser.add_argument('--inst_value_relu_output', type=int, default=0, help='Indicator for using ReLU on output of value network.')

    parser.add_argument('--inst_agg_type', type=str, default="sum", help='Type of aggregation (sum, mean).')

    # random seed
    parser.add_argument('--seed', type=int, default=12345, help='Seed.')

    args = parser.parse_args()

    main(args)


