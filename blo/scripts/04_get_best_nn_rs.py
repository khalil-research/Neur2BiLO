import argparse
import pickle as pkl
import shutil

import numpy as np

import blo.params as params
from blo.utils import factory_get_path


#--------------------------------------------------------------------#
#                                                                    #
#       File to get best trained model from random search.           #
#       Should be called even if only a single model is trained.     #
#                                                                    #
#--------------------------------------------------------------------#


def get_best_model(args):
    """ Get best model from random_search/ and saves to main directory.  """

    # get paths
    cfg = getattr(params, args.problem)
    get_path = factory_get_path(args)
    
    # paths to save/store best model
    best_fp_model = get_path(cfg.data_path, cfg, ptype=f'nn_{args.model_type}_{args.approx_type}', suffix='pt')
    best_fp_res = get_path(cfg.data_path, cfg, ptype=f'nn_res_{args.model_type}_{args.approx_type}', suffix='pkl')

    # paths of all models
    fp_model = get_path(cfg.data_path, cfg, ptype=f'random_search/nn_{args.model_type}_{args.approx_type}', suffix='pt')
    fp_res = get_path(cfg.data_path, cfg, ptype=f'random_search/nn_res_{args.model_type}_{args.approx_type}', suffix='pkl')
    fp_res_prefix = str(fp_res.stem)

    # get all tr_res files
    results_paths = [str(x) for x in fp_model.parent.iterdir()]

    results_paths = [x for x in fp_model.parent.iterdir() if fp_res_prefix in str(x.stem)]
    results_paths = [x for x in results_paths if "__" in str(x.stem)]

    # loop to get model with best criterion
    best_criterion, best_fp_rs_res = np.infty, None
    print(f'Checking {len(results_paths)} model files...')
    for rp in results_paths:
        rdict = pkl.load(open(rp, 'rb'))
        if best_criterion > rdict['val_metric_min']:
            best_criterion = rdict['val_metric_min']
            best_fp_rs_res = rp

    # generate best model path assosiated with results
    best_fp_rs_model = str(best_fp_rs_res).replace("nn_res_", "nn_")
    best_fp_rs_model = best_fp_rs_model.replace(".pkl", ".pt")

    # convert to str
    best_fp_model = str(best_fp_model)
    best_fp_res = str(best_fp_res)
    best_fp_rs_model = str(best_fp_rs_model)
    best_fp_rs_res = str(best_fp_rs_res)
   
    print(f'Best model: {best_fp_rs_model}')
    print(f'Best {rdict["val_metric"]}: {best_criterion}')

    print(f"Copying model to:   ", best_fp_model)
    print(f"Copying results to: ", best_fp_res)

    # save
    shutil.copy(best_fp_rs_model, best_fp_model)
    shutil.copy(best_fp_rs_res, best_fp_res)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', type=str, default='kp')
    parser.add_argument('--model_type', type=str, default='inst_encoder', choices=['ff_fixed', 'ff_invariant', 'set_invariant', 'inst_encoder'])
    parser.add_argument('--approx_type', type=str, default='both', choices=['lower', 'upper', 'both'])

    args = parser.parse_args()

    get_best_model(args)
