import argparse
import hashlib

import numpy as np


#-----------------------------------------------------------------------#
#                                                                       #
#               Compute Canada Meta script generation                   #
#                           Random Search                               #
#                                                                       #
#-----------------------------------------------------------------------#


class ContinuousValueSampler(object):
    """ A class to sample uniformly at random in the range of [lb,ub].
        Additionally includes a probability of sampling zero if needed.  """

    def __init__(self, lb, ub, prob_zero=0.0):
        self.lb = lb
        self.ub = ub
        self.prob_zero = prob_zero

    def sample(self):
        if np.random.rand() < self.prob_zero:
            return 0
        return np.round(np.random.uniform(self.lb, self.ub), 5)


class DiscreteSampler(object):
    """ A class to sample uniformly at random in the range of [lb,ub]. """
    def __init__(self, choices):

        self.choices = choices
        self.choice_dict = {}

        for k, v in enumerate(self.choices):
            self.choice_dict[k] = v

        self.choice_list = list(self.choice_dict.keys())

    def sample(self):
        choice = np.random.choice(self.choice_list)
        return self.choice_dict[choice]



def get_nn_config(model_type):
    """ Defines params space for nn_single_cut. """
    LR_LB, LR_UB = 1e-5, 1e-1

    config = {
        # general parameters
        "batch_size": DiscreteSampler([16, 32, 64, 128]),
        "lr": ContinuousValueSampler(LR_LB, LR_UB),
        "n_epochs": DiscreteSampler([1000]),
        "dropout": ContinuousValueSampler(0.0, 0.5, prob_zero=0.5),
        "optimizer": DiscreteSampler(['Adam', 'Adagrad', 'RMSprop']),
        "use_coef": DiscreteSampler([0, 1]),
    }

    if "ff" in model_type:
        config["ff_hidden_dim"] = DiscreteSampler([[16], [32], [64], [128], [256], [512]])
        config["ff_relu_output"] = DiscreteSampler([0])

    elif "set" in model_type:
        config["set_embed_hidden_dim"] = DiscreteSampler([[16], [32], [64], [128]])
        config["set_embed_output_dim"] = DiscreteSampler([4, 8, 16, 32])
        config["set_value_hidden_dim"] =  DiscreteSampler([[16], [32], [64], [128]])
        config["set_embed_relu_output"] = DiscreteSampler([0])
        config["set_value_relu_output"] =  DiscreteSampler([0])
        config["set_agg_type"] = DiscreteSampler(["sum"])

    return config


def sample_config(problem, model_type, config):
    """ Samples a configuration for model. """
    config_cmd = f"python -m blo.scripts.03_train_nn --problem {problem} --model_type {model_type}"
    for param_name, param_sampler in config.items():
        param_val = param_sampler.sample()

        # list of args to string with spaces
        if isinstance(param_val, list):
            param_val_str = list(map(lambda x: str(x), param_val))
            param_val_str = " ".join(param_val_str)
            config_cmd += f" --{param_name} {param_val_str}"

        else:
            config_cmd += f" --{param_name} {param_val}"

    return config_cmd


def main(args):
    cmds = []

    for problem in args.problems:
        for model_type in args.model_type:
            config = get_nn_config(model_type)
            for i in range(args.n_configs):
                # uncomment if needed (likely will not be tho)
                # p_hash = int(hashlib.md5(b'{ptypes}').hexdigest(), 16)
                # np.random.seed((args.seed + i + p_hash) % (2 ** 32 - 1))
                np.random.seed((args.seed + i) % (2 ** 32 - 1))
                cmds.append(sample_config(problem, model_type, config))

    # write to text file
    textfile = open(args.file_name, "w")
    for i, cmd in enumerate(cmds[:-1]):
        if args.use_idx:
            textfile.write(f"{i + args.start_idx} {cmd}\n")
        else:
            textfile.write(f"{cmd}\n")

    if args.use_idx:
        textfile.write(f"{i + 2} {cmds[-1]}\n")
    else:
        textfile.write(f"{cmds[-1]}\n")


    textfile.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generates a list of configs to run for random search.')
    parser.add_argument('--problems', type=str, nargs='+', default=['kp'])
    parser.add_argument('--model_type', type=str, nargs='+', default=['ff_fixed', 'ff_invariant', 'set_invariant'])
    parser.add_argument('--n_configs', type=int, default=100)
    parser.add_argument('--file_name', type=str, default='table.dat')
    parser.add_argument('--start_idx', type=int, default=1)
    parser.add_argument('--use_problem_for_rng', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1234)

    parser.add_argument('--use_idx', type=int, default=1)

    args = parser.parse_args()

    main(args)