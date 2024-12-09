import blo.params as params
from .kp import KnapsackSolver
from .clique import CliqueSolver
from .cng import CriticalNodeGameSolver

def factory_solver(args, cfg, opt_model):
    """ Factory approximator. """
    if "kp" in args.problem:
        return KnapsackSolver(args, cfg, opt_model)

    elif "cng" in args.problem:
        return CriticalNodeGameSolver(args, cfg, opt_model)

    else:
        raise ValueError("Invalid problem type!")