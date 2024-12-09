import blo.params as params
from .kp import KnapsackApproximator
from .cng import CriticalNodeGameApproximator
from .dr import DonorRecipientApproximator


def factory_approximator(args, cfg, blo, net, instance):
    """ Factory approximator. """
    if "kp" in args.problem:
        return KnapsackApproximator(args, cfg, blo, net, instance)

    elif "cng" in args.problem:
        return CriticalNodeGameApproximator(args, cfg, blo, net, instance)

    elif "dr" in args.problem:
        return DonorRecipientApproximator(args, cfg, blo, net, instance)

    else:
        raise ValueError("Invalid problem type!")
