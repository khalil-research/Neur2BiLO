import blo.params as params


def factory_blo(problem):

    if "kp" in problem:
        print("Loading Knapsack BLO...")
        from .kp import Knapsack
        return Knapsack()

    elif "cng" in problem:
        print("Loading CNG BLO...")
        from .cng import CriticalNodeGame
        return CriticalNodeGame()

    elif "dr" in problem:
        print("Loading DonorRecipient BLO...")
        from .dr import DonorRecipient
        return DonorRecipient()

    else:
        raise ValueError("Invalid problem type!")
