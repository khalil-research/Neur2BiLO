import blo.params as params


def factory_dm(problem):
    cfg = getattr(params, problem)

    if "kp" in problem:
        print("Loading Knapsack data manager...")
        from .kp import KnapsackDataManager
        return KnapsackDataManager(cfg)

    elif "cng" in problem:
        print("Loading CriticalNodeGame data manager...")
        from .cng import CriticalNodeGameDataManager
        return CriticalNodeGameDataManager(cfg)

    elif "dr" in problem:
        print("Loading DonorRecipient data manager...")
        from .dr import DonorRecipientDataManager
        return DonorRecipientDataManager(cfg)

    else:
        raise ValueError("Invalid problem type!")