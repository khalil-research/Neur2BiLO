import blo.params as params


def factory_dp(args, model_type, approx_type, problem, device):
    cfg = getattr(params, problem)

    if "kp" in problem:
        print("Loading Knapsack data preprocessor...")
        from .kp import KnapsackDataPreprocessor
        return KnapsackDataPreprocessor(model_type, approx_type, device, args.kp_use_greedy)

    elif "cng" in problem:
        print("Loading CriticalNodeGame data preprocessor...")
        from .cng import CriticalNodeGameDataPreprocessor
        return CriticalNodeGameDataPreprocessor(model_type, approx_type, device)

    elif "dr" in problem:
        print("Loading DonorRecipient data preprocessor...")
        from .dr import DonorRecipientDataPreprocessor
        return DonorRecipientDataPreprocessor(model_type, approx_type, device)

    else:
        raise ValueError("Invalid problem type!")
