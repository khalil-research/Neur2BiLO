from pathlib import Path


def lst_to_str(lst):
    """ Convert list to a string.  """
    str_lst = list(map(lambda x: str(x), lst))
    lst_as_str = "-".join(str_lst)
    return lst_as_str


def ratio_lst_to_str(lst):
    """ Convert list to a string.  """
    str_lst = list(map(lambda x: str(x[0]) + "d" + str(x[1]) , lst))
    lst_as_str = "-".join(str_lst)
    return lst_as_str


def get_path(data_path, cfg, ptype, suffix="pkl",):
    """ Gets path for knapsack problem. """
    p = Path(data_path) / "kp"
    p.mkdir(parents=True, exist_ok=True)

    p = p / f"{ptype}_" \
        f"n-{lst_to_str(cfg.n)}_" \
        f"kr-{lst_to_str(cfg.k_ratio)}_" \
        f"nsi-{cfg.n_samples_inst}_" \
        f"nspi-{cfg.n_samples_per_inst}_" \
        f"s-{cfg.seed}" \
        f".{suffix}"

    return p

