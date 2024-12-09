from pathlib import Path


def lst_to_str(lst):
    """ Convert list to a string.  """
    str_lst = list(map(lambda x: str(x), lst))
    lst_as_str = "-".join(str_lst)
    return lst_as_str


def get_path(data_path, cfg, ptype, suffix="pkl",):
    """ Gets path for knapsack problem. """
    p = Path(data_path) / "cng"
    p.mkdir(parents=True, exist_ok=True)

    p = p / f"{ptype}_" \
        f"v-{lst_to_str(cfg.v)}_" \
        f"g-{lst_to_str(cfg.gamma)}_" \
        f"et-{lst_to_str(cfg.eta)}_" \
        f"ep-{lst_to_str(cfg.epsilon_ratio)}_" \
        f"d-{lst_to_str(cfg.delta_ratio)}_" \
        f"dr-{lst_to_str(cfg.d_ratio)}_" \
        f"ar-{lst_to_str(cfg.a_ratio)}_" \
        f"nsi-{cfg.n_samples_inst}_" \
        f"nspi-{cfg.n_samples_per_inst}_" \
        f"s-{cfg.seed}" \
        f".{suffix}"

    return p

