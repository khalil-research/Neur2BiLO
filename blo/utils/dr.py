from pathlib import Path


def lst_to_str(lst):
    """ Convert list to a string.  """
    str_lst = list(map(lambda x: str(x), lst))
    lst_as_str = "-".join(str_lst)
    return lst_as_str


def get_path(data_path, cfg, ptype, suffix="pkl",):
    """ Gets path for knapsack problem. """
    p = Path(data_path) / "dr"
    p.mkdir(parents=True, exist_ok=True)

    p = p / f"{ptype}_" \
        f"n-{lst_to_str(cfg.n)}_" \
        f"id-{cfg.id_data}_" \
        f"h-{cfg.hard}_" \
        f"cr-{lst_to_str(cfg.cost_range)}_" \
        f"cer-{lst_to_str(cfg.cost_ext_range)}_" \
        f"pcr-{lst_to_str(cfg.p_c_ratios)}_" \
        f"bd-{cfg.budget_donor_perc}_" \
        f"br-{cfg.budget_rec_perc}_" \
        f"nsi-{cfg.n_samples_inst}_" \
        f"nspi-{cfg.n_samples_per_inst}_" \
        f"s-{cfg.seed}" \
        f".{suffix}"

    return p

