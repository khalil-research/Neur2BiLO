from types import SimpleNamespace

# ---------------------#
#   Knapsack Problem   #
# ---------------------#

kp_18 = SimpleNamespace(
    # type of knapsack instances
    prob_type = "tang_2016",

    # values for knapsack instances
    n = [18],
    # k_ratio = [(3,4)], # [(1,4), (1,2), (3,4)]
    k_ratio =  [1/4, 1/2, 3/4],

    # number of samples
    n_samples_inst = 1000,
    n_samples_per_inst = 100,

    # data generation parameters
    time_limit = 30,            # for data generation only
    mip_gap = 0.01,             # for data generation only
    verbose = 0,                # for data generation only
    threads = 1,                # for data generation only
    tr_split=0.80,              # train/test split size

    # generic parameters
    seed = 7,
    data_path = './data/',

)

kp_20 = SimpleNamespace(
    # type of knapsack instances
    prob_type = "tang_2016",

    # values for knapsack instances
    n = [20],
    # k_ratio = [(3,4)], # [(1,4), (1,2), (3,4)]
    k_ratio =  [1/4, 1/2, 3/4],

    # number of samples
    n_samples_inst = 1000,
    n_samples_per_inst = 100,

    # data generation parameters
    time_limit = 30,            # for data generation only
    mip_gap = 0.01,             # for data generation only
    verbose = 0,                # for data generation only
    threads = 1,                # for data generation only
    tr_split=0.80,              # train/test split size

    # generic parameters
    seed = 7,
    data_path = './data/',

)

kp_22 = SimpleNamespace(
    # type of knapsack instances
    prob_type = "tang_2016",

    # values for knapsack instances
    n = [22],
    # k_ratio = [(3,4)], # [(1,4), (1,2), (3,4)]
    k_ratio =  [1/4, 1/2, 3/4],

    # number of samples
    n_samples_inst = 1000,
    n_samples_per_inst = 100,

    # data generation parameters
    time_limit = 30,            # for data generation only
    mip_gap = 0.01,             # for data generation only
    verbose = 0,                # for data generation only
    threads = 1,                # for data generation only
    tr_split=0.80,              # train/test split size

    # generic parameters
    seed = 7,
    data_path = './data/',

)

kp_25 = SimpleNamespace(
    # type of knapsack instances
    prob_type = "tang_2016",

    # values for knapsack instances
    n = [25],
    # k_ratio = [(3,4)], # [(1,4), (1,2), (3,4)]
    k_ratio =  [1/4, 1/2, 3/4],

    # number of samples
    n_samples_inst = 1000,
    n_samples_per_inst = 100,

    # data generation parameters
    time_limit = 30,            # for data generation only
    mip_gap = 0.01,             # for data generation only
    verbose = 0,                # for data generation only
    threads = 1,                # for data generation only
    tr_split=0.80,              # train/test split size

    # generic parameters
    seed = 7,
    data_path = './data/',

)

kp_28 = SimpleNamespace(
    # type of knapsack instances
    prob_type = "tang_2016",

    # values for knapsack instances
    n = [28],
    # k_ratio = [(3,4)], # [(1,4), (1,2), (3,4)]
    k_ratio =  [1/4, 1/2, 3/4],

    # number of samples
    n_samples_inst = 1000,
    n_samples_per_inst = 100,

    # data generation parameters
    time_limit = 30,            # for data generation only
    mip_gap = 0.01,             # for data generation only
    verbose = 0,                # for data generation only
    threads = 1,                # for data generation only
    tr_split=0.80,              # train/test split size

    # generic parameters
    seed = 7,
    data_path = './data/',

)

kp_30 = SimpleNamespace(
    # type of knapsack instances
    prob_type = "tang_2016",

    # values for knapsack instances
    n = [30],
    # k_ratio = [(3,4)], # [(1,4), (1,2), (3,4)]
    k_ratio =  [1/4, 1/2, 3/4],

    # number of samples
    n_samples_inst = 1000,
    n_samples_per_inst = 100,

    # data generation parameters
    time_limit = 30,            # for data generation only
    mip_gap = 0.01,             # for data generation only
    verbose = 0,                # for data generation only
    threads = 1,                # for data generation only
    tr_split=0.80,              # train/test split size

    # generic parameters
    seed = 7,
    data_path = './data/',

)


kp_100 = SimpleNamespace(
    # type of knapsack instances
    prob_type = "tang_2016",

    # values for knapsack instances
    n = [100],
    # k_ratio = [(3,4)], # [(1,4), (1,2), (3,4)]
    k_ratio =  [1/4, 1/2, 3/4],

    # number of samples
    n_samples_inst = 1000,
    n_samples_per_inst = 100,

    # data generation parameters
    time_limit = 30,            # for data generation only
    mip_gap = 0.01,             # for data generation only
    verbose = 0,                # for data generation only
    threads = 1,                # for data generation only
    tr_split=0.80,              # train/test split size

    # generic parameters
    seed = 7,
    data_path = './data/',

)



#---------------------------------------#
#           Critical Node Game          #
#---------------------------------------#


cng_10 = SimpleNamespace(
    prob_type = "dragotto_2023",

    # values for knapsack instances
    v = [10],                           # number of nodes

    gamma = [0.0, 0.1],                 # Attacker’s opportunity cost factor
    eta = [0.60, 0.80],                 # Defender’s mitigated-attack factor
    epsilon_ratio = [1.25],             # Defender’s mitigation-without-attack factor
    delta_ratio = [0.80],               # Attacker’s successful-attack factor
    d_ratio = [0.30, 0.75],             # Defender’s budget ratio
    a_ratio = [0.03, 0.10, 0.30],       # Attacker’s budget ratio

    budget_range = [1, 25],             # budget ranges
    profit_range = [1, 25],             # profit ranges

    # number of samples
    n_samples_inst = 1000,
    n_samples_per_inst = 100,

    # data generation parameters
    time_limit = 30,            # for data generation only
    mip_gap = 0.01,             # for data generation only
    verbose = 0,                # for data generation only
    threads = 1,                # for data generation only
    tr_split=0.80,              # train/test split size

    # generic parameters
    seed = 7,
    data_path = './data/',
)


cng_25 = SimpleNamespace(
    prob_type = "dragotto_2023",

    # values for knapsack instances
    v = [25],                           # number of nodes

    gamma = [0.0, 0.1],                 # Attacker’s opportunity cost factor
    eta = [0.60, 0.80],                 # Defender’s mitigated-attack factor
    epsilon_ratio = [1.25],             # Defender’s mitigation-without-attack factor
    delta_ratio = [0.80],               # Attacker’s successful-attack factor
    d_ratio = [0.30, 0.75],             # Defender’s budget ratio
    a_ratio = [0.03, 0.10, 0.30],       # Attacker’s budget ratio

    budget_range = [1, 25],             # budget ranges
    profit_range = [1, 25],             # profit ranges

    # number of samples
    n_samples_inst = 1000,
    n_samples_per_inst = 100,

    # data generation parameters
    time_limit = 30,            # for data generation only
    mip_gap = 0.01,             # for data generation only
    verbose = 0,                # for data generation only
    threads = 1,                # for data generation only
    tr_split=0.80,              # train/test split size

    # generic parameters
    seed = 7,
    data_path = './data/',
)


cng_50 = SimpleNamespace(
    prob_type = "dragotto_2023",

    # values for knapsack instances
    v = [50],                           # number of nodes

    gamma = [0.0, 0.1],                 # Attacker’s opportunity cost factor
    eta = [0.60, 0.80],                 # Defender’s mitigated-attack factor
    epsilon_ratio = [1.25],             # Defender’s mitigation-without-attack factor
    delta_ratio = [0.80],               # Attacker’s successful-attack factor
    d_ratio = [0.30, 0.75],             # Defender’s budget ratio
    a_ratio = [0.03, 0.10, 0.30],       # Attacker’s budget ratio

    budget_range = [1, 25],             # budget ranges
    profit_range = [1, 25],             # profit ranges

    # number of samples
    n_samples_inst = 1000,
    n_samples_per_inst = 100,

    # data generation parameters
    time_limit = 30,            # for data generation only
    mip_gap = 0.01,             # for data generation only
    verbose = 0,                # for data generation only
    threads = 1,                # for data generation only
    tr_split=0.80,              # train/test split size

    # generic parameters
    seed = 7,
    data_path = './data/',
)


cng_100 = SimpleNamespace(
    prob_type = "dragotto_2023",

    # values for knapsack instances
    v = [100],                           # number of nodes

    gamma = [0.0, 0.1],                 # Attacker’s opportunity cost factor
    eta = [0.60, 0.80],                 # Defender’s mitigated-attack factor
    epsilon_ratio = [1.25],             # Defender’s mitigation-without-attack factor
    delta_ratio = [0.80],               # Attacker’s successful-attack factor
    d_ratio = [0.30, 0.75],             # Defender’s budget ratio
    a_ratio = [0.03, 0.10, 0.30],       # Attacker’s budget ratio

    budget_range = [1, 25],             # budget ranges
    profit_range = [1, 25],             # profit ranges

    # number of samples
    n_samples_inst = 1000,
    n_samples_per_inst = 100,

    # data generation parameters
    time_limit = 30,            # for data generation only
    mip_gap = 0.01,             # for data generation only
    verbose = 0,                # for data generation only
    threads = 1,                # for data generation only
    tr_split=0.80,              # train/test split size

    # generic parameters
    seed = 7,
    data_path = './data/',
)


cng_300 = SimpleNamespace(
    prob_type = "dragotto_2023",

    # values for knapsack instances
    v = [300],                           # number of nodes

    gamma = [0.0, 0.1],                 # Attacker’s opportunity cost factor
    eta = [0.60, 0.80],                 # Defender’s mitigated-attack factor
    epsilon_ratio = [1.25],             # Defender’s mitigation-without-attack factor
    delta_ratio = [0.80],               # Attacker’s successful-attack factor
    d_ratio = [0.30, 0.75],             # Defender’s budget ratio
    a_ratio = [0.03, 0.10, 0.30],       # Attacker’s budget ratio

    budget_range = [1, 25],             # budget ranges
    profit_range = [1, 25],             # profit ranges

    # number of samples
    n_samples_inst = 1000,
    n_samples_per_inst = 100,

    # data generation parameters
    time_limit = 30,            # for data generation only
    mip_gap = 0.01,             # for data generation only
    verbose = 0,                # for data generation only
    threads = 1,                # for data generation only
    tr_split=0.80,              # train/test split size

    # generic parameters
    seed = 7,
    data_path = './data/',
)


cng_500 = SimpleNamespace(
    prob_type = "dragotto_2023",

    # values for knapsack instances
    v = [500],                           # number of nodes

    gamma = [0.0, 0.1],                 # Attacker’s opportunity cost factor
    eta = [0.60, 0.80],                 # Defender’s mitigated-attack factor
    epsilon_ratio = [1.25],             # Defender’s mitigation-without-attack factor
    delta_ratio = [0.80],               # Attacker’s successful-attack factor
    d_ratio = [0.30, 0.75],             # Defender’s budget ratio
    a_ratio = [0.03, 0.10, 0.30],       # Attacker’s budget ratio

    budget_range = [1, 25],             # budget ranges
    profit_range = [1, 25],             # profit ranges

    # number of samples
    n_samples_inst = 1000,
    n_samples_per_inst = 100,

    # data generation parameters
    time_limit = 30,            # for data generation only
    mip_gap = 0.01,             # for data generation only
    verbose = 0,                # for data generation only
    threads = 1,                # for data generation only
    tr_split=0.80,              # train/test split size

    # generic parameters
    seed = 7,
    data_path = './data/',
)



#---------------------------------------#
#           Donor Receipient            #
#---------------------------------------#

dr_30_hard = SimpleNamespace(
    id_data = 15,
    hard=1,

    # values for knapsack instances
    n = [30],                               # number of projects

    cost_range = [5000, 10000],             # cost ranges
    cost_ext_range = [1000000, 2000000],    # cost external ranges
    p_c_ratios = [1, .7, .5],               # price cost ratios of the 3 classes
    gamma_range = [1, 1],

    budget_donor_perc= 0.2,
    budget_rec_perc = 0.3,
    # number of samples
    n_samples_inst = 1000,
    n_samples_per_inst = 100,

    # data generation parameters
    time_limit = 30,            # for data generation only
    mip_gap = 0.01,             # for data generation only
    verbose = 0,                # for data generation only
    threads = 1,                # for data generation only
    tr_split=0.80,              # train/test split size

    # generic parameters
    seed = 7,
    data_path = './data/',
)

