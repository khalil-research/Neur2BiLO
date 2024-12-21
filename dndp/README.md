# Notebook for Discrete Network Design Problem (DNDP)

This self-contained notebook contains all the code to reproduce the results for the DNDP.  For DNDP, the implementation is more straightforward as a single model is trained for a fixed road network. In contrast, KP, CNP, and DRP use trained models across instances with variable parameters.  This assumption for DNDP allows simpler models to be used, i.e., feed-forward networks and gradient-boosted trees, and simpler surrogate models. his assumption for DNDP allows simpler models, i.e., feed-forward networks and gradient-boosted trees, and simpler surrogate models.  

See `dndp.ipynb` for a Jupyter notebook with the code. The experiments were run using `dndp.py` on the same computing cluster as other benchmarks, and the output is available in `dndp.log.`

## Getting Started
- `ipopt` must be included in this directory.
- `Instances/` are available at [https://github.com/davidrey123/DNDP/](https://github.com/davidrey123/DNDP/) and should be placed in this directory directly.  

## Parsing the results

dndp_results.csv has the following header columns: ['instance_name', 'budget', 'n_edges', 'method', 'obj', 'time']

Each (instance_name, budget={25,50,75}, n_edges={10,20}) triplet represents one test case.

For each test case, there is one row for each of the following methods:
- [3 methods] The MKKT solver at time limits 5, 10, 30 seconds
- [3 methods] Our method with an NN embedding in:
    ml-u mode (method = nn_u_0)
    ml-s mode with Gurobi's Piecewise-linear approxiamtion (method = nn_s_0)
    ml-s mode with Gurobi's spatial branch-and-bound (method = nn_s_1, not used, but implemented)
- [3 methods] Our method with a GBT (Gradient-Boosted Tree) embedding in:
    ml-u mode (method = gbt_u_0)
    ml-s mode with Gurobi's Piecewise-linear approxiamtion (method = gbt_s_0)
    ml-s mode with Gurobi's spatial branch-and-bound (method = gbt_s_1, not used, but implemented)

Column 'obj' is the leader objective value (lower is better).

The CSV has 540 rows = 2 problem sizes (n_edges = 10 or 20) * 3 leader budgets (25,50,75%) * 10 instances * 9 methods
