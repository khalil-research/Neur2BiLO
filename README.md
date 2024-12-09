# Neur2BiLO: Neural Bilevel Optimization


Implementation of Neur2BiLO, an efficient learning-based algorithm for mixed-integer (non-)linear bilevel optimization.  Reference below.
 - \[1\] Dumouchelle, J., Julien, E., Kurtz, J., & Khalil, E. B. Neur2BiLO: Neural Bilevel Optimization.  *The Thirty-eighth Annual Conference on Neural Information Processing Systems (NeurIPS)*, 2024. [\[Paper\]](https://openreview.net/pdf?id=T5Xb0iGCCv) [\[Website\]](https://khalil-research.github.io/Neur2BiLO/)
   

## Notes
 - All models and results have been included in the data directories.
 - To reproduce tables from the paper, use the corresponding notebook in the `notebook/` directory.
 - To rerun commands, see `commands_to_run_all.md` and reference the general instructions below.
 - DNDP is in a self-contained Jupyter Notebook and will be uploaded soon.

   
  
# How to Run the Code

## A 1-D Knapsack Example

This is an example of running everything for a 1-D knapsack problem (`kp`).  Note that Gurobi 11.0 is required.  

#### Initial Commands
0. Initialize directories for the knapsack problem.
   ```
   python -m blo.scripts.00_initialize_directories
   ```
1. Initializing the problem (a python dictionary) based on `blo/params.py`.
   ```
   python -m blo.scripts.01_initialize_problem --problem kp
   ```


#### End-to-end ML pipeline
Below is the list of commands to run the NN-based approximation.

2. Generating the data for training.
   ```
   python -m blo.scripts.02_generate_data --problem kp --n_procs N
   ```
3. Training the default model config and storing it.  For each model `--model_type`, see the above sections.  Arguments for the ml model can be modified here as well. Random search over model parameters can optionally be done as well.  See Appendix for details.
   ```
   python -m blo.scripts.03_train_nn --problem kp --model_type inst_encoder
   python -m blo.scripts.04_get_best_nn_rs --problem kp --model_type inst_encoder
   ```
4. Solving the ML-based surrogate optimization problem.  Important arguements:
   - Problem-specific arguments such as the number of items in the knapsack, `n`, are given by `--kp_n n`.
   - To change surrogate model between the lower and upper level, the argument `--approx_type {lower,upper}` can be used.
   - To change the type of value function correction the argument `--vf_constr_type {slack,dampening,none}` can be use.  If slack is used, then the coefficient can also be change (`--slack_obj_coef`)
   ```
   python -m blo.scripts.05_run_ml_blo --problem kp --model_type inst_encoder
   ```

#### Baseline: Knapsack-specific greedy surrogate model
6. Run the greedy surrogate optimization
   ```
   python -m blo.scripts.05_run_ml_blo --problem kp --model_type greedy --vf_constr_type none
   ```

#### Baseline: Running the Bilevel Solver from [Fischetti, 2016] [https://msinnl.github.io/pages/bilevel.html]
7. Initializing the bilevel optimization solver (https://msinnl.github.io/pages/bilevel.html).  Not that this only needs to be done once. 
   ```
   bash blo/baselines/blo_solver/solver/make_cplex_shared.sh
   ```

8. Run the bilevel optimization solver.  Note that addition problem arguments, such as `--kp_n n`, can also be modified here.
   ```
   python -m blo.scripts.06_run_blo_solver --problem kp
   ```
   For a new problem, `p`, this will need to be done by adding `blo/blo_solver/p.py` to write the problem to `.mps` and `.aux` files.
   
Notes:  
- CPLEX 12.7 is required for the bilevel solver. 
- Dynamic libraries for CPLEX (`libconcert.so`, `libcplex.so`,  `libilocplex.so`) are required to be in the root directory of the repo (see https://msinnl.github.io/pages/bilevel.html).  I.e., what is done in 7.
- This assumes the bilevel solver is located in `./blo/baselines/blo_solver/solver/` (should be done by default).
- The license file for the bilevel solver (`bilevel.license`) must be in the root directory  (should be done by default).


# Contributing & Making Additions:

### Adding New Benchmarks
In order to add a new benchmark problem, `p`, the following files need to be added:
- `blo/p.py`: To implement solving the follower problem and sampling/reading instances.
- `data_manager/p.py`: To implement sampling decisions and collect raw features.  Note all parallelization is handled by the base class.
- `data_preprocessor/p.py`: To implement features preprocessing for each model based on the raw features.
- `approximator/p.py`: To implement the upper/lower level surrogates and any other problem-specific functions (for example, greedy in the knapsack case).
- `baselines/blo_solver/p.py`: To implement writing the problem to the `.mps` and `.aux` files and calling the bilevel solver.
- `utils/p.py`: General utilities and getting paths to read/write to.

Note that the `__init__.py` files will also need to be edited slightly.  Additionally, problem information, such as instance sizes, # of samples, etc, should be added as a dictionary to `blo/params.py`.

### Adding New ML Models
How to add more ML models.  This will need to be done for every problem `p`.
- To add new models, simply add them to `blo/models/models.py`.
Note in most cases, changing `blo/data_preprocessing/p.py` and `blo/approximations/p.py` will need to be modified to preprocess data and modify the approximations accordingly.  In some cases, such as new features that need to be computed, such as heuristic/greedy solutions or features derived from problem information, should be added in `blo/data_managers/p.py` and `blo/blo/p.py`.

### Adding/Modifying Features
To add/modify features for problem `p`, simply edit `blo/data_preprocessing/p.py` and `blo/approximations/p.py` to include changes. 




# Reference

Please cite our work if you find our code/paper useful to your work. 

```
@inproceedings{
  dumouchelle2024neurbilo,
  title={Neur2Bi{LO}: Neural Bilevel Optimization},
  author={Justin Dumouchelle and Esther Julien and Jannis Kurtz and Elias Boutros Khalil},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024},
}
```


## Benchmark Instances and References

If using any of the benchmark problems/instances from our paper, please cite the appropriate references.  

### Knapsack Interdiction Problem
- Reference: Yen Tang, Jean-Philippe P Richard, and J Cole Smith. A class of algorithms for mixed-integer bilevel min–max optimization. *Journal of Global Optimization*, 66:225–262, 2016.
- Link to instances: [https://web.archive.org/web/20220121032905/http://jcsmith.people.clemson.edu/Test_Instances_files/BKPIns.zip](https://web.archive.org/web/20220121032905/http://jcsmith.people.clemson.edu/Test_Instances_files/BKPIns.zip)
- Note we provide these instances and instances with 100 items in the MibS input file format in `data/kp/solver_instances/`.  

### Critical Node Problem/Game
- Reference: Gabriele Dragotto, Amine Boukhtouta, Andrea Lodi, and Mehdi Taobane. The critical
node game, 2023.
- Link to instances: [https://github.com/ds4dm/CNG-Instances](https://github.com/ds4dm/CNG-Instances)
- Note that the instances used in this work are contained in the data directory that differ from those at the above link but were randomly generated using the same procedure.  We provide the MibS input file format instances used in our experiments in `data/cng/solver_instances/`.  

### Donor Recipient Problem
- Reference: Shraddha Ghatkar, Ashwin Arulselvan, and Alec Morton. Solution techniques for bi-level knapsack problems. *Computers & Operations Research*, 159:106343, 2023.
- Link to instances: [https://github.com/ashwin-1983/DR-BKP/](https://github.com/ashwin-1983/DR-BKP/)
- Note that if using these instances with our code, they will need to be downloaded, unzipped, and moved to `data/dr/DR-BKP-main/`.

### Discrete Network Design Problem
- Reference: David Rey. Computational benchmarking of exact methods for the bilevel discrete network design problem. *Transportation Research Procedia*, 47:11–18, 2020.
- Link to instances: [https://github.com/davidrey123/DNDP/](https://github.com/davidrey123/DNDP/)




## Machine Learning Baselines

### Input Supermodular Neural Network
- Reference: Bo Zhou, Ruiwei Jiang, and Siqian Shen. "Learning to solve bilevel programs with binary tender." *The Twelfth International Conference on Learning Representations*, 2024.
- Link to Full Repository: [https://github.com/bozlamberth/LearnBilevel](https://github.com/bozlamberth/LearnBilevel)


