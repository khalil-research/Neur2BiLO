# Initialize directories
```
python -m blo.scripts.00_initialize_directories --problem cng_10
python -m blo.scripts.00_initialize_directories --problem kp_18
python -m blo.scripts.00_initialize_directories --problem dr_30_hard
```

# Initialize problems
```
python -m blo.scripts.01_initialize_problem --problem kp_18
python -m blo.scripts.01_initialize_problem --problem kp_20
python -m blo.scripts.01_initialize_problem --problem kp_22
python -m blo.scripts.01_initialize_problem --problem kp_25
python -m blo.scripts.01_initialize_problem --problem kp_28
python -m blo.scripts.01_initialize_problem --problem kp_30

python -m blo.scripts.01_initialize_problem --problem kp_100
python -m blo.scripts.01_initialize_problem --problem cng_10
python -m blo.scripts.01_initialize_problem --problem cng_25
python -m blo.scripts.01_initialize_problem --problem cng_50
python -m blo.scripts.01_initialize_problem --problem cng_100
python -m blo.scripts.01_initialize_problem --problem cng_300
python -m blo.scripts.01_initialize_problem --problem cng_500

python -m blo.scripts.01_initialize_problem --problem dr_30_hard
```

# Generate dataset.  set --n_procs for parallelization
```
python -m blo.scripts.02_generate_data --problem kp_18
python -m blo.scripts.02_generate_data --problem kp_20
python -m blo.scripts.02_generate_data --problem kp_22
python -m blo.scripts.02_generate_data --problem kp_25
python -m blo.scripts.02_generate_data --problem kp_28
python -m blo.scripts.02_generate_data --problem kp_30

python -m blo.scripts.02_generate_data --problem cng_10
python -m blo.scripts.02_generate_data --problem cng_25
python -m blo.scripts.02_generate_data --problem cng_50
python -m blo.scripts.02_generate_data --problem cng_100
python -m blo.scripts.02_generate_data --problem cng_300
python -m blo.scripts.02_generate_data --problem cng_500

python -m blo.scripts.02_generate_data --problem dr_30_hard
```

# train models
```
python -m blo.scripts.03_train_nn --problem kp_18 --model_type inst_encoder --kp_use_greedy 1
python -m blo.scripts.03_train_nn --problem kp_20 --model_type inst_encoder --kp_use_greedy 1
python -m blo.scripts.03_train_nn --problem kp_22 --model_type inst_encoder --kp_use_greedy 1
python -m blo.scripts.03_train_nn --problem kp_25 --model_type inst_encoder --kp_use_greedy 1
python -m blo.scripts.03_train_nn --problem kp_28 --model_type inst_encoder --kp_use_greedy 1
python -m blo.scripts.03_train_nn --problem kp_30 --model_type inst_encoder --kp_use_greedy 1
python -m blo.scripts.03_train_nn --problem kp_100 --model_type inst_encoder --kp_use_greedy 1

python -m blo.scripts.03_train_nn --problem cng_10 --model_type inst_encoder --approx_type lower
python -m blo.scripts.03_train_nn --problem cng_25 --model_type inst_encoder --approx_type lower
python -m blo.scripts.03_train_nn --problem cng_50 --model_type inst_encoder --approx_type lower
python -m blo.scripts.03_train_nn --problem cng_100 --model_type inst_encoder --approx_type lower
python -m blo.scripts.03_train_nn --problem cng_300 --model_type inst_encoder --approx_type lower
python -m blo.scripts.03_train_nn --problem cng_500 --model_type inst_encoder --approx_type lower

python -m blo.scripts.03_train_nn --problem cng_10 --model_type inst_encoder --approx_type upper
python -m blo.scripts.03_train_nn --problem cng_25 --model_type inst_encoder --approx_type upper
python -m blo.scripts.03_train_nn --problem cng_50 --model_type inst_encoder --approx_type upper
python -m blo.scripts.03_train_nn --problem cng_100 --model_type inst_encoder --approx_type upper
python -m blo.scripts.03_train_nn --problem cng_300 --model_type inst_encoder --approx_type upper
python -m blo.scripts.03_train_nn --problem cng_500 --model_type inst_encoder --approx_type upper

python -m blo.scripts.03_train_nn --problem dr_30_hard --model_type inst_encoder --approx_type lower --scale_labels 1 --inst_value_relu_output 1
python -m blo.scripts.03_train_nn --problem dr_30_hard --model_type inst_encoder --approx_type upper --scale_labels 1 --inst_value_relu_output 1
```

# Get best models
```
python -m blo.scripts.04_get_best_nn_rs --problem kp_18 --model_type inst_encoder
python -m blo.scripts.04_get_best_nn_rs --problem kp_20 --model_type inst_encoder
python -m blo.scripts.04_get_best_nn_rs --problem kp_22 --model_type inst_encoder
python -m blo.scripts.04_get_best_nn_rs --problem kp_25 --model_type inst_encoder
python -m blo.scripts.04_get_best_nn_rs --problem kp_28 --model_type inst_encoder
python -m blo.scripts.04_get_best_nn_rs --problem kp_30 --model_type inst_encoder
python -m blo.scripts.04_get_best_nn_rs --problem kp_100 --model_type inst_encoder

python -m blo.scripts.04_get_best_nn_rs --problem cng_10 --model_type inst_encoder --approx_type lower
python -m blo.scripts.04_get_best_nn_rs --problem cng_25 --model_type inst_encoder --approx_type lower
python -m blo.scripts.04_get_best_nn_rs --problem cng_50 --model_type inst_encoder --approx_type lower
python -m blo.scripts.04_get_best_nn_rs --problem cng_100 --model_type inst_encoder --approx_type lower
python -m blo.scripts.04_get_best_nn_rs --problem cng_300 --model_type inst_encoder --approx_type lower
python -m blo.scripts.04_get_best_nn_rs --problem cng_500 --model_type inst_encoder --approx_type lower

python -m blo.scripts.04_get_best_nn_rs --problem cng_10 --model_type inst_encoder --approx_type upper
python -m blo.scripts.04_get_best_nn_rs --problem cng_25 --model_type inst_encoder --approx_type upper
python -m blo.scripts.04_get_best_nn_rs --problem cng_50 --model_type inst_encoder --approx_type upper
python -m blo.scripts.04_get_best_nn_rs --problem cng_100 --model_type inst_encoder --approx_type upper
python -m blo.scripts.04_get_best_nn_rs --problem cng_300 --model_type inst_encoder --approx_type upper
python -m blo.scripts.04_get_best_nn_rs --problem cng_500 --model_type inst_encoder --approx_type upper

python -m blo.scripts.04_get_best_nn_rs --problem dr_30_hard --model_type inst_encoder --approx_type lower

python -m blo.scripts.04_get_best_nn_rs --problem dr_30_hard --model_type inst_encoder --approx_type upper
```

# Evaluation. 
Notes:
 - These commands will create `table.dat` files for which a list of numbered Python commands will need to be run.
 - This will require running several thousand jobs, and most of them will be short, with the exception of running the solver.
```
python -m blo.scripts.cc_scripts.dat_eval --eval_type ml --model_type inst_encoder --approx_type lower --vf_constr_type slack
python -m blo.scripts.dat_scripts.dat_eval --eval_type ml --model_type inst_encoder --approx_type lower --vf_constr_type dampening
python -m blo.scripts.dat_scripts.dat_eval --eval_type ml --model_type inst_encoder --approx_type lower --vf_constr_type none
python -m blo.scripts.dat_scripts.dat_eval --eval_type ml --model_type inst_encoder --approx_type upper --vf_constr_type none
python -m blo.scripts.dat_scripts.dat_eval --eval_type greedy --model_type greedy --vf_constr_type none
python -m blo.scripts.dat_scripts.dat_eval --problem kp --eval_type solver
python -m blo.scripts.dat_scripts.dat_eval --problem kp --eval_type re_solver

python -m blo.scripts.dat_scripts.dat_eval --problem cng --eval_type ml --model_type inst_encoder --approx_type lower --vf_constr_type slack
python -m blo.scripts.dat_scripts.dat_eval --problem cng --eval_type ml --model_type inst_encoder --approx_type upper --vf_constr_type none
python -m blo.scripts.dat_scripts.dat_eval --problem cng --eval_type solver
python -m blo.scripts.dat_scripts.dat_eval --problem cng --eval_type re_solver

python -m blo.scripts.dat_scripts.dat_eval --problem dr_30_hard --eval_type ml --model_type inst_encoder --approx_type lower --vf_constr_type slack
python -m blo.scripts.dat_scripts.dat_eval --problem dr_30_hard --eval_type ml --model_type inst_encoder --approx_type upper --vf_constr_type none
```
