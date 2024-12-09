import gurobipy as gp
import numpy as np

import pyomo.environ as pe
from pao.pyomo import SubModel
from pyomo.environ import value

from .solver import Solver



class CriticalNodeGameSolver(Solver):
    """ Wrapper class for bilevel optimization solver by Fiscetti et al., 2017. """
    
    def __init__(self, args, cfg, opt_model):
        """ Constructor for solver. """
        super(CriticalNodeGameSolver, self).__init__(args, cfg, opt_model)
        
        # CNG parameters
        self.v = self.opt_model.v
        self.budget_mult_factor = 100
        self.profit_mult_factor = 1

        # rescale budgets
        if self.budget_mult_factor == 1:
            self.d_budgets = self.opt_model.d_budget_coefs
            self.a_budgets = self.opt_model.a_budget_coefs
            self.D = self.opt_model.D
            self.A = self.opt_model.A
        else:
            self.d_budgets = list(map(lambda x: int(self.budget_mult_factor * x), self.opt_model.d_budget_coefs)) 
            self.a_budgets = list(map(lambda x: int(self.budget_mult_factor * x), self.opt_model.a_budget_coefs)) 
            self.D = int(self.opt_model.D * self.budget_mult_factor)
            self.A = int(self.opt_model.A * self.budget_mult_factor)

        # rescale profits
        if self.profit_mult_factor == 1:
            self.d_profits = self.opt_model.d_profit_coefs
            self.a_profits = self.opt_model.a_profit_coefs
        else:
            self.d_profits = list(map(lambda x: int(self.profit_mult_factor * x), self.opt_model.d_profit_coefs)) 
            self.a_profits = list(map(lambda x: int(self.profit_mult_factor * x), self.opt_model.a_profit_coefs)) 


    def write_to_files(self, fp_mps, fp_aux, verbose=1):
        """ Writes instance to MPS/AUX files to be solved. """
        # MPS file
        # HEADER/NAME
        mps_file = [f'NAME: cng_test_instance']

        # Objective 
        mps_file.append('OBJSENSE')
        mps_file.append(' MAX')

        # ROWS names (i.e., constraints)
        mps_file.append('ROWS')
        mps_file.append(' N cost')
        mps_file.append(f' L DKP')                   # defender budget constraint
        mps_file.append(f' L AKP')                   # attacker budget constraint

        # constraints for linearization for (1-x) * (1-y)
        mps_file += [f' L lin_xy_1_{i}' for i in range(self.v)]
        mps_file += [f' L lin_xy_2_{i}' for i in range(self.v)]
        mps_file += [f' L lin_xy_3_{i}' for i in range(self.v)]

        # constraints for linearization for (1-x) * (1-y)
        mps_file += [f' L lin_xny_1_{i}' for i in range(self.v)]
        mps_file += [f' L lin_xny_2_{i}' for i in range(self.v)]
        mps_file += [f' L lin_xny_3_{i}' for i in range(self.v)]

        # constraints for linearization for (1-x) * (1-y)
        mps_file += [f' L lin_nxy_1_{i}' for i in range(self.v)]
        mps_file += [f' L lin_nxy_2_{i}' for i in range(self.v)]
        mps_file += [f' L lin_nxy_3_{i}' for i in range(self.v)]

        # constraints for linearization for (1-x) * (1-y)
        mps_file += [f' L lin_nxny_1_{i}' for i in range(self.v)]
        mps_file += [f' L lin_nxny_2_{i}' for i in range(self.v)]
        mps_file += [f' L lin_nxny_3_{i}' for i in range(self.v)]

        # COLUMNS (i.e., coefficients of variables for objectives and constraints)
        mps_file.append('COLUMNS')
        
        # upper-level variables/constraints
        for i in range(self.v):
            # upper-level objective coefs
            mps_file.append(f' x{i} cost 0')

            # budget constraints for defender
            mps_file.append(f' x{i} DKP {self.d_budgets[i]}')

            # constraints forlinearization for x * y
            mps_file.append(f' x{i} lin_xy_1_{i} -1')
            mps_file.append(f' x{i} lin_xy_3_{i} 1')

            # constraints for linearization for x * (1-y)
            mps_file.append(f' x{i} lin_xny_1_{i} -1')
            mps_file.append(f' x{i} lin_xny_3_{i} 1')

            # constraints for linearization for (1-x) * y
            mps_file.append(f' x{i} lin_nxy_1_{i} 1')
            mps_file.append(f' x{i} lin_nxy_3_{i} -1')

            # constraints for linearization for (1-x) * (1-y)
            mps_file.append(f' x{i} lin_nxny_1_{i} 1')
            mps_file.append(f' x{i} lin_nxny_3_{i} -1')

        # lower-level variables/constraints
        for i in range(self.v):

            # lower-level objective coefs
            mps_file.append(f' y{i} cost {0}')

            # bugdet constraints for attacker
            mps_file.append(f' y{i} AKP {self.a_budgets[i]}')

            # constraints for linearization for x * y
            mps_file.append(f' y{i} lin_xy_2_{i} -1')
            mps_file.append(f' y{i} lin_xy_3_{i} -1')

            # constraints for linearization for x * (1-y)
            mps_file.append(f' y{i} lin_xny_2_{i} 1')
            mps_file.append(f' y{i} lin_xny_3_{i} -1')

            # constraints for linearization for (1-x) * y
            mps_file.append(f' y{i} lin_nxy_2_{i} -1')
            mps_file.append(f' y{i} lin_nxy_3_{i} 1')

            # constraints for linearization for (1-x) * (1-y)
            mps_file.append(f' y{i} lin_nxny_2_{i} 1')
            mps_file.append(f' y{i} lin_nxny_3_{i} -1')

        # variables/constraints for linearization for x * y
        for i in range(self.v):
            mps_file.append(f' z_xy{i} cost {self.d_profits[i] * self.opt_model.eta}')
            mps_file.append(f' z_xy{i} lin_xy_1_{i} 1')
            mps_file.append(f' z_xy{i} lin_xy_2_{i} 1')
            mps_file.append(f' z_xy{i} lin_xy_3_{i} -1')

        # variables/constraints for linearization for x * (1-y)
        for i in range(self.v):
            mps_file.append(f' z_xny{i} cost {self.d_profits[i] * self.opt_model.epsilon}')
            mps_file.append(f' z_xny{i} lin_xny_1_{i} 1')
            mps_file.append(f' z_xny{i} lin_xny_2_{i} 1')
            mps_file.append(f' z_xny{i} lin_xny_3_{i} -1')

        # variables/constraints for linearization for (1-x) * y
        for i in range(self.v):
            mps_file.append(f' z_nxy{i} cost {self.d_profits[i] * self.opt_model.delta}')
            mps_file.append(f' z_nxy{i} lin_nxy_1_{i} 1')
            mps_file.append(f' z_nxy{i} lin_nxy_2_{i} 1')
            mps_file.append(f' z_nxy{i} lin_nxy_3_{i} -1')

        # variables/constraints for linearization for (1-x) * (1-y)
        for i in range(self.v):
            mps_file.append(f' z_nxny{i} cost {self.d_profits[i] * 1}')
            mps_file.append(f' z_nxny{i} lin_nxny_1_{i} 1')
            mps_file.append(f' z_nxny{i} lin_nxny_2_{i} 1')
            mps_file.append(f' z_nxny{i} lin_nxny_3_{i} -1')

        # RHS
        mps_file.append('RHS')
        mps_file.append(f' rhs DKP {self.D}')
        mps_file.append(f' rhs AKP {self.A}')

        # RHS for linearization
        for i in range(self.v):
            mps_file.append(f' rhs lin_xy_1_{i} 0')
            mps_file.append(f' rhs lin_xy_2_{i} 0')
            mps_file.append(f' rhs lin_xy_3_{i} 1')

            mps_file.append(f' rhs lin_xny_1_{i} 0')
            mps_file.append(f' rhs lin_xny_2_{i} 1')
            mps_file.append(f' rhs lin_xny_3_{i} 0')

            mps_file.append(f' rhs lin_nxy_1_{i} 1')
            mps_file.append(f' rhs lin_nxy_2_{i} 0')
            mps_file.append(f' rhs lin_nxy_3_{i} 0')

            mps_file.append(f' rhs lin_nxny_1_{i} 1')
            mps_file.append(f' rhs lin_nxny_2_{i} 1')
            mps_file.append(f' rhs lin_nxny_3_{i} -1')
        
        # BOUNDS
        mps_file.append('BOUNDS')
        
        # bounds for upper-level decisions
        for i in range(self.v):
            mps_file.append(f' LI bound x{i} 0.0000000')
            mps_file.append(f' UI bound x{i} 1.0000000')

        # bounds for lower-level decisions
        for i in range(self.v):
            mps_file.append(f' LI bound y{i} 0.0000000')
            mps_file.append(f' UI bound y{i} 1.0000000')

        # bounds for linearization decisions x * y
        for i in range(self.v):
            mps_file.append(f' LI bound z_xy{i} 0.0000000')
            mps_file.append(f' UI bound z_xy{i} 1.0000000')

        # bounds for linearization decisions x * (1-y)
        for i in range(self.v):
            mps_file.append(f' LI bound z_xny{i} 0.0000000')
            mps_file.append(f' UI bound z_xny{i} 1.0000000')

        # bounds for linearization decisions (1-x) * y
        for i in range(self.v):
            mps_file.append(f' LI bound z_nxy{i} 0.0000000')
            mps_file.append(f' UI bound z_nxy{i} 1.0000000')

        # bounds for linearization decisions (1-x) * (1-y)
        for i in range(self.v):
            mps_file.append(f' LI bound z_nxny{i} 0.0000000')
            mps_file.append(f' UI bound z_nxny{i} 1.0000000')

        mps_file.append('ENDATA')

        # save mps to file
        with open(fp_mps, 'w') as f:
            f.write('\n'.join(mps_file))

        if verbose:
            print(f'  Saved .mps file to: {fp_mps}')

        # AUX file
        n_vars = self.v * 5
        n_constrs = 1 + self.v * 12


        # write aux file
        aux_file = [f'N {n_vars}']
        aux_file.append(f'M {n_constrs}')
        aux_file +=  [f'LC {i}' for i in range(self.v, 6 * self.v)]
        aux_file +=  [f'LR {i}' for i in range(1, n_constrs + 1)]

        # objective for y
        aux_file +=  [f'LO 0' for i in range(self.v)]

        # objective for x * y, i.e., z_xy
        aux_file +=  [f'LO {self.opt_model.a_profit_coefs[i] * (1 - self.opt_model.eta) * self.profit_mult_factor}' for i in range(self.v)]

        # objective for x * (1-y), i.e., z_xny
        aux_file +=  [f'LO 0' for i in range(self.v)]

        # objective for (1-x) * y, i.e., z_nxy
        aux_file +=  [f'LO {self.opt_model.a_profit_coefs[i] * self.profit_mult_factor}' for i in range(self.v)]

        # objective for (1-x) * (1-y), i.e., z_nxny
        aux_file +=  [f'LO {- self.opt_model.gamma * self.opt_model.a_profit_coefs[i] * self.profit_mult_factor}' for i in range(self.v)]

        aux_file.append('OS -1')

        # save aux to file
        with open(fp_aux, 'w') as f:
            f.write('\n'.join(aux_file))

        if verbose:
            print(f'  Saved .aux file to: {fp_aux}')
    

    def read_solution(self, fp_sol):
        """ Reads solution from file.  """
        leader_obj, follower_obj, solution = self.get_final_res(fp_sol)

        leader_obj /= self.profit_mult_factor
        follower_obj /= self.profit_mult_factor

        follower_obj = - follower_obj

        x = []
        y = []

        for i in range(self.v):
            x.append(solution[f"x{i}"])
            y.append(solution[f"y{i}"])

        return leader_obj, follower_obj, x, y
