import gurobipy as gp
import numpy as np

import pyomo.environ as pe
from pao.pyomo import SubModel
from pyomo.environ import value

from .solver import Solver






class KnapsackSolver(Solver):
    """ Wrapper class for bilevel optimization solver by Fiscetti et al., 2017. """
    
    def __init__(self, args, cfg, opt_model):
        """ Constructor for solver. """
        super(KnapsackSolver, self).__init__(args, cfg, opt_model)
        
        # knapsack parameters
        self.n = len(self.opt_model.I)
        self.k = self.opt_model.k
        self.p = self.opt_model.p
        self.a = self.opt_model.a
        self.b = self.opt_model.b

    
    def write_to_files(self, fp_mps, fp_aux, verbose=1):
        """ Writes instance to MPS/AUX files to be solved. """
        # MPS file
        # HEADER/NAME
        mps_file = [f'NAME: kp_test_instance']

        # Objective 
        mps_file.append('OBJSENSE')
        mps_file.append(' MIN')

        # ROWS names (i.e., constraints)
        mps_file.append('ROWS')
        mps_file.append(' N cost')
        mps_file.append(f' L IDB')                  # interdiction budget constraint
        mps_file.append(f' L KP')                   # knapsack constraint
        mps_file += [f' L I{i}' for i in range(self.n)]  # integrality constraints

        # COLUMNS (i.e., coefficients of variables for objectives and constraints)
        mps_file.append('COLUMNS')
        
        # upper-level variables/constraints
        for i in range(self.n):
            # upper-level objective coefs
            mps_file.append(f' x{i} cost 0')

            # upper-level interdiction budget
            mps_file.append(f' x{i} IDB 1')

            # upper-level interdiction
            mps_file.append(f' x{i} I{i} 1')

        # lower-level variables/constraints
        for i in range(self.n):

            # lower-level objective coefs
            mps_file.append(f' y{i} cost {self.p[i]}')

            # interdiction x and y
            mps_file.append(f' y{i} I{i} 1')

            # lower-level budget
            mps_file.append(f' y{i} KP {self.a[i]}')


        # RHS
        mps_file.append('RHS')
        mps_file.append(f' rhs IDB {self.k}')
        mps_file.append(f' rhs KP {self.b}')
        mps_file += [f' rhs I{i} 1' for i in range(self.n)]
        
        # BOUNDS
        mps_file.append('BOUNDS')
        
        # bounds for upper-level decisions
        for i in range(self.n):
            mps_file.append(f' LI bound x{i} 0.0000000')
            mps_file.append(f' UI bound x{i} 1.0000000')

        # bounds for lower-level decisions
        for i in range(self.n):
            mps_file.append(f' LI bound y{i} 0.0000000')
            mps_file.append(f' UI bound y{i} 1.0000000')

        mps_file.append('ENDATA')

        # save mps to file
        with open(fp_mps, 'w') as f:
            f.write('\n'.join(mps_file))

        if verbose:
            print(f'  Saved .mps file to: {fp_mps}')

        # AUX file
        n_constrs = self.n + 1
        # todo.  
        aux_file = [f'N {self.n}']
        aux_file.append(f'M {n_constrs}')
        aux_file +=  [f'LC {i}' for i in range(self.n, 2 * self.n)]
        aux_file +=  [f'LR {i}' for i in range(1, n_constrs + 1)]
        aux_file +=  [f'LO {self.p[i]}' for i in range(self.n)]
        aux_file.append('OS -1')

        # save aux to file
        with open(fp_aux, 'w') as f:
            f.write('\n'.join(aux_file))

        if verbose:
            print(f'  Saved .aux file to: {fp_aux}')
    

    def read_solution(self, fp_sol):
        """ Reads solution from file.  """
        leader_obj, follower_obj, solution = self.get_final_res(fp_sol)

        follower_obj = - follower_obj

        x = []
        y = []

        for i in range(self.n):
            x.append(solution[f"x{i}"])
            y.append(solution[f"y{i}"])

        return leader_obj, follower_obj, x, y
