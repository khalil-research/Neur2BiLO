import os
import sys
import subprocess

import numpy as np

from abc import ABC, abstractmethod



class Solver(ABC):
    """ Wrapper class for bilevel optimization solver by Fiscetti et al., 2017. """
    
    @abstractmethod
    def write_to_files(self, fp_mps, fp_aux):
        """ Samples instance. """
        pass


    @abstractmethod
    def read_solution(self, fp_sol):
        """ Reads solution from file.  """
        pass


    def __init__(self, args, cfg, opt_model):
        """ Constructor for solver. """
        self.args = args
        self.cfg = cfg
        self.opt_model = opt_model


    def call_solver(self, fp_mps, fp_sol, solver_dir, setting=4, time_limit=3600):
        """ Call solver on mps file specified. """
        # get solver command
        solver_cmd = f"{solver_dir}bilevel -mpsfile {fp_mps} -setting {setting} -time_limit {time_limit} -print_sol 2"
        
        # call solver to std out and exit
        if self.args.debug:
            subprocess.run(solver_cmd.split(" "))
            exit()

        # intiialize file to write to
        f = open(fp_sol, "w")

        # call solver, write to file and exit
        subprocess.run(solver_cmd.split(" "), stdout=f)


    def get_final_res(self, fp_sol):
        """ Reads solution file. """
        with open(fp_sol, 'r') as f:
            lines = f.readlines()

        obj_results = None
        sol_results = []

        for line in lines:

            # get objective results
            if line[:12] == ' LEADER COST':
                obj_results = line

            # once we have objective, then deal with the rest
            elif obj_results is not None:
                
                # skip empty lines
                if line == '\n':
                    continue

                # break if done processing solution (i.e., find this line in solver output)
                if line[:120] == '------------------------------------------------------------------------------------------------------------------------':
                    break
                
                # otherwise append results
                sol_results.append(line[:-2])


        obj_results = obj_results.split()

        # # process results
        leader_obj = float(obj_results[2])
        follower_obj = float(obj_results[5])

        solution = {}
        for line in sol_results:
            k, v = line.split()
            solution[k] = float(v)

        return leader_obj, follower_obj, solution


    def get_incumbent_times(self, fp_sol):
        """ Reads solution file. """
        with open(fp_sol, 'r') as f:
            lines = f.readlines()

        inc_times = []
        inc_objs = []

        for line in lines:

            if "Found incumbent of value"  in line: 

                line_list = line.split()

                inc_obj = float(line_list[4])
                inc_time = float(line_list[6])
                
                inc_objs.append(inc_obj)
                inc_times.append(inc_time)

        return inc_objs, inc_times


    def get_incumbent_times_revised(self, fp_sol):
        """ Revised version of getting incumbents more accurately. """
        def is_nodecount_line(line):
            """ Check if line is one that contains a nodecount and incumebnt. """
            l_split = line.split()

            # check for a cutoff
            if len(l_split) == 7 and "cutoff" == l_split[2]:
                return True, float(l_split[0]), float(l_split[3])

            elif len(l_split) == 8:
                try:
                    for val in l_split[:-1]:
                        if "Cuts" not in val:
                            s = float(val)
                    if l_split[-1][-1] == "%":
                        s = float(l_split[-1][:-1])
                    return True, float(l_split[0]), float(l_split[4])

                except:
                    return False, -1, -1

            elif len(l_split) == 9 and l_split[5] == "":
                l_split = l_split[1:]
                try:
                    for val in l_split[:-1]:
                        s = float(val)
                    if l_split[-1][-1] == "%":
                        s = float(l_split[-1][:-1])
                    return True, float(l_split[0]), float(l_split[4])

                except:
                    return False, -1, -1

            elif len(l_split) == 9 and "Cuts:" == l_split[5]:
                try:
                    for val in l_split[:5] + l_split[6:-1]:
                        s = float(val)
                    if l_split[-1][-1] == "%":
                        s = float(l_split[-1][:-1])
                    return True, float(l_split[0]), float(l_split[4])

                except:
                    return False, -1, -1

            return False, -1, -1

        def is_inc_line(line):
            """ Check if line is one that contains a nodecount and incumebnt. """
            l_split = line.split()
            if "Found incumbent of value"  in line: 
                l_split = line.split()
                inc_obj = float(l_split[4])
                inc_time = float(l_split[6])
                return True, inc_obj, inc_time
            return False, -1, -1

        def is_total_time(line):
            """ Check if line is one that contains a nodecount and incumebnt. """
            l_split = line.split()
            if "Total (root+branch&cut)"  in line: 
                l_split = line.split()
                inc_time = float(l_split[3])
                return True, inc_time
            return False, -1

        def get_time_approximation_from_node(nc, n_nodes, total_time):
            """ Gets linear approximation of time based on node count. """
            return (nc / n_nodes) * total_time

            l_split = line.split()
            if "Found incumbent of value"  in line: 
                l_split = line.split()
                inc_obj = float(l_split[4])
                inc_time = float(l_split[6])
            return inc_obj, inc_time

        # open file with lines
        with open(fp_sol, 'r') as f:
            lines = f.readlines()

        # get number of nodes in B&B and total time
        n_nodes = -1
        total_time = -1
        for line in lines:
            # check for number of nodes in line
            is_nc, nc, _ = is_nodecount_line(line)
            if is_nc:
                n_nodes = nc

            # check for total time in line
            is_time, t_time = is_total_time(line)
            if is_time:
                total_time = t_time
            
        
        print("Number of Nodes: ", n_nodes)
        print("Time:            ", total_time)

        if n_nodes == 0 or (n_nodes == -1 and total_time >= 0):    
            return [], []
        
        assert(n_nodes != -1)
        
        # store incumebent times and objectives in list
        inc_times = []
        inc_objs = []

        # parse for incumbents in B&B
        for line in lines:

            # check for new incumebnt in solving line
            is_nc, nc, inc_obj = is_nodecount_line(line)
            if is_nc:
                inc_time = get_time_approximation_from_node(nc, n_nodes, total_time)
                if np.abs(inc_obj - inc_objs[-1]) / np.abs(inc_objs[-1]) > 1e-10:
                    inc_objs.append(inc_obj)
                    inc_times.append(inc_time)
                continue
            
            # check for new incumebnt in incumbent
            is_nc, inc_obj, inc_time = is_inc_line(line)
            if is_nc:
                inc_objs.append(inc_obj)
                inc_times.append(inc_time)
                
                if np.abs(inc_obj - inc_objs[-1]) / np.abs(inc_objs[-1]) > 1e-10:
                    inc_objs.append(inc_obj)
                    inc_times.append(inc_time)

                continue

        return inc_objs, inc_times