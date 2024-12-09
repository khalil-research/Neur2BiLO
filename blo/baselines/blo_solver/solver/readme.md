# A Solver for Mixed-Integer Bilevel Linear Problems based on Intersection-Cuts

The solver is associated with the publications 

> Intersection Cuts for Bilevel Optimization  
> A new general-purpose algorithm for mixed-integer bilevel linear programs  

by M. Fischetti, I. Ljubic, M. Monaci and M. Sinnl.

The software is for academic purposes only, see also license.md. To use the solver, a **license file** must be requested from the authors, request it by [mail][4].
The solver is provided as binary called *bilevel* compiled under Ubuntu 14.04 64bit with g++ 4.8.4 using CPLEX 12.7. It needs dynamic CPLEX libraries, which need to be generated manually by the user. 
The procedure is as following:

1. In the script *make_cplex_dynamic.sh* provided with the binary set the variable *CPLEX_DIR* to the base directory of the CPLEX installation on your system (e.g., /opt/ibm/ILOG/CPLEX_Studio127).
2. Run the resulting script *make_cplex_dynamic.sh* to create the dynamic CPLEX library files libconcert.so, libcplex.so and libilocplex.so.
3. Put the generated dynamic libraries and the requested license *bilevel.license* in the folder where the binary *bilevel* is.

## Running the Solver

The solver can be run as, e.g.,

>./bilevel -mpsfile myInstanceFolder/myInstance.mps -setting *NUMBER*

Setting refers to different settings as described in the papers.
They are encoded as follows

	1 SEP1++  
	2 SEP2++  
	3 XU++  
	4 MIX++  
	21 SEP1+F  
	22 SEP2+F  
	23 XU+F  
	24 MIX+F  
	31 SEP1+P  
	32 SEP2+P  
	33 XU+P  
	34 MIX+P  
	41 SEP1  
	42 SEP2  
	43 XU  
	44 MIX  
	98 HC  
	99 HC++  

There are also some other potential input parameters, namely

* mpsfile : the input file, part 1, needs to end with .mps (description see next section)
* auxfile : the input file, part 2, needs to end with .aux (if mpsfile and auxfile have the same name except the ending, the auxfile does not need to be given)
* time_limit : in seconds
* available_memory : in MB
* nodefile : location for CPLEX nodefile to be used if available_memory is exhausted
* num_threads : number of threads for CPLEX, 0: all available; note: opportunistic mode of CPLEX is used, to change see the randomseed option
* randomseed : random seed of CPLEX; negative seed turns on deterministic multithread mode of CPLEX
* node_limit : nodelimit of CPLEX; -1: no limit
* cplex_cuts : level of CPLEX-cuts; 0: off, 1: default, 2: moderate, 3:aggressive
* print_sol : 1 prints the solution at the end
* setting : the setting as mentioned above

These settings and parameters can also be shown by

>./bilevel -help

## Instance Format

The instance format is the same as proposed for the open-source solver [MiBS][1], it is described in detail, e.g., [here][2]. The input consists of two files, the mpsfile and the auxfile. In the mpsfile, the high-point relaxation (HPR) of the problem needs to be given. The mps format is a widely used format for (mixed integer) linear programming, see, e.g., [wikipedia][3] for details on this file format. The auxfile specifies the follower objective and also, which variables and constraints of the HPR are associated with the follower problem. It is defined as follows; text after # is comment for explanation and should *not* be part of the auxfile.

    N 2 # number of follower variables  
    M 3 # number of follower constraints   
    LC 0 # index in the MPS file of follower variable  
    LC 1 # index in the MPS file of follower variable  
    LR 0 # index in the MPS file of follower constraint  
    LR 1 # index in the MPS file of follower constraint  
    LR 2 # index in the MPS file of follower constraint  
    LO -786 # follower objective of follower variable (order as given by LC)  
    LO -529 # follower objective of follower variable (order as given by LC)  
    OS 1 # objective sense of the follower, 1: min, -1: max  

[1]: https://github.com/tkralphs/MibS
[2]: http://coral.ise.lehigh.edu/wp-content/uploads/bilevel/MibS_inputFile_html.html
[3]: https://en.wikipedia.org/wiki/MPS_%28format%29
[4]: mailto:markus.sinnl@univie.ac.at?subject=[BILEVEL]%20License%20Key%20Request&cc=m.fischetti@gmail.com,ivana.ljubic@essec.edu,michele.monaci@unibo.it
