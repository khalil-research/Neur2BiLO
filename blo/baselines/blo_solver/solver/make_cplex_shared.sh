#!/bin/bash
# Builds dynamic CPLEX libraries from static libraries

# cplex directory
CPLEX_DIR="/opt/ibm/ILOG/CPLEX_Studio127"
echo $CPLEX_DIR

# paths to dynamic link libraries
# default: all 3 binaries assumed to lie in the same directory
TARGET_CPLEX="."
TARGET_CONCERT="."

LIB_CPLEX="${CPLEX_DIR}/cplex/lib/x86-64_linux/static_pic/libcplex.a"
LIB_ILO_CPLEX="${CPLEX_DIR}/cplex/lib/x86-64_linux/static_pic/libilocplex.a"
LIB_CONCERT="${CPLEX_DIR}/concert/lib/x86-64_linux/static_pic/libconcert.a"

gcc -fpic -shared -Wl,--whole-archive "${LIB_CPLEX}" -Wl,--no-whole-archive -o libcplex.so
gcc -fpic -shared -Wl,--whole-archive "${LIB_CONCERT}" -Wl,--no-whole-archive -L"$TARGET_CPLEX" -lcplex  -Wl,-rpath,"$TARGET_CPLEX" -o libconcert.so
gcc -fpic -shared -Wl,--whole-archive "${LIB_ILO_CPLEX}" -Wl,--no-whole-archive -L"$TARGET_CPLEX" -lcplex -lconcert -Wl,-rpath,"$TARGET_CPLEX" -o libilocplex.so
