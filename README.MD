A collection of parallel implementations of the Cholesky Descomposition.
=======================================================================
Scalar aglorithm with OpenMP parallel for.

Blocked algorithm using BLAS with OpenMP parallel for, OpenMP tasks and Intel Threading Building Blocks

Scalar algorithm in OpenCL.


Compiling:
=========
#Scalar


icc -o program program.c -fopenmp -lm -mkl

#Blocked OpenMP

icc -o program program.c -fopenmp -lm -lmkl_intel_lp64 -lmkl_sequential -lmkl_core

#Blocked TBB

icc -o program program.cpp -tbb -std=c++0x -fopenmp -lm -lmkl_intel_lp64 -lmkl_sequential -lmkl_core

#Scalar OpenCL

gcc -fopenmp -Wall -g main.c cholesky.c spd_matrix.c -o main -lm -l OpenCL
