# Introduction
Polynomial chaos (PC) sampling algorithm implementation for UQ. Interfaces nicely with the solver provided by https://github.com/adegenna/cahnhilliard_2d.

# Solver
First, edit the filepaths in src/input_pce.dat and src/input_solver.dat to reflect what is on your machine. Then, run:

python driver.py input_pce.dat input_solver.dat