# Introduction
Polynomial chaos (PC) sampling algorithm implementation for UQ. Interfaces nicely with the solver provided by https://github.com/adegenna/cahnhilliard_2d.

# Solver
First, edit the filepaths in src/input_pce.dat and src/input_solver.dat to reflect what is on your machine. Then, run:

python driver.py input_pce.dat input_solver.dat

# Visualization
First, edit the filepaths in the visualization/plotPosterior.py script to reflect what is on your machine. Then, run:

python plotPosterior.py

# Example Output
Here is an example posterior calculation. The physics are calculated for the 2D Cahn Hilliard equations using the code references above, with dt = 5e-5. Inference is done on the uncertain parameter epsilon that appears in the Cahn Hilliard equation. The "ground truth" used is a Cahn Hilliard simulation run using a value of epsilon=0.1. We calculate 1000 MCMC samples in the posterior. The prior distribution is uniform[0,0.04], and we use Legendre chaos of order 50 (corresponding to 50 solver evaluations).

<img src="https://github.com/adegenna/mcmc/blob/master/pce.png">
