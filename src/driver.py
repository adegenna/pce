import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
from InputFile import *
from PCE import *
sys.path.append('../../')
import cahnhilliard_2d as ch

def main():
    """
    Main driver script for doing UQ on the CH2D equations.

    **Inputs**

    ----------
    args : command line arguments
        Command line arguments used in shell call for this main driver script. args must have a inputfilename member that specifies the desired inputfile name.

    **Outputs**

    -------
    inputs.outdir/results.txt : text file 
        time-integrated state
    """

    # Read inputs
    parser  = argparse.ArgumentParser(description='Input filename');
    parser.add_argument('inputfilename_pce',\
                        metavar='inputfilename_pce',type=str,\
                        help='Filename of the input file for the pce sampler')
    parser.add_argument('inputfilename',\
                        metavar='inputfilename',type=str,\
                        help='Filename of the input file for the forward solver')
    args          = parser.parse_args()
    inputs_pce    = InputFile(args)
    inputs_solver = ch.src.InputFile.InputFile(args)
    inputs_pce.printInputs()
    inputs_solver.printInputs()

    # Physics setup
    C0         = np.genfromtxt(inputs_solver.initialstatepath , delimiter=',')
    state      = ch.src.CahnHilliardState.CahnHilliardState(C0)
    physics    = ch.src.CahnHilliardPhysics.CahnHilliardPhysics(inputs_solver, state)

    # PCE sampler setup
    C_truth           = np.genfromtxt(inputs_pce.truestatepath)
    polynomial        = Legendre(inputs_pce)
    pce               = PCE(inputs_pce,polynomial)
    pce.set_true_state(C_truth)
    pce.set_forward_model(physics)
    coeff_H,eval_nodes,L_nodes   = pce.polynomial.compute_surrogate_using_gauss_quadrature(pce.likelihood_gaussian)
    H_test            = np.linspace(pce.polynomial.nodes[0],pce.polynomial.nodes[-1],100)
    eval_test         = pce.polynomial.transform_base_nodes_with_prior(H_test)
    ysurrogate        = pce.polynomial.evaluate_hermite_surrogate(coeff_H,H_test)
    plt.plot(eval_nodes,L_nodes,'bo',eval_test,ysurrogate,'r'); plt.show()
    #pce.calculate_posterior()
    
    # Output
    #pce.write(inputs_pce.outdir + "pce.out")
    
if __name__ == '__main__':
    main()
