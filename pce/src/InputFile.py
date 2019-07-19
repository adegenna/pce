import numpy as np
from pprint import pprint
import sys

class InputFile():
    """
    Class for packaging all input/config file options together.

    **Inputs**

    ----------
    args : command line arguments 
        (passed to constructor at runtime) command line arguments used in shell call for main driver script.

    **Options**

    ----------
    projdir : string
        Absolute path to project directory
    datadir : string
        Absolute path to data directory
    loaddir : string
        Absolute path to load directory
    outdir : string
        Absolute path to output directory
    truestatepath : string
        Absolute path to true state file
    """
    
    def __init__(self,args=[]):
        try:
            inputfilename           = args.inputfilename_pce
            inputfilestream         = open(inputfilename)
            self.projdir            = inputfilestream.readline().strip().split('= ')[1];
            self.datadir            = inputfilestream.readline().strip().split('= ')[1];
            self.loaddir            = inputfilestream.readline().strip().split('= ')[1];
            self.outdir             = inputfilestream.readline().strip().split('= ')[1];
            self.truestatepath      = inputfilestream.readline().strip().split('= ')[1];
            initial_parameters      = inputfilestream.readline().strip().split('= ')[1];
            self.initial_parameters = np.array( initial_parameters.split(',') , dtype='float' )
            self.posterior_samples  = int(inputfilestream.readline().strip().split('= ')[1])
            self.sigma_likelihood   = float(inputfilestream.readline().strip().split('= ')[1])
            self.sigma_step         = float(inputfilestream.readline().strip().split('= ')[1])
            prior_mu                = inputfilestream.readline().strip().split('= ')[1]
            self.prior_mu           = np.array( prior_mu.split(',') , dtype='float' )
            prior_sigma             = inputfilestream.readline().strip().split('= ')[1]
            self.prior_sigma        = np.array( prior_sigma.split(',') , dtype='float' )
            self.polynomial_order   = int(inputfilestream.readline().strip().split('= ')[1])
            inputfilestream.close();
        except:
            print("Using no input file (blank initialization).")
    def printInputs(self):
        """
        Method to print all config options.
        """
        attrs = vars(self);
        print('\n');
        print("********************* INPUTS *********************")
        print('\n'.join("%s: %s" % item for item in attrs.items()))
        print("**************************************************")
        print('\n');
