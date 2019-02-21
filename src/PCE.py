import numpy as np
import matplotlib.pyplot as plt
import sys

class PCE():
    """
    Class for PCE sampling.
    """
    def __init__(self,inputs):
        self.initial_parameters = inputs.initial_parameters
        self.P                  = len(self.initial_parameters)
        self.sigma_likelihood   = inputs.sigma_likelihood
        #self.pce_order          = inputs.posterior_samples
        self.prior_mu           = inputs.prior_mu
        self.prior_sigma        = inputs.prior_sigma
        self.param_iter         = 0
        self.params             = []
        self.posterior          = []
        self.outdir             = inputs.outdir
        
    def set_true_state(self,U_truth):
        """
        Method to set the true state.
        """
        self.U_truth = U_truth
        
    def set_forward_model(self,physics):
        """
        Method to set the forward model used by the MCMC sampler.
        """
        self.forward_model = physics

    def initialize_parameters(self,p0):
        """
        Method to initialize parameters.
        """        
        self.params = p0.reshape((-1,1))

    def append_parameters(self,pnew):
        """
        Method to append to parameters list.
        """
        self.params      = np.vstack([self.params,pnew])
        self.param_iter += 1
        
    def compute_model_data_discrepancy_Tfinal(self,Umodel_tf):
        """
        Method to compute discrepancy function between data and model.
        """
        return np.linalg.norm( self.U_truth - Umodel_tf )

    def compute_likelihood_gaussian(self,discrepancy):
        """
        Method to compute likelihood function with Gaussian assumption.
        """
        return np.exp( -0.5*(discrepancy)**2/(self.sigma_likelihood**2) )

    def evaluate_gaussian_prior(self,x):
        """
        Method to evaluate Gaussian prior at a point x.
        """
        eval_x = 1.0
        for i in range(self.P):
            eval_x *= np.exp( -0.5*(x-self.prior_mu[i])**2/(self.prior_sigma[i]**2) )
        eval_x /= np.sqrt( (2*np.pi)**self.P * np.prod(self.prior_sigma) )
        return eval_x
        
    def append_posterior_sample(self,posterior):
        """
        Method to append a sample to the posterior record.
        """
        self.posterior = np.hstack( [self.posterior,posterior] )
    
    def write(self,outfile):
        """
        Writes to outdir/ the results of the sampling, ie. (parameter_i , posterior_i).
        """
        f = open(outfile,'w+')
        for i in range(self.params.shape[0]):
            line = str(self.params[i])[1:-1] + " , " + str(self.posterior[i])
            f.write(line)
            f.write('\n')            
        f.close()

    def evaluate_1d_hermite_polynomial(self,n,x):
        """
        Evaluate nth order hermite polynomial at point x.
        """
        Hm1 = 1.0
        H   = x 
        for i in range(1,n):
            Hp1 = x*H - i*Hm1
            Hm1 = H.copy()
            H   = Hp1.copy()
        return H

    def compute_jacobi_matrix(self,n):
        # H_p1 + (Bn - x)*H + An*H_m1 = 0
        # An = -n
        # Bn = 0
        J = np.zeros([n,n],dtype='complex')
        for i in range(n-1):
            J[i,i+1] = 1*np.sqrt(i+1)
            J[i+1,i] = 1*np.sqrt(i+1)
        return J

    def compute_nodes(self,n):
        J     = self.compute_jacobi_matrix(n)
        lam,v = np.linalg.eig(J)
        return lam
