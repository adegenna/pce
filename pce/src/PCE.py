import numpy as np
import matplotlib.pyplot as plt
import sys
from math import factorial
from pce.src.Polynomial import *

class PCE():
    """
    Class for PCE sampling.
    """
    def __init__(self,inputs,polynomial):
        self.initial_parameters = inputs.initial_parameters
        self.P                  = len(self.initial_parameters)
        self.sigma_likelihood   = inputs.sigma_likelihood
        self.pce_order          = inputs.polynomial_order
        self.prior_mu           = inputs.prior_mu
        self.prior_sigma        = inputs.prior_sigma
        self.param_iter         = 0
        self.params             = []
        self.posterior          = []
        self.outdir             = inputs.outdir
        self.polynomial         = polynomial
        self.posterior_samples  = inputs.posterior_samples
        
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

    def likelihood_gaussian(self,params_i):
        self.forward_model.reset_state()
        self.forward_model.set_parameters(params_i)
        self.forward_model.reset()
        self.forward_model.solve()
        U_model     = self.forward_model.get_current_state()
        U_model     = self.forward_model.state.state2D_to_1D( U_model )
        discrepancy = self.compute_model_data_discrepancy_Tfinal(U_model)
        return self.compute_likelihood_gaussian(discrepancy)

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

    def calculate_posterior(self,coeff):
        prior_x,prior_evals,posterior_x = self.polynomial.generate_samples_from_prior(self.posterior_samples)
        F_evals                         = self.polynomial.evaluate_surrogate(coeff,prior_x)
        posterior_evals                 = F_evals * prior_evals / np.mean(F_evals)
        return posterior_x, posterior_evals
