import numpy as np
from math import factorial

class Polynomial:

    def __init__(self,inputs):
        self.order       = inputs.polynomial_order
        self.prior_mu    = inputs.prior_mu
        self.prior_sigma = inputs.prior_sigma

    def evaluate_1d_polynomial(self,n,x):
        """
        Evaluate nth order polynomial at point x.
        """
        if (n == 0):
            return np.ones_like(x)
        else:
            Hm1 = np.ones_like(x)
            H   = x
            for i in range(1,n):
                Hp1 = self.calculate_Pp1_from_three_term_recurrence(x,H,Hm1,i)
                Hm1 = H.copy()
                H   = Hp1.copy()
            return H

    def calculate_Pp1_from_three_term_recurrence(self,x,P,Pm1,n):
        Pp1  = ( self.recurrence_A(n)*x + self.recurrence_B(n) )*P - self.recurrence_C(n)*Pm1
        return Pp1

    def compute_jacobi_matrix(self,n):
        J = np.zeros([n,n],dtype='complex')
        for i in range(n-1):
            alpha    = -self.recurrence_B(i) / self.recurrence_A(i)
            beta     = np.sqrt( self.recurrence_C(i+1) / (self.recurrence_A(i)*self.recurrence_A(i+1)) )
            J[i,i]   = alpha
            J[i,i+1] = beta
            J[i+1,i] = beta
        return J
    
    def compute_nodes_and_weights(self,n):
        J        = self.compute_jacobi_matrix(n)
        nodes,v  = np.linalg.eig(J)
        nodes    = np.real(nodes)
        beta_0   = self.compute_1d_polynomial_norm(0)**2
        weights  = np.real(beta_0 * (v[0]**2 / np.linalg.norm(v,axis=0)**2))
        idxnodes = np.argsort(nodes)
        return nodes[idxnodes],weights[idxnodes]

    def transform_base_nodes_with_prior(self,x):
        return self.prior_mu + self.prior_sigma*x
    
    def compute_gauss_quadrature_projection(self,nodes,weights,F):
        """
        Method to compute a Gauss-quadrature discrete projection of F.
        """
        Q       = len(weights)
        coeff   = np.zeros(self.order)
        F_nodes = np.zeros(Q)
        # Transform nodes according to prior distribution
        eval_nodes     = self.transform_base_nodes_with_prior(nodes)
        # First compute Q evaluations of F
        for k in range(Q):
            print("Computing function evaluation %d..." %(k+1))
            F_nodes[k] = F(eval_nodes[k])
        # Compute discrete projection
        for j in range(self.order):
            H_j       = self.evaluate_1d_polynomial(j,nodes)
            coeff[j]  = np.sum(F_nodes * H_j * weights)
            coeff[j] *= 1./self.compute_1d_polynomial_norm(j)**2
        return coeff,eval_nodes,F_nodes


    def compute_surrogate_using_gauss_quadrature(self,F):
        self.nodes,self.weights    = self.compute_nodes_and_weights(self.order)
        coeff,eval_nodes,F_nodes   = self.compute_gauss_quadrature_projection(self.nodes,self.weights,F)
        return coeff,eval_nodes,F_nodes

    def evaluate_surrogate(self,coeff,x):
        u_H      = np.zeros_like(x)
        for j in range(self.order):
            H_j  = self.evaluate_1d_polynomial(j,x)
            u_H += coeff[j] * H_j
        return u_H


# Different instantiations of Polynomial base
class Hermite(Polynomial):

    def __init__(self,inputs):
        Polynomial.__init__(self,inputs)
        self.recurrence_A  = lambda n : 1
        self.recurrence_B  = lambda n : 0
        self.recurrence_C  = lambda n : n

    def compute_1d_polynomial_norm(self,n):
        return np.sqrt(factorial(n))

    def generate_samples_from_prior(self,n):
        P_basic = np.random.gaussian(0,1,n)
        P_eval  = 1./np.sqrt(2*np.pi) * np.exp( -0.5*P_basic**2 )
        P_prior = self.transform_base_nodes_with_prior(P_basic)
        return P_basic,P_eval,P_prior

class Legendre(Polynomial):

    def __init__(self,inputs):
        Polynomial.__init__(self,inputs)
        self.recurrence_A  = lambda n : (2*n+1.)/(n+1)
        self.recurrence_B  = lambda n : 0
        self.recurrence_C  = lambda n : n/(n+1.)

    def compute_1d_polynomial_norm(self,n):
        return np.sqrt( 2./(2*n+1) )

    def generate_samples_from_prior(self,n):
        P_basic = np.random.uniform(-1,1,n)
        P_eval  = np.ones(n)
        P_prior = self.transform_base_nodes_with_prior(P_basic)
        return P_basic,P_eval,P_prior


class Polynomial_Basic_Inputs():
    
    def __init__(self , order=5 , mu=0 , sigma=1 ):
        
        self.polynomial_order = order
        self.prior_mu         = mu
        self.prior_sigma      = sigma

    
class Hermite_2d():
    
    def __init__( self , idx_IJ , prior_mu , prior_sigma ):
        
        assert( np.all( [ len(idx_ij) == 2 for idx_ij in idx_IJ ] ) )
        assert( len( prior_mu ) == 2 )
        assert( len( prior_sigma ) == 2 )
        
        self.idx_IJ      = idx_IJ
        self.prior_mu    = prior_mu
        self.prior_sigma = prior_sigma
        
    def compute_2d_polynomial_norm( self , n1 , n2 ):
        
        H1 = Hermite( Polynomial_Basic_Inputs( order=n1 ) )
        H2 = Hermite( Polynomial_Basic_Inputs( order=n2 ) )
        
        return H1.compute_1d_polynomial_norm(n1) * H2.compute_1d_polynomial_norm(n2)
    
    def evaluate_surrogate( self , coeff , x ):
        
        assert( np.all( [ np.size(ci) == 1 for ci in coeff ] ) )
        assert( np.shape(x)[0] == 2 )
        
        u_H      = np.zeros( np.shape(x)[1] )
        
        for (k,ij) in enumerate(self.idx_IJ):
            
            H_i   = Hermite( Polynomial_Basic_Inputs( ij[0] , self.prior_mu[0] , self.prior_sigma[0] ) )
            H_j   = Hermite( Polynomial_Basic_Inputs( ij[1] , self.prior_mu[1] , self.prior_sigma[1] ) )
            
            H_i_x = H_i.evaluate_1d_polynomial(ij[0],x[0])
            H_j_y = H_i.evaluate_1d_polynomial(ij[1],x[1])
            
            u_H  += coeff[k] * H_i_x * H_j_y
        
        return u_H



    
# Test script
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    class I():
        def __init__(self):
            self.polynomial_order = 6
            self.prior_mu         = 0
            self.prior_sigma      = 1

    inputs = I()
    h      = Hermite(inputs)
    l      = Legendre(inputs)
    
    xtest  = np.linspace(-3,3,100)
    cols   = np.linspace(0, 1, inputs.polynomial_order)
    for i in range(h.order):
        h_i = h.evaluate_1d_polynomial(i,xtest)
        l_i = l.evaluate_1d_polynomial(i,xtest)
        plt.figure(1); plt.subplot(121); plt.plot(xtest,h_i,c=plt.cm.cool(cols[i]) );
        plt.xlim([-2,2]); plt.ylim([-5.05,5.05])
        plt.figure(1); plt.subplot(122); plt.plot(xtest,l_i,c=plt.cm.cool(cols[i]) );
        plt.xlim([-1,1]); plt.ylim([-1.05,1.05])
        if (i > 0):
            h_nodes,h_weights = h.compute_nodes_and_weights(i)
            l_nodes,l_weights = l.compute_nodes_and_weights(i)
            plt.subplot(121); plt.plot(h_nodes,np.zeros_like(h_nodes),'o',c=plt.cm.cool(cols[i]) ); plt.grid()
            plt.subplot(122); plt.plot(l_nodes,np.zeros_like(l_nodes),'o',c=plt.cm.cool(cols[i]) ); plt.grid()
    plt.figure(2)
    plt.subplot(121); plt.plot(h_nodes,h_weights,'bo')
    plt.subplot(122); plt.plot(l_nodes,l_weights,'bo')
    plt.show()
