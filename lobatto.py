from abc import ABC, abstractmethod
import numpy as np
from numpy.polynomial import legendre
from scipy.special import eval_legendre as Legendre # replace with numpy?
from scipy.linalg import lu_factor, lu_solve

class GaussLobatto(ABC):
    """ Abstract base class for Gauss Lobatto quadrature."""
    
    
    def __init__(self, a, b, N_tot):
        """ Constructor.
        
        Args:
            a (float): Left endpoint of interval
            b (float): Right endpoint of interval
            N_tot (int): Total number of grid points
        """
        
        self.set_grid(a, b, N_tot)

    def quad(self, y):
        """ Quadrature of y = y(x), a function evaluated on the grid, *without* weight function usually included.
        
        Args:
            y (ndarray): Vector of function values.
            
        Returns:
            Quadrature over grid of function y.
        """
        
        if len(y) == self.N_tot:
            return np.sum(self.qw * y)
        elif len(y) == self.N_tot - 2:
            return np.sum(self.qw[1:-1] * y)
        else:
            raise ValueError
                    
    
    def gauss_quad(self, y):
        """ Gaussian quadrature of y = y(x), a function evaluated on the grid, *without* weight function usually included.
        
        Args:
            y (ndarray): Vector of function values.
            
        Returns:
            Quadrature over grid of function y.
        """
        
        if len(y) == self.N_tot:
            return np.sum(self.qw * y)
        elif len(y) == self.N_tot - 2:
            return np.sum(self.qw[1:-1] * y)
        else:
            raise ValueError
        
    @abstractmethod
    def set_grid(self, a, b, N):
        """ Set the grid parameters and compute matrices etc. Abstract method.
        
        Implementation should define the following attributes:
            self.a
            self.b
            self.N_tot
            self.x # quadrature nodes
            self.w # quadrature weights
            self.qw # quadrature weights for integral without weight
            self.Dx # differentiation matrix
            self.T  # polynomial matrix, "coeff to grid" basis change matrix,  T[j,i] = poly_j(x_i)
            self.M  # analysis matrix, "grid to coeff" basis change matrix, inverse of T.T
        
        Args:
            a (float): Left endpoint of interval
            b (float): Right endpoint of interval
            N_tot (int): Total number of grid points
            
        """


        pass
    

class GaussChebyshevLobatto(GaussLobatto):
    """ class ChebyshevLobato:
    Simple class for Chebyshef-Lobatto pseudospectral method and quadrature.
    
    WARNING; There is a bug in the diff matrix. DO NOT USE!
    """


    def set_grid(self, a, b, N):
        """ Set the grid parameters and compute matrices etc. See documentation of superclass GaussLobatto.
        

        """

        # Grid parameters
        assert(N > 0)
        self.a = a
        self.b = b
        self.N_tot = N


        self.theta = np.pi * np.arange(N) / (N - 1)
        self.theta = np.flip(self.theta)

        theta = self.theta
        self.x = np.cos(self.theta)
        x = self.x
        self.T = np.zeros((N,N))
        self.Dx = np.zeros((N,N))
        self.w = np.ones((N,))
        self.w[0] = .5
        self.w[-1] = .5



        self.intT = np.zeros((N,))
        for i in range(N):
            if i == 1:
                self.intT[i] = 0
            else:
                self.intT[i] = ((-1)**i + 1) / (1 - i**2)



        self.p = 1/self.w
        p = self.p

        for i in range(N):
            self.T[:,i] = np.cos(i * theta)
            for j in range(N):
                if i == 0 and j == 0:
                    self.Dx[i,j] = (1 + 2*(N-1)**2)/6
                elif i == N-1 and j == N-1:
                    self.Dx[i,j] = -(1 + 2*(N-1)**2)/6
                elif i == j:
                    self.Dx[i,j] = -x[j]/(2*(1-x[j]**2))
                else:
                    self.Dx[i,j] = (-1)**(i+j) * p[i]/(p[j]*(x[i]-x[j]))

        # scale domain and diff matrix
        self.x = (b-a) * (x + 1) / 2 + a
        self.Dx = 2 / (b-a) * self.Dx
        self.w /= ((N-1)/2)**.5

        # Analysis matrix: For f = Tc, c = M f, where f is on the grid and c are coeffs.
        self.M =   np.diag(self.w) @ self.T.T @ np.diag(self.w) # analysis 

        # Copute quadrature weights for computing definite integrals via self.quad.
        self.qw = self.M.T @ self.intT * (b-a) / 2
        self.w *= (b-a)/2



class GaussLegendreLobatto(GaussLobatto):
    """ Gauss--Legendre--Lobatto class. """
    
    
    def set_grid(self, a, b, N_tot):
        """Set the grid parameters and compute matrices etc.

        """

        # Grid parameters
        assert N_tot > 0

        ##########################################################
        self.N_tot = N_tot
        N = N_tot - 1

        # Get inner nodes
        c = np.zeros((N + 1,))
        c[-1] = 1
        dc = legendre.legder(c)

        self.x = np.zeros(N + 1)
        self.x[0] = -1
        self.x[-1] = 1
        self.x[1:-1] = legendre.legroots(dc)

        # Check for correct number of roots, and uniqueness
        assert len(self.x[1:-1]) == N - 1
        assert np.allclose(np.unique(self.x[1:-1]), self.x[1:-1])

        # Compute PN(x_i) and the weights
        self.PN_x = legendre.legval(self.x, c)
        self.w = 2 / (N * (N + 1) * self.PN_x**2)
        ############################################################

        self.Dx = np.zeros((N_tot, N_tot))
        for i in range(N_tot):
            for j in range(N_tot):
                if i == 0 and j == 0:
                    self.Dx[i, j] = -1 / 4 * (N_tot - 1) * N_tot
                elif i == N_tot - 1 and j == N_tot - 1:
                    self.Dx[i, j] = 1 / 4 * (N_tot - 1) * N_tot
                elif (i == j) and (0 < j < N_tot - 1):
                    self.Dx[i, j] = 0
                else:
                    self.Dx[i, j] = self.PN_x[i] / (
                        self.PN_x[j] * (self.x[i] - self.x[j])
                    )
        
        # # evaluate Legendre polynomials up to degree N on grid, as
        # # well as their derivatives.
        # L = np.zeros((N_tot,N_tot))
        # dL = np.zeros((N_tot,N_tot))
        # for n in range(N_tot):
        #     c = np.zeros((n+1,))
        #     c[-1] = 1
        #     L[:,n] = legendre.legval(self.x, c)
        #     dc = legendre.legder(c)
        #     dL[:,n] = legendre.legval(self.x, dc)
            
        # # Compute analysis matrix, i.e., if
        # # f(x) = Lc, then c = Tf (evaluated at nodes)
        # self.T = np.linalg.inv(L)
        
        # L_lu = lu_factor(L.T)
        
        # # Compute differentiation matrix in nodal basis.
        # self.Dx = lu_solve(L_lu, dL.T).T 



        # scale domain and diff matrix
        self.x = (b - a) * (self.x + 1) / 2 + a
        self.Dx = 2 / (b - a) * self.Dx
        self.w = self.w * (b - a) / 2
        self.qw = self.w
        
        
        



def quad_test(C):
    N = 15
    a = -4
    b = 2
    C = C(a, b, N)
    x = C.x
    f = np.exp(x)
    integral_quad = C.quad(f)
    integral_exact = np.exp(b) - np.exp(a)
    print(f'Integral of exp(x) from {a} to {b} is {integral_quad}, the error is {integral_quad - integral_exact}.')

def quad_test2(CClass):

    N = 4
    a = 0
    b = 1
    C = CClass(a, b, N)
    print(C.x, C.w)
    x = C.x
 
    for p in range(2*N):
        f = (p+1) * x ** p
        integral_quad = C.quad(f)
        integral_exact = 1
        print(f'p = {p} integral error is {integral_quad - integral_exact}.')

if __name__ == "__main__":

    # Just a simple test of quadrature.
    # Compute definite integral of exp(x)

    quad_test(GaussChebyshevLobatto)
    quad_test2(GaussChebyshevLobatto)

    quad_test(GaussLegendreLobatto)
    quad_test2(GaussLegendreLobatto)
