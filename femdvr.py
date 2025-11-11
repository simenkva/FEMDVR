import numpy as np 
from lobatto import GaussLobatto
from scipy.sparse import coo_matrix, csr_matrix
import matplotlib.pyplot as plt
from typing import Type

class FEMDVR:
    """ A simple FEMDVR class. 
    
    Interface very similar to the GaussLobattto classes.
    
    """
    def __init__(self, nodes, n_points, element_class : Type[GaussLobatto]):
        """Constructor.
        
        Args:
            nodes (ndarray): Vector of nodes of boundaries of elements. Number of elements is this len(nodes) - 1. nodes[i] and nodes[i+1] defines the boundary of element i.
            n_points (ndarray): Number of *total* grid points per element. The polynomial degree over  element i is n_points[i] - 1.
            element_class: Class for the elements. A subclass of GaussLobatto (which is an abstract base class).
            
        The constructor creates the following attributes:
            self.D (ndarray): Differentiation matrix
            self.D2 (ndarray): Second-order differentiation matrix
            self.R (ndarray): Matrix that identifies nodes at element interfaces.
            self.x (ndarray): Nodes
            self.w (ndarray): Weights
            self.dvr (list): List of dvr object instances for each element
            self.edge_indices (ndarray): Indices of elements of x corresponding to element boundaries
            
        """
        
        self.n_points = n_points
        self.nodes = nodes
        self.degrees = self.n_points - 1
        self.n_intervals = len(nodes) - 1
        n_intervals = self.n_intervals

        n_constraints = n_intervals - 1    
        dim0 = np.sum(n_points) # dimension of function space without boundary matching
        D = np.zeros((dim0, dim0))
        R = np.zeros((dim0, dim0 - n_constraints))
        w = np.zeros((dim0,))
        x = np.zeros((dim0 - n_constraints,))
        
        # Extract intervals and generate individual DVRs.
        a = []
        b = []
        dim = []
        dvr = []
        block_start = [0] # indices for start of original basis
        block_start2 = [0] # indices for start of interface matched basis
        
        # Create DVRs over each element
        for i in range(n_intervals):
            a.append(nodes[i])
            b.append(nodes[i+1])
            dvr.append(element_class(a[i], b[i], n_points[i]))
            
        for i in range(n_intervals):
            # update dimension
            dim.append(n_points[i])
            # update block start indices
            # block_start[i] is the start of the original basis, block_start2[i]
            # is the start of the interface matched basis.
            block_start.append(block_start[i] + dim[i]) 
            block_start2.append(block_start2[i] + dim[i] - 1)

            # if this is the last interval, we need to add one more point to the interface matched basis
            # to account for the last point of the last element.
            if i == n_intervals-1:
                block_start2[-1] += 1

            # Fill the differentiation matrix D and the weights w
            # D is the differentiation matrix in the original basis, w are the weights.
            D[block_start[i]:block_start[i+1], block_start[i]:block_start[i+1]] += dvr[i].Dx
            w[block_start[i]:block_start[i+1]] = dvr[i].w 
            
            # Fill the interface matching matrix R
            # R is the matrix that identifies nodes at element interfaces.
            R[block_start[i]:block_start[i+1], block_start2[i]:block_start2[i+1]+1] = np.eye(dim[i])
            
            # Update nodes x
            # x are the nodes of the interface matched basis.
            x[block_start2[i]:block_start2[i+1]] = dvr[i].x[:block_start2[i+1] - block_start2[i]]
        
        
        
        # Compute second derivative matrix. 
        temp = D @ R
        D2 = -temp.T @ np.diag(w) @ temp # D2 is the second derivative matrix
        #self.D = R.T @ D @ R
        D = R.T @ np.diag(w) @ D @ R
        self.S = R.T @ np.diag(w) @ R # S is the mass matrix
        self.w = w @ R
        #print(self.S)
        #S_lu = np.linalg.cholesky(self.S) # Cholesky factorization of the mass matrix
        #self.D2 = np.linalg.inv(self.S) @ D2
        #self.D2 = csr_matrix(np.linalg.solve(S_lu, np.linalg.solve(S_lu.T, D2)))
        #self.D = csr_matrix(np.linalg.solve(S_lu, np.linalg.solve(S_lu.T, D)))
        
        # S is diagonal. Compute square root of inverse of S directly.
        S_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(self.S)))
        self.D2 = csr_matrix(S_inv_sqrt @ D2 @ S_inv_sqrt)
        self.D = csr_matrix(S_inv_sqrt @ D @ S_inv_sqrt)

        # Store square root of overlap matrix diagonal for converting wavefunctions.
        self.S_inv_sqrt = 1.0 / np.sqrt(np.diag(self.S))
        self.S_sqrt = np.sqrt(np.diag(self.S))

        self.R = R
        self.x = x
        self.dvr = dvr
        self.edge_indices = block_start2.copy()
        self.edge_indices[-1] -= 1

    
if __name__ == '__main__':
    from lobatto import GaussLegendreLobatto, GaussChebyshevLobatto
    a = -10
    b = 10
    n_elem = 10
    points_per_elem = 11

    nodes = a + np.linspace(0, (b-a)**.5, n_elem+1)**2
    n_points = np.ones((n_elem,), dtype = int) * points_per_elem

    femdvr = FEMDVR(nodes, n_points, GaussLegendreLobatto)

    # Get nodes and weights and differentiation matrix.
    x = femdvr.x
    w = femdvr.w
    D = femdvr.D
    D2 = femdvr.D2
    ei = femdvr.edge_indices
    
    H_full = -0.5 * D2 + np.diag(0.5*x**2)
    H = H_full[1:-1,1:-1]

    E, U = np.linalg.eig(H)
    i = np.argsort(E)
    E = E[i]
    U = U[:,i]

    print('Orthogonality check: ', U[:,:4].T @ np.diag(w[1:-1]) @ U[:,:4])
    print(E)
    
    import matplotlib.pyplot as plt
    plt.figure()
    for i in range(3):
        y = np.zeros((len(x),))
        y[1:-1] = U[:,i]
        plt.plot(x, y, color = f'C{i}', marker = '.', linestyle = '-')
        plt.plot(x[ei], y[ei], color = f'C{i}', marker = 'o', linestyle = 'none')    
    plt.show()
    
    plt.figure()
    plt.imshow(H, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title('Hamiltonian matrix')
    plt.show()
    