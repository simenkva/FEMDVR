import numpy as np 
from lobatto import GaussLobatto
from scipy.sparse import coo_matrix, csr_matrix
import matplotlib.pyplot as plt
from typing import Type

class FEMDVR:
    """Finite Element Method with Discrete Variable Representation.
    
    Constructs global differentiation matrices by combining local DVR elements
    with interface matching constraints for quantum mechanical calculations.
    """
    def __init__(self, nodes, n_points, element_class : Type[GaussLobatto]):
        """Initialize FEM-DVR grid and matrices.
        
        Args:
            nodes: Element boundary points (length n_elements + 1)
            n_points: Number of grid points per element (length n_elements)
            element_class: DVR element class (subclass of GaussLobatto)
            
        Creates:
            x: Global grid points
            w: Quadrature weights  
            D: First derivative matrix (original basis)
            D2: Second derivative matrix (original basis)
            D_symmetric: Symmetric first derivative matrix
            D2_symmetric: Symmetric second derivative matrix
            S: Mass/overlap matrix
            S_sqrt: Square root of mass matrix diagonal
            S_inv_sqrt: Inverse square root of mass matrix diagonal
            R: Interface matching matrix
        """
        
        self.n_points = np.array( n_points )
        self.nodes = np.array( nodes )
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

        # Store symmetric versions of the differentiation matrices.
        self.D2_symmetric = csr_matrix(S_inv_sqrt @ D2 @ S_inv_sqrt)
        self.D_symmetric = csr_matrix(S_inv_sqrt @ D @ S_inv_sqrt)

        # Compute differentiation matrices in the original basis.
        self.D = csr_matrix(np.linalg.solve(self.S,  D))
        self.D2 = csr_matrix(np.linalg.solve(self.S, D2))
        
        # Store square root of overlap matrix diagonal for converting wavefunctions when using symmetric operators.
        self.S_inv_sqrt = 1.0 / np.sqrt(np.diag(self.S))
        self.S_sqrt = np.sqrt(np.diag(self.S))


        self.R = R
        self.x = x
        self.dvr = dvr
        self.edge_indices = block_start2.copy()
        self.edge_indices[-1] -= 1

    
if __name__ == '__main__':
    # Simple test of the FEMDVR class.
    from lobatto import GaussLegendreLobatto
    nodes = [0.0, 1.0, 2.0]
    n_points = [5, 5]
    femdvr = FEMDVR(nodes, n_points, GaussLegendreLobatto)

    print("x:", femdvr.x)
    print("w:", femdvr.w)
    print("D:\n", femdvr.D.toarray())
    print("D2:\n", femdvr.D2.toarray())
    print("S:\n", femdvr.S)
    