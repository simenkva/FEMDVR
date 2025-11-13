import numpy as np


from numpy.polynomial.legendre import legval

def _gauss_jacobi_nodes(alpha: float, beta: float, m: int) -> np.ndarray:
    """Return the m Gauss–Jacobi nodes on (-1,1) for weight (1-x)^alpha(1+x)^beta via Golub–Welsch."""
    if m <= 0:
        return np.empty(0, dtype=float)
    k = np.arange(m, dtype=float)
    # Jacobi matrix diagonal
    if abs(alpha - beta) < 1e-15:
        a = np.zeros_like(k)
    else:
        a = (beta**2 - alpha**2) / ((2*k + alpha + beta)*(2*k + alpha + beta + 2))
        if abs(alpha + beta) < 1e-15:
            a[0] = 0.0
    # Off-diagonal (symmetric tridiagonal)
    kk = np.arange(m-1, dtype=float)
    b_sq = (
        4*(kk+1)*(kk+1+alpha)*(kk+1+beta)*(kk+1+alpha+beta)
        / ((2*kk+alpha+beta+1)*(2*kk+alpha+beta+2)**2*(2*kk+alpha+beta+3))
    ) if m > 1 else np.empty(0)
    J = np.diag(a) + np.diag(np.sqrt(b_sq), 1) + np.diag(np.sqrt(b_sq), -1)
    return np.sort(np.linalg.eigvalsh(J))



def gauss_legendre_radau(n: int):
    """
    Right-sided Gauss–Legendre–Radau rule on (0,1].
    
    Parameters
    ----------
    n : int
        Total number of nodes (>=2). Includes x=1, excludes x=0.
    
    Returns
    -------
    x : (n,) ndarray
        Nodes on (0,1], ascending, with x[-1] = 1.0
    w : (n,) ndarray
        Quadrature weights on [0,1]; sum(w) = 1.  Exact for polynomials up to degree 2n-2.
    D : (n,n) ndarray
        First-order differentiation matrix at x (barycentric formulation).
    """
    if n < 2:
        raise ValueError("n must be at least 2.")
    m = n - 1
    
    # Interior nodes on [-1,1]: zeros of Jacobi P_{n-1}^{(1,0)}; append endpoint +1
    t = _gauss_jacobi_nodes(alpha=1.0, beta=0.0, m=m)
    t_full = np.concatenate([t, [1.0]])
    t_full.sort()
    
    # True Gauss–Radau weights on [-1,1], then map to [0,1]
    # w_i = (1+t_i) / (n^2 * P_{n-1}(t_i)^2),  i=1..n-1;  w_end = 2/n^2
    coeffs = np.zeros(n); coeffs[-1] = 1.0  # Legendre P_{n-1}
    P_nm1_at_t = legval(t, coeffs)
    w_interior = (1.0 + t) / (n**2 * (P_nm1_at_t**2))
    w_end = 2.0 / (n**2)
    w_full = np.concatenate([w_interior, [w_end]])
    
    # Affine map to [0,1]
    x = 0.5*(t_full + 1.0)
    w = 0.5*w_full
    
    # First-derivative matrix via barycentric weights
    X = x[:, None]
    # Products for barycentric weights with diag=1 trick
    M = (X - X.T) + np.eye(n)
    lam = 1.0 / np.prod(M, axis=1)
    dX = X - X.T
    np.fill_diagonal(dX, np.inf)  # avoid warnings; division gives 0 on diag
    D = (lam[None, :] / lam[:, None]) / dX
    np.fill_diagonal(D, 0.0)
    D[np.diag_indices_from(D)] = -np.sum(D, axis=1)
    return x, w, D


# --------- Verification ---------
def verify_radau(n: int = 10) -> None:
    x, w, D = gauss_legendre_radau(n)
    # (1) quadrature exactness up to degree 2n-2
    q_errs = [abs(w @ (x**k) - 1/(k+1)) for k in range(2*n-1)]
    # (2) exact derivative for degree n-1 polynomial at the nodes
    k = n-1
    f = x**k
    df_exact = k * x**(k-1) if k>0 else np.zeros_like(x)
    d_err = np.max(np.abs(D @ f - df_exact))
    # (3) a smooth test
    f_s = np.sin(2.3*np.pi*x)
    df_s_exact = 2.3*np.pi*np.cos(2.3*np.pi*x)
    I_exact = (1 - np.cos(2.3*np.pi)) / (2.3*np.pi)
    I_err = abs(w @ f_s - I_exact)
    d_s_err = np.max(np.abs(D @ f_s - df_s_exact))
    # Report
    print(f"n = {n} nodes  |  sum(w) = {w.sum():.16f}  |  endpoint weight = {w[-1]:.6g}")
    print(f"Quadrature exactness (max error, deg ≤ 2n-2): {max(q_errs):.2e}")
    print(f"Derivative (degree n-1 poly) max error:      {d_err:.2e}")
    print(f"Smooth test: |∫ sin error| = {I_err:.2e} ; max |f' error| = {d_s_err:.2e}")
    print("First few nodes:", x[:min(6, n)])

# Demo run
# verify_radau(50)

def gauss_legendre_lobatto(n: int):
    if n < 2:
        raise ValueError("n must be at least 2.")
    m = n - 2
    t_interior = _gauss_jacobi_nodes(1.0, 1.0, m)
    t_full = np.concatenate(([-1.0], t_interior, [1.0]))
    coeffs = np.zeros(n); coeffs[-1] = 1.0  # P_{n-1}
    P_nm1_at_t = legval(t_interior, coeffs) if m > 0 else np.empty(0)
    w_end = 2.0 / (n * (n - 1))
    w_interior = 2.0 / (n * (n - 1) * (P_nm1_at_t**2)) if m > 0 else np.empty(0)
    w_full = np.concatenate(([w_end], w_interior, [w_end]))
    x = 0.5 * (t_full + 1.0)
    w = 0.5 * w_full
    # Differentiation matrix
    X = x[:, None]
    M = (X - X.T) + np.eye(n)
    lam = 1.0 / np.prod(M, axis=1)
    dX = X - X.T
    np.fill_diagonal(dX, np.inf)
    D = (lam[None, :] / lam[:, None]) / dX
    np.fill_diagonal(D, 0.0)
    D[np.diag_indices_from(D)] = -np.sum(D, axis=1)
    return x, w, D

def verify_lobatto(n: int = 12) -> None:
    x, w, D = gauss_legendre_lobatto(n)
    q_errs = [abs(float(w @ (x**k)) - 1.0/(k+1.0)) for k in range(2*n - 2)]
    q_err_max = max(q_errs)
    k = n - 1
    f = x**k
    df_exact = k * x**(k-1) if k > 0 else np.zeros_like(x)
    d_err_max = float(np.max(np.abs(D @ f - df_exact)))
    a = 2.3*np.pi
    f_s = np.sin(a*x); df_s_exact = a*np.cos(a*x)
    I_exact = (1 - np.cos(a)) / a
    I_err = abs(float(w @ f_s) - I_exact)
    d_s_err_max = float(np.max(np.abs(D @ f_s - df_s_exact)))
    print(f"n = {n} nodes | sum(w) = {w.sum():.16f}")
    print("Quadrature exactness (max error, deg ≤ 2n-3):", f"{q_err_max:.2e}")
    print("Derivative (degree n-1 poly) max error:      ", f"{d_err_max:.2e}")
    print("Smooth sin test: |∫ error| =", f"{I_err:.2e} ; max |f' error| =", f"{d_s_err_max:.2e}")
    print("First few nodes:", x[:min(6, n)], " ... last:", x[-3:])

#verify_lobatto(50)


def gauss_legendre(n):
    """
    Gauss–Legendre rule on (-1,1).
    
    Parameters
    ----------
    n : int
        Number of nodes (>=1).
    
    Returns
    -------
    x : (n,) ndarray
        Nodes on (-1,1), ascending.
    w : (n,) ndarray
        Quadrature weights on [-1,1]; sum(w) = 2.  Exact for polynomials up to degree 2n-1.
    D : (n,n) ndarray
        First-order differentiation matrix at x (barycentric formulation).
    """
    if n < 1:
        raise ValueError("n must be at least 1.")
    # Nodes: zeros of Legendre P_n
    t = _gauss_jacobi_nodes(alpha=0.0, beta=0.0, m=n)
    
    # Weights: w_i = 2 / ( (1 - t_i^2) * [P_n'(t_i)]^2 )
    coeffs = np.zeros(n+1); coeffs[-1] = 1.0  # Legendre P_n
    dcoeffs = np.polynomial.legendre.legder(coeffs)
    Pn_prime_at_t = legval(t, dcoeffs)
    w = 2.0 / ((1.0 - t**2) * (Pn_prime_at_t**2))
    
    # First-derivative matrix via barycentric weights
    X = t[:, None]
    M = (X - X.T) + np.eye(n)
    lam = 1.0 / np.prod(M, axis=1)
    dX = X - X.T
    np.fill_diagonal(dX, np.inf)
    D = (lam[None, :] / lam[:, None]) / dX
    np.fill_diagonal(D, 0.0)
    D[np.diag_indices_from(D)] = -np.sum(D, axis=1)
    return t, w, D

def gauss_legendre_grid(a, b, n):
    """
    Gauss–Legendre grid on [a,b].
    
    Parameters
    ----------
    a, b : float
        Interval endpoints, a < b.
    n : int
        Number of nodes (>=1).
    """
    if a >= b:
        raise ValueError("a must be less than b.")
    x, w, D = gauss_legendre(n)
    # Affine map from (-1,1) to (a,b)
    x_mapped = a + (b - a)*(x + 1)/2
    w_mapped = (b - a)/2 * w
    D_mapped = 2/(b - a) * D
    return x_mapped, w_mapped, D_mapped


def gauss_legendre_radau_grid(a, b, n):
    """
    Right-sided Gauss–Legendre–Radau grid on (a,b].
    
    Parameters
    ----------
    a, b : float
        Interval endpoints, a < b.
    n : int
        Total number of nodes (>=2). Includes x=b, excludes x=a.
    """
    if a >= b:
        raise ValueError("a must be less than b.")
    x, w, D = gauss_legendre_radau(n)
    # Affine map from (0,1] to (a,b]
    x_mapped = a + (b - a)*x
    w_mapped = (b - a)*w
    D_mapped = D / (b - a)
    return x_mapped, w_mapped, D_mapped

def gauss_legendre_lobatto_grid(a, b, n):
    """
    Gauss–Legendre–Lobatto grid on [a,b].
    
    Parameters
    ----------
    a, b : float
        Interval endpoints, a < b.
    n : int
        Total number of nodes (>=2). Includes both endpoints.
    """
    if a >= b:
        raise ValueError("a must be less than b.")
    x, w, D = gauss_legendre_lobatto(n)
    x_mapped = a + (b - a)*x
    w_mapped = (b - a)*w
    D_mapped = D / (b - a)
    return x_mapped, w_mapped, D_mapped

