import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import scipy as scp
import argparse

# Function to set the boundary values
def B(u):
    B = dx ** 2 * (u**4)
    for k in range(N**2):
        i, j = k % N, k // N
        if i == 0:
            B[k] = B[k] - 1
        if i == N - 1:
            B[k] = B[k] - 1
        if j == 0:
            B[k] = B[k] - 1 
        if j == N - 1:
            B[k] = B[k] - 1
    return B

# Function to describe the equations
def f(u):
    return L @ u - B(u)

# Function describing the Jacobian matrix of the system (We are using sparse matrices)
def jac(u):
    return L - scp.sparse.diags(dx**2 * (4 * u**3), 0)

# Function for the matrix inversion
def solve_inv(A, b):
    return scp.sparse.linalg.inv(A) @ b

# Function for the matrix conjugate gradient
def solve_conjugate_grad(N):
    def solve(A, b):
        x = np.zeros_like(b)
        d = b - A @ x
        r = d.copy()
        for k in range(N):
            p = A @ d
            alpha = (r.T @ r) / (d.T @ p)
            x = x + alpha * d
            new_r = r - alpha * p
            beta = (new_r.T @ new_r) / (r.T @ r)
            d = new_r + beta * d
            r = new_r
        return x
    return solve

def raphson_newton(u0, func, jac, max_iter, method):
    u = u0.copy()
    for k in range(max_iter):
        print(f"Iteration = {k + 1} / {max_iter}", end="\r")
        u = u - method(jac(u), func(u))
    return u

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Solve the stiff equation of chemical kinetics')
    parser.add_argument('--inv', action='store_true', help='Use matrix inversion for the solution')
    parser.add_argument('--conj', action='store_true', help='Use conjugate gradient for the solution')
    parser.add_argument('--N', type=int, default=64, help='Size of the simulation domain')
    args = parser.parse_args()

    N = args.N
    N2 = N**2
    dx = 1 / (N + 2)

    # Creating L as a sparse matrix

    rows = []
    cols = []
    coefs = []

    for k in range(N**2):
        i, j = k%N, k//N
        if  i < N - 1:
            rows.append(k)
            cols.append(i + 1 + j*N)
            coefs.append(1)
        if i > 0:
            rows.append(k)
            cols.append(i - 1 + j*N)
            coefs.append(1)
        if j < N - 1:
            rows.append(k)
            cols.append(i + (j + 1)*N)
            coefs.append(1)
        if j > 0:
            rows.append(k)
            cols.append(i + (j - 1)*N)
            coefs.append(1)
        rows.append(k)
        cols.append(i + j*N)
        coefs.append(-4)  

    L = scp.sparse.csr_matrix((coefs, (rows, cols)))  

    # Setting up the domain
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    x, y = np.meshgrid(x, y)

    u0 = np.zeros(N2)
    if args.inv:
        solution = raphson_newton(u0, f, jac, 10, method=solve_inv)
    if args.conj:
        method = solve_conjugate_grad(20)
        solution = raphson_newton(u0, f, jac, 10, method=method) 
    solution = solution.reshape(N, N)
    plt.pcolormesh(x, y, solution, label="u(x, y)")
    plt.colorbar()
    plt.xlabel("x")
    plt.ylabel("y")
    if args.inv:
        plt.title("u(x, y) with N = %d using matrix inversion" %N)
        plt.savefig('p1a.png', dpi=300, facecolor='white')
    if args.conj:
        plt.title("u(x, y) with N = %d using conjugate gradient" %N)
        plt.savefig('p1b_%d.png' %N, dpi=300, facecolor='white')

    