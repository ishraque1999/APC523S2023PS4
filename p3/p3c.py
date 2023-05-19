import numpy as np
import matplotlib.pyplot as plt
import argparse

def U0(x, y):
    return np.exp(- ((x - 0.5) ** 2 + (y - 0.5) ** 2) / 0.15 ** 2)

def Ut(t, x, y, a, b):

    # Using the analytical solution
    k = x - a * t
    l = y - b * t

    # To stay within the domain
    k = k - np.floor(k)
    l = l - np.floor(l)

    U = U0(k,l)
    return U

# Function for the CTU Advection Solution
def CTU_advec(a, b, N, dt, T):
    dx = 1 / N
    dy = 1 / N

    mu = a * dt / dx
    nu = b * dt / dy
    
    x = np.arange(0, 1, dx)
    y = np.arange(0, 1, dy)

    u = U0(*np.meshgrid(x,y))
    error_arr = []

    t = 0

    while t < T:
        u_new = u.copy()
        for i in range(N):
            for j in range(N):
                # Applying the CTU method accounting for periodic BC
                u_new[i,j] = ((1 - mu) * (1 - nu) * u[i,j] + 
                              mu * (1 - nu) * u[(i - 1)%N, j] +
                              nu * (1 - mu) * u[i, (j - 1)%N] +
                               mu * nu * u[(i - 1)%N, (j - 1)%N])
        u = u_new
        # Finding the L2 error
        U_true = Ut(t, *np.meshgrid(x, y), a, b)
        error = ((u - U_true)**2).mean()
        error_arr.append(error)

        t += dt
        print(f"t = {t}", end="\r")
    return u, error_arr

# Function of the Modified LW Method

def mod_LW(a, b, N, dt, T):
    dx = 1 / N
    dy = 1 / N

    mu = a * dt / dx
    nu = b * dt / dy
    
    x = np.arange(0, 1, dx)
    y = np.arange(0, 1, dy)

    U = U0(*np.meshgrid(x,y))
    u_new = np.zeros_like(U)
    error_arr = []

    t = 0

    while t < T:
        U_start = U.copy()
        u = lambda i, j : U_start[i % N, j % N]
        for i in range(N):
            for j in range(N):
                k1 = a**2 * dt * (u(i + 1, j) - 2 * u(i, j) + u(i - 1, j)) / 2 / dx**2
                k2 = b**2 * dt * (u(i, j + 1) - 2 * u(i, j) + u(i, j - 1)) / 2 / dy**2
                k3 = a * b * dt * (u(i + 1, j + 1) - u(i + 1, j -1) - u(i - 1, j + 1) + u(i - 1, j - 1)) / 4 / dx / dy
                k4 = a * (u(i + 1, j) - u(i - 1, j)) / 2 / dx
                k5 = b * (u(i, j + 1) - u(i, j - 1)) / 2 / dy

                u_new[i,j] = (u(i,j) + dt * (k1 + k2 + k3 - k4 - k5))
        U = u_new
        U_true = Ut(t, *np.meshgrid(x, y), a, b)
        error = ((U - U_true)**2).mean()
        error_arr.append(error)
        t += dt
        print(f"t = {t}", end="\r")
    return U, error_arr

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Solve the Advection Equation')
    parser.add_argument('--CTU', action='store_true', help='Use CTU method for the solution')
    parser.add_argument('--LW', action='store_true', help='Use the Modified LW methof for the solution')
    parser.add_argument('--N', type=int, default=64, help='Size of the simulation domain')
    args = parser.parse_args()

    T = 10.
    N = args.N

    a = 1
    b = 2

    dx = dy = 1 / N
    dt = 0.2 / N

    x = np.arange(0, 1, dx)
    y = np.arange(0, 1, dy)

    if args.CTU:
        u_t10, errors = CTU_advec(a, b, N, dt, T)
        h = np.arange(0,T,dt)

        # Create a plot of the results 

        # Create the contour plot
        plt.pcolormesh(x, y, u_t10)
        plt.colorbar()
        # Add labels and title
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Contour Plot')
        plt.savefig('p3a_1_%d.png' %N, dpi=300, facecolor='white')
        plt.show()

        # Plotting the errors
        plt.plot(h, errors)
        plt.title('L2 Error of CTU Method when N = %d' %N)
        plt.xlabel('time (s)')
        plt.ylabel('Error') 
        plt.savefig('p3b_1_%d.png' %N, dpi=300, facecolor='white')
        plt.show()

    if args.LW:
        u_t10, errors = mod_LW(a, b, N, dt, T)
        h = np.arange(0,T,dt)

        # Create a plot of the results 

        # Create the contour plot
        plt.pcolormesh(x, y, u_t10)
        plt.colorbar()
        # Add labels and title
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Contour Plot')
        plt.savefig('p3a_2_%d.png' %N, dpi=300, facecolor='white')
        plt.show()

        # Plotting the errors
        plt.plot(h, errors)
        plt.title('L2 Error of LW Method when N = %d' %N)
        plt.xlabel('time (s)')
        plt.ylabel('Error') 
        plt.savefig('p3b_2_%d.png' %N, dpi=300, facecolor='white')
        plt.show()   





