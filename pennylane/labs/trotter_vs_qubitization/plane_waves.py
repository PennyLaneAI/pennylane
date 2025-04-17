import numpy as np
from scipy import integrate

N = 1e6
eta = 1e3

def pauli_rotation_synthesis(epsilon):
    return 9.2 + 1.15*np.log2(1/epsilon)
def c_pauli_rotation_synthesis(epsilon):
    return 2*pauli_rotation_synthesis(epsilon)

def f(x, y):
    return 1/(x**2 + y**2)
def I(N0):
    return integrate.nquad(f, [[1, N0],[1, N0]])[0]

def d(x0, y0, z0, x1, y1, z1):
    return np.sqrt((x0-x1)**2 + (y0-y1)**2 + (z0-z1)**2)

def dT(x, y, z): return x**2 + y**2 + z**2

def I1(N0):
    r"""
    I1 is the integral of the 1/r potential over a box of size N0
    """
    d1 = partial(d, x0=0, y0=0, z0=0)
    return integrate.nquad(d1,[[-N0/2, N0/2],[-N0/2, N0/2], [-N0/2, N0/2]])[0]

def QFT(N, eta, epsilon):
    n = np.log2(N**(1/3))
    return 3 * eta * n**2 * c_pauli_rotation_synthesis(epsilon)

def I2(N0):
    r"""
    I2 is the integral of the 1/r^2 potential over a box of size N0
    """
    return integrate.nquad(d, [[-N0/2, N0/2],[-N0/2, N0/2],
                                [-N0/2, N0/2],[-N0/2, N0/2], 
                                [-N0/2, N0/2],[-N0/2, N0/2]])[0]

def IT(N0):
    return integrate.nquad(dT, [[-N0/2, N0/2],[-N0/2, N0/2], [-N0/2, N0/2]])[0]

def second_quantized_trotter(epsilons, p_fail, N, eta, Omega):
    r"""
    https://journals.aps.org/prx/pdf/10.1103/PhysRevX.8.011044, taken from TFermion
    """

    epsilon_QPE = epsilons[0]
    epsilon_HS = epsilons[1]
    epsilon_S = epsilons[2]
    
    t = np.pi/(2*epsilon_QPE)*(1/2+1/(2*p_fail))
    sum_1_nu = 4*np.pi*(np.sqrt(3)*N**(1/3)/2 - 1) + 3 - 3/N**(1/3) + 3*I(N**(1/3))
    max_V = eta**2/(2*np.pi*Omega**(1/3))*sum_1_nu
    max_U = eta**2/(np.pi*Omega**(1/3))*sum_1_nu
    nu_max = np.sqrt(3*(N**(1/3))**2)
    max_T = 2*np.pi**2*eta/(Omega**(2/3))* nu_max**2
    
    r = np.sqrt(2*t**3/epsilon_HS *(max_T**2*(max_U + max_V) + max_T*(max_U + max_V)**2))

    # Arbitrary precision rotations, does not include the Ry gates in F_2
    single_qubit_rotations = r*(8*N + 2*8*N*(8*N-1) + 8*N + N*np.ceil(np.log2(N/2))) # U, V, T and FFFT single rotations; the 2 comes from the controlled rotations, see appendix A
    epsilon_SS = epsilon_S/single_qubit_rotations
    
    exp_UV_cost = 8*N*(8*N-1)*c_pauli_rotation_synthesis(epsilon_SS) + 8*N*pauli_rotation_synthesis(epsilon_SS)
    exp_T_cost = 8*N*self.tools.pauli_rotation_synthesis(epsilon_SS)
    F2 = 2
    FFFT_cost = N/2*np.ceil(np.log2(N))*F2 + N/2*(np.ceil(np.log2(N))-1)*pauli_rotation_synthesis(epsilon_SS) 
    
    return r*(2*exp_UV_cost + exp_T_cost + 2*FFFT_cost)

def first_quantized_trotter(epsilons, p_fail, N, eta, Omega, nuclei):
    r"""
    See eq B1 to B5 in https://arxiv.org/pdf/2301.01203
    """

    epsilon_QPE = epsilons[0]
    epsilon_HS = epsilons[1]
    epsilon_S = epsilons[2]
    
    t = np.pi/(2*epsilon_QPE)*(1/2+1/(2*p_fail))
    norm_V = eta**2/(2*Omega**(1/3))*I2(N**(1/3))
    norm_U = eta**2/(Omega**(1/3))*I1(N**(1/3))
    norm_T = (2*np.pi**2/Omega**(2/3))*IT(N**(1/3))

    r = np.sqrt(2*t**3/epsilon_HS *(max_T**2*(max_U + max_V) + max_T*(max_U + max_V)**2))

    n = np.log2(N**(1/3))
    
    #todo: change this, I made it up
    cost_sum = lambda n: n
    cost_product = lambda n: n**2
    cost_sqrt = lambda n: n**2

    T_cost = 2*QFT(N, eta, epsilon_SS) + 3*eta*cost_sum(n) + 3*eta*cost_product(n)

    V_cost = eta*(eta-1)/2 * (3 * cost_product(n) + 2* cost_sum(n) + cost_sqrt(n) + cost_sum(n))
    U_cost = eta * nuclei * (3 * cost_product(n) + 2* cost_sum(n) + cost_sqrt(n) + cost_sum(n))

    return r*(2*exp_UV_cost + exp_T_cost + 2*FFFT_cost)
