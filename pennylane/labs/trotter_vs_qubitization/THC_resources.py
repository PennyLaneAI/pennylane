import numpy as np
from scipy.special import jv


def Toffoli_cost(N, M, aleph, beth, br = 7):
    r"""
    Cost of implementing THC according to eq 44 in https://journals.aps.org/prxquantum/pdf/10.1103/PRXQuantum.2.030305


    """

    nM = np.ceil(np.log2(M+1))
    d = N/2 + M*(M+1)/2
    m = 2*nM + 2 + aleph

    ks1 = 2**np.argmin([np.ceil(d/2**n) + m*(2**n - 1) for n in range(1, 25)])
    ks2 = 2**np.argmin([np.ceil(d/2**n) + 2**n for n in range(1, 25)])
    kr1 = 2**np.argmin([np.ceil(M/2**n) + np.ceil(N/(2*2**n)) + 2**n for n in range(1, 25)])
    kr2 = 2**np.argmin([np.ceil(M/2**n) + 2**n for n in range(1, 25)])

    cost = 30*nM + 4*br - 16 + 2*nM**2 + 3*aleph + np.ceil(d/ks1) \
        + m*(ks1 - 1) + np.ceil(d/ks2) + ks2 \
        + 2*M + 4*N*beth - 11*N/2 + np.ceil(M/kr1) + np.ceil(N/(2*kr1)) \
        + kr1 + np.ceil(M/kr2) + kr2

    return cost

def qubit_cost(N, M, aleph, beth, br = 7):
    r"""
    Qubit cost of implementing THC according to eq 46 in https://journals.aps.org/prxquantum/pdf/10.1103/PRXQuantum.2.030305

    N is the number of qubits.
    M is the number of CDF terms.
    aleph is the number of RX gates.
    beth is the number of RY gates.
    br is the number of CNOT gates.
    """
    nM = float(np.ceil(np.log2(M+1)))
    d = N/2 + M*(M+1)/2
    m = 2*nM + 2 + aleph

    ks1 = 2**float(np.argmin([np.ceil(d/2**n) + m*(2**n - 1) for n in range(1, 25)]))

    cost = N + 2*nM + beth + np.log2(d) + aleph + 5 + \
        np.max([m*ks1 + np.ceil(np.log2(d/ks1)), m + beth*N/2 + beth - 2])

    return cost


def qubitization_order(t, epsilon, one_norm):
    r"""
    t is the scaled simulation parameter
    epsilon is the error tolerance
    one_norm is the norm of the Hamiltonian
    """

    n = 0
    t *= one_norm
    theta = np.linspace(0, np.pi, 10000)
    target = np.exp(1j*t * np.cos(theta))
    ft = jv(0, t)
    error = np.abs(target - ft)
    errormax = np.max(error)

    while np.any(errormax > epsilon):
        n += 1
        ft += (1j)**n * jv(n, t) * np.exp(1j*n*theta)
        ft += (1j)**(-n) * jv(-n, t) * np.exp(-1j*n*theta)

        error = np.abs(target - ft)
        errormax = np.max(error)

    print(f"Order of the polynomial: {n}")
    return n



# Parameters from https://arxiv.org/abs/2501.06165 for QPE
# System: P450 
#M = 160
#N = 2*58
#aleph = 13
#beth = 13
#one_norm = 130.9

# System: FeMoco
M = 290
N = 2*76
aleph = 15
beth = 16
one_norm = 198.9


epsilon = 1e-3
time = 1e3
br = 7

# One step
toffoli = Toffoli_cost(N, M, aleph, beth, br) * 4
print(f'Tgate cost one step: {toffoli:.2e}')

qpe_cost = Toffoli_cost(N, M, aleph, beth, br) * np.pi*one_norm/(2*epsilon) * 4
print(f'QPE cost: {qpe_cost:.2e}')
qubit_cost = qubit_cost(N, M, aleph, beth, br)
print(f'Qubit cost: {qubit_cost:.2e}')
simulation_cost = Toffoli_cost(N, M, aleph, beth, br) * qubitization_order(time, epsilon, one_norm) * 4
print(f'Simulation cost: {simulation_cost:.2e}')

