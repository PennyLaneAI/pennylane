import numpy as np

print('System: FeMoco')
N = 2*76
Tgates_per_rot = np.ceil(9.2 + 1.15*np.log2(1/1e-4))
order = 6
Y2kp1dN2 = 1
time = 1e3

def scientific_notation(num):
    """Convert number to LaTeX scientific notation"""
    if num == 0:
        return "$0$"
    exp = int(np.floor(np.log10(abs(num))))
    coef = num / 10**exp
    return f"${coef:.2f} \\times 10^{{{exp}}}$"


def CDF_terms(N): return N

def basis_rotation(N, spin_breaking = False):
    r"""
    N is the number of spin orbitals.
    If spin_breaking is False:
        Each spin is rotated independently.
        We need (N/2)(N/2-1)/2 Givens rotations for each spin
        Each Givens rotation requires two single qubit rotation, one of type RX and another of type RY.
    else:
        We need N(N-1)/2 Givens rotations.
        Each Givens rotation requires two single qubit rotation, one of type RX and another of type RY.
    Additionally we always need N RZ rotations.
    """
    if not spin_breaking:
        rotations = {
            'RZ': N,
            'RX': 2*(N/2)*((N/2)-1)/2,
            'RY': 2*(N/2)*((N/2)-1)/2,
        }
    else:
        rotations = {
            'RZ': N,
            'RX': N*(N-1)/2,
            'RY': N*(N-1)/2,
        }
    rotations['Hadamard'] = 4*(rotations['RX'] + rotations['RY'])
    rotations['S'] = 2*(rotations['RY'])
    rotations['Adjoint(S)'] = 2*(rotations['RY'])
    rotations['CNOT'] = 2*(rotations['RX'] + rotations['RY'] + rotations['RZ'])
    return rotations

def CZZ(N, one_body = False):
    r"""
    N is the number of spin orbitals.
    Each spin is rotated independently.
    We need N*(N-1)/2 CZZ gates.
    """
    rotations = {
        'RZ': N if one_body else N*(N+1)/2,
        'CNOT': 0 if one_body else 2*N*(N+1)/2,
        'RX': 0, 'RY': 0, 'Hadamard': 0, 'S': 0, 'Adjoint(S)': 0
    }
    return rotations

def Trotter_cost(N, order):
    r"""
    This function computes the cost of implementing a Trotter step for a given number of spatial orbitals.
    N is the number of spin orbitals.
    Each CDF term contains a basis rotation and a set of CZZ gates.
    """
    if order == 1:
        first_order_uses = 1
    elif order == 2:
        first_order_uses = 2
    elif order == 4:
        first_order_uses = 10
    elif order == 6:
        first_order_uses = 26
    rotations = {'RZ': 0, 'RX': 0, 'RY': 0, 'Hadamard': 0, 'S': 0, 'Adjoint(S)': 0, 'CNOT': 0, 'Toffoli': 0, 'SWAP': 0, 'T': 0, 'Adjoint(T)': 0}
    for rot in ['RX', 'RY', 'RZ', 'Hadamard', 'S', 'Adjoint(S)', 'CNOT']:
        rotations[rot] += basis_rotation(N)[rot] * CDF_terms(N) * first_order_uses
        rotations[rot] += CZZ(N)[rot] * (CDF_terms(N)-1) * first_order_uses
        rotations[rot] += CZZ(N, one_body = True)[rot] * first_order_uses
    return rotations

def active_volume(rotations: dict) -> float:
    """
    Computes the active volume of the circuit
    """
    b = 10 # 10 bits of precision per rotation

    av = 0
    T = 25
    CCZ = 35

    av += rotations['RZ']*(4+b/40*(305+6*CCZ+24*T))
    av += rotations['RX']*(4+b/40*(305+6*CCZ+24*T))
    av += rotations['RY']*(4+3/2+b/40*(305+6*CCZ+24*T))

    av += rotations['Hadamard']*3
    av += rotations['CNOT']*4
    av += rotations['Toffoli']*(12+CCZ)

    av += rotations['SWAP']*12
    av += rotations['S']*3
    av += rotations['Adjoint(S)']*3

    av += rotations['T']*(T + (3+1)+1.5)
    av += rotations['Adjoint(T)']*(T + (3+1)+1.5)

    return av

alg_t = Tgates_per_rot * np.sum([Trotter_cost(N, order)[k] for k in ['RZ', 'RX', 'RY']])
print(f'Number of T gates per step = {scientific_notation(alg_t)}')
rotations = {k: v for k, v in Trotter_cost(N, order).items()}
alg_av = active_volume(rotations)
print(f'Active volume of per step = {scientific_notation(alg_av)}')


def QPE_cost(N, order, Y2kp1dN2 = 1, epsilon = 1e-3):
    r"""
    This function computes the cost of implementing a QPE step for a given number of spatial orbitals.
    N is the number of spin orbitals.
    Each CDF term contains a basis rotation and a set of CZZ gates.
    """
    Y2kp1 = Y2kp1dN2 * N**2
    rotations = Trotter_cost(N, order)
    for key, value in rotations.items():
        rotations[key] = value * (Y2kp1/epsilon)**(1/order) * 1/epsilon * np.pi/2
    return rotations



print('QPE')


alg_t = Tgates_per_rot * np.sum([QPE_cost(N, order, Y2kp1dN2)[k] for k in ['RZ', 'RX', 'RY']])
print(f'Number of T gates in algorithm = {scientific_notation(alg_t)}')
rotations = {k: v for k, v in QPE_cost(N, order, Y2kp1dN2).items()}
alg_av = active_volume(rotations)
print(f'Active volume of the algorithm = {scientific_notation(alg_av)}')

def spectroscopy_longest(N, order, epsilon, time, Y2kp1dN2 = 1):
    r"""
    This function computes the cost of implementing a QPE step for a given number of spatial orbitals.
    N is the number of spin orbitals.
    Each CDF term contains a basis rotation and a set of CZZ gates.
    """
    Y2kp1 = Y2kp1dN2 * N**2
    rotations = Trotter_cost(N, order)
    for key, value in rotations.items():
        rotations[key] = value * (Y2kp1/epsilon)**(1/(order)) * time
    return rotations

print('Spectroscopy simulation time')
order = 6
Tgates_per_rot = np.ceil(9.2 + 1.15*np.log2(1/1e-4))
Y2kp1dN2 = 1
time = 1e3
epsilon = 1e-3

alg_t = Tgates_per_rot * np.sum([spectroscopy_longest(N, order, epsilon, time, Y2kp1dN2)[k] for k in ['RZ', 'RX', 'RY']])
print(f'Number of T gates in algorithm = {scientific_notation(alg_t)}')
rotations = {k: v for k, v in spectroscopy_longest(N, order, epsilon, time, Y2kp1dN2).items()}
alg_av = active_volume(rotations)
print(f'Active volume of the algorithm = {scientific_notation(alg_av)}')


def simulation_time(N, order, time, Y2kp1dN2 = 1, epsilon = 1e-3):
    r"""
    This function computes the cost of simulating the Trotter formula for a fixed amount of time.
    N is the number of spin orbitals.
    order is the order of the Trotter formula.
    time is the time to simulate.
    """
    Y2kp1 = Y2kp1dN2 * N**2
    rotations = Trotter_cost(N, order)

    for key, value in rotations.items():
        rotations[key] = value * (Y2kp1/epsilon)**(1/order) * time**(1+1/order)
    return rotations

print('Simulation')

order = 6
alg_t = Tgates_per_rot * np.sum([simulation_time(N, order, time)[k] for k in ['RZ', 'RX', 'RY']])
print(f'Number of T gates in algorithm = {scientific_notation(alg_t)}')
rotations = {k: v for k, v in simulation_time(N, order, time).items()}
alg_av = active_volume(rotations)
print(f'Active volume of the algorithm = {scientific_notation(alg_av)}')