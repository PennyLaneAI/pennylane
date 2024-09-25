import json
import os
import pennylane as qml
import numpy as np
from tqdm import tqdm
from scipy.linalg import logm
from scipy.special import roots_legendre

from coefficients.expCommutators.generate_coefficients import NCP_3_6, NCP_4_10, NCP_5_18, PCP_5_16, PCP_6_26
from coefficients.NI.coeffsNI import coeffsNI
from coefficients.NI.coeffsNIproc import coeffsNIproc
from coefficients.PF.coeffsSS import coeffsSS
from coefficients.PF.coeffsSSproc import coeffsSSproc

import copy
gs = copy.copy(qml.resource.resource.StandardGateSet)
gs.add('StatePrep')

# You can choose from BoseHubbard, FermiHubbard, Heisenberg, Ising, check https://pennylane.ai/datasets/
def load_hamiltonian(name = "Ising", periodicity="open", lattice="chain", layout="1x4"):

    hamiltonians = qml.data.load("qspin", sysname=name, periodicity=periodicity, 
                                lattice=lattice, layout=layout, attributes=["hamiltonians"])

    return hamiltonians[0].hamiltonians[1], eval(layout.replace('x', '*'))

def split_hamiltonian(H):
    # todo: fix
    H0 = 0*qml.Identity(0)
    H1 = 0*qml.Identity(0)

    for coeff, op in zip(H.coeffs, H.ops):
        if 'X' in str(op.pauli_rep):
            H1 = H1 + coeff * op
        else:
            H0 = H0 + coeff * op
    return H0, H1


def CommutatorEvolution(H0, H1, h, cs, positive = True):
    r"""
    Implements the evolution of the commutator term in the LieTrotter step.

    Arguments:
    ---------
    H0: FermiSentence
        The first (fast-forwardable) Hamiltonian
    H1: FermiSentence
        The second (fast-forwardable) Hamiltonian
    h: float
        The time step
    c: float
        The coupling strength of the interaction term
    cs: np.array
        The coefficients of the commutator
    positive: bool
        Whether to use positive or negative counte-palyndromic coefficients
    """
    m = len(cs)-1

    cs_ = cs * (-1)**(int(positive) + 1)
    for c0, c1 in zip(cs_[::2], cs_[1::2]):
        qml.TrotterProduct(H0, h*c0)
        qml.TrotterProduct(H1, h*c1)
    if m%2 == 0: 
        qml.TrotterProduct(H0, h*cs_[-1])

    if m%2 == 0: H0, H1 = H1, H0

    cs_ = cs[::-1]
    for c0, c1 in zip(cs_[::2], cs_[1::2]):
        qml.TrotterProduct(H0, h*c0)
        qml.TrotterProduct(H1, h*c1)
    if m%2 == 0: 
        qml.TrotterProduct(H0, h*cs_[-1])


def LieTrotter(H0, H1, h):
    r"""
    Simulates Lie Trotter step for the Hamiltonian H = hH0 + hH1 -i h*c[H0, H1]

    Arguments:
    ---------
    H0: FermiSentence
        The first (fast-forwardable) Hamiltonian
    H1: FermiSentence
        The second (fast-forwardable) Hamiltonian
    h: float
        The time step
    c: float
        The coupling strength of the interaction term
    commutator_method: str
        The method to compute the commutator.
    reversed: bool
        Whether to reverse the order of the terms in the Hamiltonian

    Returns:
    --------
    None
    """

    qml.TrotterProduct(H0, h)
    qml.TrotterProduct(H1, h)

def Strang(H0, H1, h):
    LieTrotter(H0, H1, h/2)
    LieTrotter(H1, H0, h/2)

def Suzuki4(H0, H1, h):
    
    uk = 1/(4-4**(1/3))

    Strang(H0, H1, uk*h)
    Strang(H0, H1, uk*h)
    Strang(H0, H1, (1-4*uk)*h)
    Strang(H0, H1, uk*h)
    Strang(H0, H1, uk*h)

def InteractionLieTrotter(H0, H1, h, ht, s, roots, zsm, reverse = False):
    r"""
    Implements the LieTrotter method for a CFQM operator.

    Arguments:
    ---------
    H0: First Hamiltonian term (fast-forwardable)
    H1: Second Hamiltonian term (fast-forwardable)
    t: float
        The current time
    h: float
        The time step
    s: int
        Half the order of the CFQM operator
    roots: np.array
        The roots of the Gauss Legendre polynomial
    zs: np.array
        The coefficients of the CFQM operator
    """

    if reverse: range_s = range(s-1, -1, -1)
    else: range_s = range(s)

    for k in range_s:
        qml.TrotterProduct(H0, -(h*roots[k])) # This selects \tau in H(\tau)
        qml.TrotterProduct(H1, -ht*zsm[k]) # This selects t in t*H(\tau)
        qml.TrotterProduct(H0, +(h*roots[k]))


def InteractionStrang(H0, H1, h, ht, s, roots, zsm):
    r"""
    Implements the LieTrotter method for a CFQM operator.

    Arguments:
    ---------
    H0: First Hamiltonian term (fast-forwardable)
    H1: Second Hamiltonian term (fast-forwardable)
    t: float
        The current time
    h: float
        The time step
    s: int
        Half the order of the CFQM operator
    roots: np.array
        The roots of the Gauss Legendre polynomial
    zs: np.array
        The coefficients of the CFQM operator

    Returns:
    --------
    None
    """

    InteractionLieTrotter(H0, H1, h, ht/2., s, roots, zsm)
    InteractionLieTrotter(H0, H1, h, ht/2., s, roots, zsm, reverse = True)

def InteractionSuzuki4(H0, H1, h, ht, s, roots, zsm):
    r"""
    Implements the Suzuki4 method for a CFQM operator.

    Arguments:
    ---------
    H0: First Hamiltonian term (fast-forwardable)
    H1: Second Hamiltonian term (fast-forwardable)
    t: float
        The current time
    h: float
        The time step
    s: int
        Half the order of the CFQM operator
    roots: np.array
        The roots of the Gauss Legendre polynomial
    zs: np.array
        The coefficients of the CFQM operator

    Returns:
    --------
    None
    """

    uk = 1/(4-4**(1/3))

    InteractionStrang(H0, H1, h, uk*ht, s, roots, zsm)
    InteractionStrang(H0, H1, h, uk*ht, s, roots, zsm)
    InteractionStrang(H0, H1, h, (1-4*uk)*ht, s, roots, zsm)
    InteractionStrang(H0, H1, h, uk*ht, s, roots, zsm)
    InteractionStrang(H0, H1, h, uk*ht, s, roots, zsm)



def expcommutator_coefficients(commutator_method):
        if commutator_method == 'NCP_3_6':
            cs, _ = NCP_3_6()
            positive = False
        elif commutator_method == 'NCP_4_10':
            cs, _ = NCP_4_10()
            positive = False
        elif commutator_method == 'PCP_5_16':
            cs, _ = PCP_5_16()
            positive = True
        elif commutator_method == 'PCP_6_26':
            cs, _ = PCP_6_26()
            positive = True
        elif commutator_method == 'PCP_4_12':
            cs, _ = PCP_5_16()
            positive = True
        elif commutator_method == 'NCP_5_18':
            cs, _ = NCP_5_18()
            positive = False
        return cs,positive

def InteractionPicture(H0, H1, time, h, s, m, stages = None, identifier = None):
    r"""
    Implements the interaction picture for the Hamiltonian H = hH0 + hH1

    Arguments:
    ---------
    H0: FermiSentence
        The first Hamiltonian term
    H1: FermiSentence
        The second Hamiltonian term
    h: float
        The time step
    s: int
        Half the order of the approximation
    m: int
        The number of exponentials in the CFQM
    stages: int
        The number of stages of the product formula
        used for each exponential in the CFQM
    identifier: str
        The identifier of the product formula
        used for each exponential in the CFQM

    Returns:
    --------
    None
    """

    dir_path = os.path.dirname(os.path.realpath(__file__))
    coeff_path = os.path.join(dir_path, 'coefficients/CFQM')
    def convert_keys_to_float(data):
        new_data = {}
        for key, value in data.items():
            new_key = float(key)
            new_data[new_key] = value
        return new_data

    with open(os.path.join(coeff_path,'zs.json'), 'r') as f:
        zs = json.load(f, object_hook=convert_keys_to_float)

    zs[1] = {1: [[1.]]}

    ##  Computing the roots of the Gauss Legendre polynomial, where the Hamiltonian is evaluated
    ##  in each segment
    roots, _ = roots_legendre(s)
    roots = (roots + 1)/2
    time_steps = np.arange(time/h)

    if s == 1:
        for _ in time_steps:
            for i in range(m-1, -1, -1): 
                InteractionLieTrotter(H0, H1, h, h, s, roots, zs[s][m][i])
            qml.TrotterProduct(H0, -h, order=1)

    else:
        kSS, _ = coeffsSS(o = 2*s, s = stages, identifier=identifier), []
        for _ in time_steps:
            for i in range(m-1, -1, -1):
                for k in kSS:
                    InteractionStrang(H0, H1, h, k*h, s, roots, zs[s][m][i])
            qml.TrotterProduct(H0, -h, order=1)



def ProductFormula(H0, H1, time, h, order, stages = None, identifier = None, processing = False):
    r"""
    Implements the interaction picture for the Hamiltonian H = hH0 + hH1

    Arguments:
    ---------
    H0: FermiSentence
        The first Hamiltonian term
    H1: FermiSentence
        The second Hamiltonian term
    h: float
        The time step
    order: int
        The order of the approximation
    stages: int
        The number of stages
    identifier: str
        The identifier of the method
    processing: bool
        Whether to use the processed coefficients

    Returns:
    --------
    None
    """

    time_steps = np.arange(time/h)

    if order == 1:
        for _ in time_steps:
            LieTrotter(H0, H1, h)
    elif order == 2:
        for _ in time_steps: 
            Strang(H0, H1, h)
    elif order == 4: 
        for _ in time_steps:
            Suzuki4(H0, H1, h)
    else:
        if processing:
            kSS, pSS, _ = coeffsSSproc(o = order, s = stages, identifier=identifier)
        else:
            kSS, pSS = coeffsSS(o = order, s = stages, identifier=identifier), []
            

            for ga in pSS:
                Strang(H0, H1, ga*h)

            for _ in time_steps:
                for k in kSS:
                    Strang(H0, H1, k*h)

            for ga in pSS[::-1]:
                Strang(H0, H1, -ga*h)


def NearIntegrable(H0, H1, time, h, order, stages, processing):
    r"""
    Implements a near integrable method for the Hamiltonian H = H0 + H1

    Arguments:
    ---------
    H0: FermiSentence
        The first Hamiltonian term
    H1: FermiSentence
        The second Hamiltonian term
    h: float
        The time step
    order: int
        The order of the approximation, for the CFQM method
    stages: int
        The number of stages
    processing: bool
        Whether to use the processed coefficients

    Returns:
    --------
    None
    """


    if processing:
        kass, kbss, Pass, Pbss, _ = coeffsNIproc(o = order, s = stages)
        pSS = [Pass, Pbss]
        kSS = [kass, kbss]
    else:
        kass, kbss, stages = coeffsNI(o = order, s = stages)
        pSS = [[], []]
        kSS = [kass, kbss]

    n_steps = np.arange(time/h)

    kass, kbss = kSS

    Pass, Pbss = pSS

    for pa, pb in zip(Pass, Pbss):
        qml.TrotterProduct(H0, time = pa*h, n = 1, order = 1)
        qml.TrotterProduct(H1, time = pb*h, n = 1, order = 1)

    for _ in n_steps:
        for ka, kb in zip(kass, kbss):
            qml.TrotterProduct(H0, time = ka*h, n = 1, order = 1)
            qml.TrotterProduct(H1, time = kb*h, n = 1, order = 1)

    for pa, pb in zip(Pass[::-1], Pbss[::-1]):
        qml.TrotterProduct(H1, time = -pb*h, n = 1, order = 1)
        qml.TrotterProduct(H1, time = -pa*h, n = 1, order = 1)


def basic_simulation(hamiltonian, time, n_steps, method, order, device, n_wires,
                    n_samples = 3, **kwargs):
    r"""
    Implements Hamiltonian simulation.

    Arguments:
    ---------
    hamiltonian: tuple
        (H0, H1, c)
    time: float
        The total time of the simulation
    n_steps: int
        The number of steps
    method: str
        One of 'InteractionPicture', 'ProductFormula', 'NearIntegrable'
    order: int
        Order of the approximation
    device: str
        The device to use for the simulation
    n_wires: int
        The number of wires
    n_samples: int
        The number of samples
    **kwargs: dict

    Returns:
    --------
    Average error
    """
    H0, H1= hamiltonian[0], hamiltonian[1]

    @qml.qnode(device)
    def initial_layer(init_weights, weights):
        # Initial state preparation, using a 2-design
        qml.SimplifiedTwoDesign(initial_layer_weights=init_weights, weights=weights, wires=range(n_wires))
        return qml.state()

    def circuit(time, n_steps, init_state):

        h = time / n_steps

        qml.StatePrep(init_state, wires=range(n_wires))

        if method == 'InteractionPicture':
            m = kwargs['m']
            s = order // 2
            stages = kwargs['stages']
            identifier = kwargs['identifier']
            InteractionPicture(H0, H1, time, h, s, m, stages, identifier)
        elif method == 'ProductFormula':
            processing = kwargs['processing']
            stages = kwargs['stages']
            identifier = kwargs['identifier']
            ProductFormula(H0, H1, time, h, order, stages, identifier, processing)
        elif method == 'NearIntegrable':
            processing = kwargs['processing']
            stages = kwargs['stages']
            NearIntegrable(H0, H1, time, h, order, stages, processing)
        else:
            raise NotImplementedError('Method not implemented')

        return qml.state()

    def call_approx_full(time, n_steps, init_state):
        resources = qml.resource.get_resources(circuit, gate_set = gs)(time, n_steps, init_state)
        state = qml.QNode(circuit, device)(time, n_steps, init_state)
        return state, resources

    average_error = 0.
    for n in tqdm(range(n_samples), desc='Initial states attempted'):
        init_weights = np.random.uniform(0, 2*np.pi, (n_wires,))
        weights = np.random.uniform(0, 2*np.pi, (3, n_wires-1, 2))
        init_state = initial_layer(init_weights, weights)
        state1, resources = call_approx_full(time, n_steps, init_state)
        state2, _ = call_approx_full(time, 2*n_steps, init_state)
        average_error = n/(n+1)*average_error + 1/(n+1)*np.linalg.norm(state1 - state2)

    return average_error, resources


