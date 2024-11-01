from functools import singledispatch
from typing import Union

import numpy as np

import pennylane as qml
from pennylane.pauli import PauliSentence, PauliWord
from .bosonic import BoseSentence, BoseWord
from .christiansenForm import christiansen_integrals, christiansen_integrals_dipole


def christiansen_mapping(
    bose_operator: Union[BoseWord, BoseSentence],
    ps: bool = False,
    wire_map: dict = None,
    tol: float = None,
):
    r"""Convert a bosonic operator to a qubit operator using the Christiansen mapping.

    The bosonic creation and annihilation operators are mapped to the Pauli operators as

    .. math::

        a^{\dagger}_0 =  \left (\frac{X_0 - iY_0}{2}  \right ), \:\: \text{...,} \:\:
        a^{\dagger}_n = \frac{X_n - iY_n}{2},

    and

    .. math::

        a_0 =  \left (\frac{X_0 + iY_0}{2}  \right ), \:\: \text{...,} \:\:
        a_n = \frac{X_n + iY_n}{2},

    where :math:`X`, :math:`Y`, and :math:`Z` are the Pauli operators.

    Args:
        bose_operator(BoseWord, BoseSentence): the bosonic operator
        ps (bool): whether to return the result as a PauliSentence instead of an
            Operator. Defaults to False.
        wire_map (dict): a dictionary defining how to map the orbitals of
            the bose operator to qubit wires. If None, the integers used to
            order the orbitals will be used as wire labels. Defaults to None.
        tol (float): tolerance for discarding the imaginary part of the coefficients

    Returns:
        Union[PauliSentence, Operator]: a linear combination of qubit operators
    """
    return _christiansen_mapping_dispatch(bose_operator, ps, wire_map, tol)


@singledispatch
def _christiansen_mapping_dispatch(bose_operator, ps, wire_map, tol):
    """Dispatches to appropriate function if bose_operator is a BoseWord or BoseSentence."""
    raise ValueError(f"bose_operator must be a BoseWord or BoseSentence, got: {bose_operator}")


@_christiansen_mapping_dispatch.register
def _(bose_operator: BoseWord, ps=False, wire_map=None, tol=None):
    wires = list(bose_operator.wires) or [0]
    identity_wire = wires[0]

    if len(bose_operator) == 0:
        qubit_operator = PauliSentence({PauliWord({}): 1.0})

    else:
        coeffs = {"+": -0.5j, "-": 0.5j}
        qubit_operator = PauliSentence({PauliWord({}): 1.0})  # Identity PS to multiply PSs with

        for item in bose_operator.items():
            (_, wire), sign = item

            # z_string = dict(zip(range(wire), ["Z"] * wire))
            z_string = {}
            qubit_operator @= PauliSentence(
                {
                    PauliWord({**z_string, **{wire: "X"}}): 0.5,
                    PauliWord({**z_string, **{wire: "Y"}}): coeffs[sign],
                }
            )

    for pw in qubit_operator:
        if tol is not None and abs(qml.math.imag(qubit_operator[pw])) <= tol:
            qubit_operator[pw] = qml.math.real(qubit_operator[pw])

    if not ps:
        # wire_order specifies wires to use for Identity (PauliWord({}))
        qubit_operator = qubit_operator.operation(wire_order=[identity_wire])

    if wire_map:
        return qubit_operator.map_wires(wire_map)

    return qubit_operator


@_christiansen_mapping_dispatch.register
def _(bose_operator: BoseSentence, ps=False, wire_map=None, tol=None):
    wires = list(bose_operator.wires) or [0]
    identity_wire = wires[0]

    qubit_operator = PauliSentence()  # Empty PS as 0 operator to add Pws to

    for fw, coeff in bose_operator.items():
        bose_word_as_ps = christiansen_mapping(fw, ps=True)

        for pw in bose_word_as_ps:
            qubit_operator[pw] = qubit_operator[pw] + bose_word_as_ps[pw] * coeff

            if tol is not None and abs(qml.math.imag(qubit_operator[pw])) <= tol:
                qubit_operator[pw] = qml.math.real(qubit_operator[pw])

    qubit_operator.simplify(tol=1e-16)

    if not ps:
        qubit_operator = qubit_operator.operation(wire_order=[identity_wire])

    if wire_map:
        return qubit_operator.map_wires(wire_map)

    return qubit_operator


def christiansen_bosonic(
    one, modes=None, modals=None, two=None, three=None, cutoff=1e-5, ordered=True
):
    r"""Build a vibrational observable in the Christiansen form (C-form) and map it
    to the Pauli basis

    Args:
        one (array): 3D array with one-body matrix elements
        modes (int): number of vibrational modes, detects from 'one' if none is provided
        modals (array): 1D array with the number of allowed vibrational modals for each mode, detects from 'one' if none is provided
        two (array): 6D array with two-body matrix elements
        three (array): 9D array with three-body matrix elements
        cutoff (float): magnitude beneath which terms are not incorporated in final expression
        ordered (bool): set True if matrix elements are ordered, i.e. two[m,n,::] = 0 for all n >= m and three[m,n,l,::] = 0 for all n >= m and l >= n

    Returns:
        tuple[int, Union[PauliSentence, Operator]]: the number of qubits and a linear combination of qubit operators
    """
    if modes is None:
        modes = np.shape(one)[0]

    if modals is None:
        imax = np.shape(one)[1]
        modals = imax * np.ones(modes, dtype=int)

    idx = {}  # dictionary mapping the tuple (l,n) to an index in the qubit register
    counter = 0
    for l in range(modes):
        for n in range(modals[l]):
            key = (l, n)
            idx[key] = counter
            counter += 1

    obs = {}  # second-quantized Hamiltonian

    # one-body terms
    for l in range(modes):
        for k_l in range(modals[l]):
            for h_l in range(modals[l]):
                (i0, i1) = ((l, k_l), (l, h_l))
                w = BoseWord({(0, idx[i0]): "+", (1, idx[i1]): "-"})
                obs[w] = one[l, k_l, h_l]

    # two-body terms
    if not two is None:
        for l in range(modes):
            if ordered is False:
                m_range = [p for p in range(modes) if p != l]
            else:
                m_range = range(l)
            for m in m_range:
                for k_l in range(modals[l]):
                    for h_l in range(modals[l]):
                        for k_m in range(modals[m]):
                            for h_m in range(modals[m]):
                                (i0, i1, i2, i3) = (
                                    (l, k_l),
                                    (m, k_m),
                                    (l, h_l),
                                    (m, h_m),
                                )
                                w = BoseWord(
                                    {
                                        (0, idx[i0]): "+",
                                        (1, idx[i1]): "+",
                                        (2, idx[i2]): "-",
                                        (3, idx[i3]): "-",
                                    }
                                )
                                obs[w] = two[l, m, k_l, k_m, h_l, h_m]

    # three-body terms
    if not three is None:
        for l in range(modes):
            if ordered is False:
                m_range = [p for p in range(modes) if p != l]
            else:
                m_range = range(l)
            for m in m_range:
                if ordered is False:
                    n_range = [p for p in range(modes) if p != l and p != m]
                else:
                    n_range = range(m)
                for n in n_range:
                    for k_l in range(modals[l]):
                        for h_l in range(modals[l]):
                            for k_m in range(modals[m]):
                                for h_m in range(modals[m]):
                                    for k_n in range(modals[n]):
                                        for h_n in range(modals[n]):
                                            (i0, i1, i2, i3, i4, i5) = (
                                                (l, k_l),
                                                (m, k_m),
                                                (n, k_n),
                                                (l, h_l),
                                                (m, h_m),
                                                (n, h_n),
                                            )
                                            w = BoseWord(
                                                {
                                                    (0, idx[i0]): "+",
                                                    (1, idx[i1]): "+",
                                                    (2, idx[i2]): "+",
                                                    (3, idx[i3]): "-",
                                                    (4, idx[i4]): "-",
                                                    (5, idx[i5]): "-",
                                                }
                                            )
                                            obs[w] = three[l, m, n, k_l, k_m, k_n, h_l, h_m, h_n]

    obs_sq = BoseSentence(obs)

    return obs_sq


def christiansen_hamiltonian(pes, nbos=16, do_cubic=False):

    h_arr = christiansen_integrals(pes, nbos=nbos, do_cubic=do_cubic)

    one = h_arr[0]
    two = h_arr[1]
    three = h_arr[2] if len(h_arr) == 3 else None
    cform_bosonic = christiansen_bosonic(one=one, two=two, three=three)
    cform_qubit = christiansen_mapping(cform_bosonic)

    return cform_qubit


def christiansen_dipole(pes, nbos=16, do_cubic=False):

    d_arr = christiansen_integrals_dipole(pes, nbos=nbos, do_cubic=do_cubic)

    one_x = d_arr[0][0,:,:,:]
    two_x = d_arr[1][0,:,:,:,:,:,:] if len(d_arr) > 1 else None
    three_x = d_arr[2][0,:,:,:,:,:,:,:,:,:] if len(d_arr)==3 else None
    cform_bosonic_x = christiansen_bosonic(one=one_x, two=two_x, three=three_x)
    print(cform_bosonic_x)
    cform_qubit_x = christiansen_mapping(cform_bosonic_x)

    one_y = d_arr[0][1,:,:,:]
    two_y = d_arr[1][1,:,:,:,:,:,:] if len(d_arr) > 1 else None
    three_y = d_arr[2][1,:,:,:,:,:,:,:,:,:] if len(d_arr)==3 else None
    cform_bosonic_y = christiansen_bosonic(one=one_y, two=two_y, three=three_y)
    cform_qubit_y = christiansen_mapping(cform_bosonic_y)

    one_z = d_arr[0][2,:,:,:]
    two_z = d_arr[1][2,:,:,:,:,:,:] if len(d_arr) > 1 else None
    three_z = d_arr[2][2,:,:,:,:,:,:,:,:,:] if len(d_arr)==3 else None
    cform_bosonic_z = christiansen_bosonic(one=one_z, two=two_z, three=three_z)
    cform_qubit_z = christiansen_mapping(cform_bosonic_z)

    
    return cform_qubit_x, cform_qubit_y, cform_qubit_z
