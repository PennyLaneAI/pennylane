# Copyright 2018-2022 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This module contains the functions for converting an external operator to a Pennylane operator.
"""
import warnings
from itertools import product

# pylint: disable=import-outside-toplevel
import pennylane as qml
from pennylane import numpy as np
from pennylane.operation import Tensor, active_new_opmath
from pennylane.pauli import pauli_sentence
from pennylane.wires import Wires


def _process_wires(wires, n_wires=None):
    r"""Checks and consolidates custom user wire mapping into a consistent, direction-free, ``Wires``
    format. Used for converting between OpenFermion qubit numbering and PennyLane wire labels.

    Since OpenFermion's qubit numbering is always consecutive int, simple iterable types such as
    list, tuple, or Wires can be used to specify the 2-way `qubit index` <-> `wire label` mapping
    with indices representing qubits. Dict can also be used as a mapping, but does not provide any
    advantage over lists other than the ability to do partial mapping/permutation in the
    `qubit index` -> `wire label` direction.

    It is recommended to pass Wires/list/tuple `wires` since it's direction-free, i.e. the same
    `wires` argument can be used to convert both ways between OpenFermion and PennyLane. Only use
    dict for partial or unordered mapping.

    Args:
        wires (Wires, list, tuple, dict): User wire labels.
            For types Wires, list, or tuple, each item in the iterable represents a wire label
            corresponding to the qubit number equal to its index.
            For type dict, only int-keyed dict (for qubit-to-wire conversion) or
            consecutive-int-valued dict (for wire-to-qubit conversion) is accepted.
            If None, will be set to consecutive int based on ``n_wires``.
        n_wires (int): Number of wires used if known. If None, will be inferred from ``wires``; if
            ``wires`` is not available, will be set to 1.

    Returns:
        Wires: Cleaned wire mapping with indices corresponding to qubits and values
            corresponding to wire labels.

    **Example**

    >>> # consec int wires if no wires mapping provided, ie. identity map: 0<->0, 1<->1, 2<->2
    >>> _process_wires(None, 3)
    <Wires = [0, 1, 2]>

    >>> # List as mapping, qubit indices with wire label values: 0<->w0, 1<->w1, 2<->w2
    >>> _process_wires(['w0','w1','w2'])
    <Wires = ['w0', 'w1', 'w2']>

    >>> # Wires as mapping, qubit indices with wire label values: 0<->w0, 1<->w1, 2<->w2
    >>> _process_wires(Wires(['w0', 'w1', 'w2']))
    <Wires = ['w0', 'w1', 'w2']>

    >>> # Dict as partial mapping, int qubits keys to wire label values: 0->w0, 1 unchanged, 2->w2
    >>> _process_wires({0:'w0',2:'w2'})
    <Wires = ['w0', 1, 'w2']>

    >>> # Dict as mapping, wires label keys to consec int qubit values: w2->2, w0->0, w1->1
    >>> _process_wires({'w2':2, 'w0':0, 'w1':1})
    <Wires = ['w0', 'w1', 'w2']>
    """

    # infer from wires, or assume 1 if wires is not of accepted types.
    if n_wires is None:
        n_wires = len(wires) if isinstance(wires, (Wires, list, tuple, dict)) else 1

    # defaults to no mapping.
    if wires is None:
        return Wires(range(n_wires))

    if isinstance(wires, (Wires, list, tuple)):
        # does not care about the tail if more wires are provided than n_wires.
        wires = Wires(wires[:n_wires])

    elif isinstance(wires, dict):
        if all(isinstance(w, int) for w in wires.keys()):
            # Assuming keys are taken from consecutive int wires. Allows for partial mapping.
            n_wires = max(wires) + 1
            labels = list(range(n_wires))  # used for completing potential partial mapping.
            for k, v in wires.items():
                if k < n_wires:
                    labels[k] = v
            wires = Wires(labels)
        elif set(range(n_wires)).issubset(set(wires.values())):
            # Assuming values are consecutive int wires (up to n_wires, ignores the rest).
            # Does NOT allow for partial mapping.
            wires = {v: k for k, v in wires.items()}  # flip for easy indexing
            wires = Wires([wires[i] for i in range(n_wires)])
        else:
            raise ValueError("Expected only int-keyed or consecutive int-valued dict for `wires`")

    else:
        raise ValueError(
            f"Expected type Wires, list, tuple, or dict for `wires`, got {type(wires)}"
        )

    if len(wires) != n_wires:
        # check length consistency when all checking and cleaning are done.
        raise ValueError(f"Length of `wires` ({len(wires)}) does not match `n_wires` ({n_wires})")

    return wires


def _openfermion_to_pennylane(qubit_operator, wires=None):
    r"""Convert OpenFermion ``QubitOperator`` to a 2-tuple of coefficients and
    PennyLane Pauli observables.

    Args:
        qubit_operator (QubitOperator): fermionic-to-qubit transformed operator in terms of
            Pauli matrices
        wires (Wires, list, tuple, dict): Custom wire mapping used to convert the qubit operator
            to an observable terms measurable in a PennyLane ansatz.
            For types Wires/list/tuple, each item in the iterable represents a wire label
            corresponding to the qubit number equal to its index.
            For type dict, only int-keyed dict (for qubit-to-wire conversion) is accepted.
            If None, will use identity map (e.g. 0->0, 1->1, ...).

    Returns:
        tuple[array[float], Iterable[pennylane.operation.Operator]]: coefficients and their
        corresponding PennyLane observables in the Pauli basis

    **Example**

    >>> q_op = 0.1*QubitOperator('X0') + 0.2*QubitOperator('Y0 Z2')
    >>> q_op
    0.1 [X0] +
    0.2 [Y0 Z2]
    >>> _openfermion_to_pennylane(q_op, wires=['w0','w1','w2','extra_wire'])
    (tensor([0.1, 0.2], requires_grad=False), [PauliX(wires=['w0']), PauliY(wires=['w0']) @ PauliZ(wires=['w2'])])

    If the new op-math is active, the list of operators will be cast as :class:`~.Prod` instances instead of
    :class:`~.Tensor` instances when appropriate.
    """
    n_wires = (
        1 + max(max(i for i, _ in t) if t else 1 for t in qubit_operator.terms)
        if qubit_operator.terms
        else 1
    )
    wires = _process_wires(wires, n_wires=n_wires)

    if not qubit_operator.terms:  # added since can't unpack empty zip to (coeffs, ops) below
        return np.array([0.0]), [qml.Identity(wires[0])]

    xyz2pauli = {"X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ}

    def _get_op(term, wires):
        """A function to compute the PL operator associated with the term string."""
        if len(term) > 1:
            if active_new_opmath():
                return qml.prod(*[xyz2pauli[op[1]](wires=wires[op[0]]) for op in term])

            return Tensor(*[xyz2pauli[op[1]](wires=wires[op[0]]) for op in term])

        if len(term) == 1:
            return xyz2pauli[term[0][1]](wires=wires[term[0][0]])

        return qml.Identity(wires[0])

    coeffs, ops = zip(
        *[(coef, _get_op(term, wires)) for term, coef in qubit_operator.terms.items()]
        # example term: ((0,'X'), (2,'Z'), (3,'Y'))
    )

    return np.real(np.array(coeffs, requires_grad=False)), list(ops)


def _ps_to_coeff_term(ps, wire_order):
    """Convert a non-empty pauli sentence to a list of coeffs and terms."""
    ops_str = []
    pws, coeffs = zip(*ps.items())

    for pw in pws:
        if len(pw) == 0:
            ops_str.append("")
            continue
        wires, ops = zip(*pw.items())
        ops_str.append(" ".join([f"{op}{wire_order.index(wire)}" for op, wire in zip(ops, wires)]))

    return coeffs, ops_str


def _pennylane_to_openfermion(coeffs, ops, wires=None):
    r"""Convert a 2-tuple of complex coefficients and PennyLane operations to
    OpenFermion ``QubitOperator``.

    Args:
        coeffs (array[complex]):
            coefficients for each observable, same length as ops
        ops (Iterable[pennylane.operation.Operations]): list of PennyLane operations that
            have a valid PauliSentence representation.
        wires (Wires, list, tuple, dict): Custom wire mapping used to convert to qubit operator
            from an observable terms measurable in a PennyLane ansatz.
            For types Wires/list/tuple, each item in the iterable represents a wire label
            corresponding to the qubit number equal to its index.
            For type dict, only consecutive-int-valued dict (for wire-to-qubit conversion) is
            accepted. If None, will map sorted wires from all `ops` to consecutive int.

    Returns:
        QubitOperator: an instance of OpenFermion's ``QubitOperator``.

    **Example**

    >>> coeffs = np.array([0.1, 0.2, 0.3, 0.4])
    >>> ops = [
    ...     qml.operation.Tensor(qml.PauliX(wires=['w0'])),
    ...     qml.operation.Tensor(qml.PauliY(wires=['w0']), qml.PauliZ(wires=['w2'])),
    ...     qml.sum(qml.PauliZ(wires=['w0']), qml.s_prod(-0.5, qml.PauliX(wires=['w0']))),
    ...     qml.prod(qml.PauliX(wires=['w0']), qml.PauliZ(wires=['w1'])),
    ... ]
    >>> _pennylane_to_openfermion(coeffs, ops, wires=Wires(['w0', 'w1', 'w2']))
    (-0.05+0j) [X0] +
    (0.4+0j) [X0 Z1] +
    (0.2+0j) [Y0 Z2] +
    (0.3+0j) [Z0]
    """
    try:
        import openfermion
    except ImportError as Error:
        raise ImportError(
            "This feature requires openfermion. "
            "It can be installed with: pip install openfermion"
        ) from Error

    all_wires = Wires.all_wires([op.wires for op in ops], sort=True)

    if wires is not None:
        qubit_indexed_wires = _process_wires(
            wires,
        )
        if not set(all_wires).issubset(set(qubit_indexed_wires)):
            raise ValueError("Supplied `wires` does not cover all wires defined in `ops`.")
    else:
        qubit_indexed_wires = all_wires

    q_op = openfermion.QubitOperator()
    for coeff, op in zip(coeffs, ops):
        if isinstance(op, Tensor):
            try:
                ps = pauli_sentence(op)
            except ValueError as e:
                raise ValueError(
                    f"Expected a Pennylane operator with a valid Pauli word representation, "
                    f"but got {op}."
                ) from e

        elif (ps := op._pauli_rep) is None:  # pylint: disable=protected-access
            raise ValueError(
                f"Expected a Pennylane operator with a valid Pauli word representation, but got {op}."
            )

        if len(ps) > 0:
            sub_coeffs, op_strs = _ps_to_coeff_term(ps, wire_order=qubit_indexed_wires)
            for c, op_str in zip(sub_coeffs, op_strs):
                # This is how one makes QubitOperator in OpenFermion
                q_op += complex(coeff * c) * openfermion.QubitOperator(op_str)

    return q_op


def _openfermion_pennylane_equivalent(
    openfermion_qubit_operator, pennylane_qubit_operator, wires=None
):
    r"""Check equivalence between OpenFermion :class:`~.QubitOperator` and Pennylane VQE
    ``Hamiltonian`` (Tensor product of Pauli matrices).

    Equality is based on OpenFermion :class:`~.QubitOperator`'s equality.

    Args:
        openfermion_qubit_operator (QubitOperator): OpenFermion qubit operator represented as
            a Pauli summation
        pennylane_qubit_operator (pennylane.Hamiltonian): PennyLane
            Hamiltonian object
        wires (Wires, list, tuple, dict): Custom wire mapping used to convert to qubit operator
            from an observable terms measurable in a PennyLane ansatz.
            For types Wires/list/tuple, each item in the iterable represents a wire label
            corresponding to the qubit number equal to its index.
            For type dict, only int-keyed dict (for qubit-to-wire conversion) is accepted.
            If None, will map sorted wires from all Pennylane terms to consecutive int.

    Returns:
        (bool): True if equivalent
    """
    coeffs, ops = pennylane_qubit_operator.terms()
    return openfermion_qubit_operator == _pennylane_to_openfermion(coeffs, ops, wires=wires)


def import_operator(qubit_observable, format="openfermion", wires=None, tol=1e010):
    r"""Convert an external operator to a PennyLane operator.

    The external format currently supported is openfermion.

    Args:
        qubit_observable: external qubit operator that will be converted
        format (str): the format of the operator object to convert from
        wires (.Wires, list, tuple, dict): Custom wire mapping used to convert the external qubit
            operator to a PennyLane operator.
            For types ``Wires``/list/tuple, each item in the iterable represents a wire label
            for the corresponding qubit index.
            For type dict, only int-keyed dictionaries (for qubit-to-wire conversion) are accepted.
            If ``None``, the identity map (e.g., ``0->0, 1->1, ...``) will be used.
        tol (float): Tolerance in `machine epsilon <https://numpy.org/doc/stable/reference/generated/numpy.real_if_close.html>`_
            for the imaginary part of the coefficients in ``qubit_observable``.
            Coefficients with imaginary part less than 2.22e-16*tol are considered to be real.

    Returns:
        (.Operator): PennyLane operator representing any operator expressed as linear combinations of
        Pauli words, e.g.,
        :math:`\sum_{k=0}^{N-1} c_k O_k`

    **Example**

    >>> from openfermion import QubitOperator
    >>> h_of = QubitOperator('X0 X1 Y2 Y3', -0.0548) + QubitOperator('Z0 Z1', 0.14297)
    >>> h_pl = import_operator(h_of, format='openfermion')
    >>> print(h_pl)
    (0.14297) [Z0 Z1]
    + (-0.0548) [X0 X1 Y2 Y3]

    If the new op-math is active, an arithmetic operator is returned instead.

    >>> qml.operation.enable_new_opmath()
    >>> h_pl = import_operator(h_of, format='openfermion')
    >>> print(h_pl)
    (-0.0548*(PauliX(wires=[0]) @ PauliX(wires=[1]) @ PauliY(wires=[2]) @ PauliY(wires=[3]))) + (0.14297*(PauliZ(wires=[0]) @ PauliZ(wires=[1])))
    """
    if format not in ["openfermion"]:
        raise TypeError(f"Converter does not exist for {format} format.")

    coeffs = np.array([np.real_if_close(coef, tol=tol) for coef in qubit_observable.terms.values()])

    if any(np.iscomplex(coeffs)):
        warnings.warn(
            f"The coefficients entering the QubitOperator must be real;"
            f" got complex coefficients in the operator"
            f" {list(coeffs[np.iscomplex(coeffs)])}"
        )

    if active_new_opmath():
        return qml.dot(*_openfermion_to_pennylane(qubit_observable, wires=wires))

    return qml.Hamiltonian(*_openfermion_to_pennylane(qubit_observable, wires=wires))


def _excitations(electrons, orbitals):
    r"""Generate all possible single and double excitations from a Hartree-Fock reference state.

    This function is a more performant version of `qchem.excitations`, where the order of the
    generated excitations is consistent with PySCF.

    Single and double excitations can be generated by acting with the operators
    :math:`\hat T_1` and :math:`\hat T_2` on the Hartree-Fock reference state:

    .. math::

        && \hat{T}_1 = \sum_{p \in \mathrm{occ} \\ q \in \mathrm{unocc}}
        \hat{c}_q^\dagger \hat{c}_p \\
        && \hat{T}_2 = \sum_{p>q \in \mathrm{occ} \\ r>s \in
        \mathrm{unocc}} \hat{c}_r^\dagger \hat{c}_s^\dagger \hat{c}_p \hat{c}_q.


    In the equations above the indices :math:`p, q, r, s` run over the
    occupied (occ) and unoccupied (unocc) *spin* orbitals and :math:`\hat c` and
    :math:`\hat c^\dagger` are the electron annihilation and creation operators,
    respectively.

    Args:
        electrons (int): number of electrons
        orbitals (int): number of spin orbitals

    Returns:
        tuple(list, list): lists with the indices of the spin orbitals involved in the excitations

    **Example**

    >>> electrons = 2
    >>> orbitals = 4
    >>> _excitations(electrons, orbitals)
    ([[[0, 2]], [[0, 3]], [[1, 2]], [[1, 3]]], [[0, 1, 2, 3]])
    """
    singles_p, singles_q = [], []
    doubles_pq, doubles_rs = [], []

    for i in range(electrons):
        singles_p += [i]
        doubles_pq += [[k, i] for k in range(i)]
    for j in range(electrons, orbitals):
        singles_q += [j]
        doubles_rs += [[k, j] for k in range(electrons, j)]

    singles = [[p] + [q] for p in singles_p for q in singles_q]
    doubles = [pq + rs for pq in doubles_pq for rs in doubles_rs]

    return singles, doubles


def _excited_configurations(electrons, orbitals, excitation):
    r"""Generate excited states from a Hartree-Fock reference state.

    This function generates excited states in the form of binary strings or integers representing
    them, from a Hartree-Fock (HF) reference state. The HF state is assumed to be
    :math:`|1 1 ...1 0 ... 0 0 \rangle` where the number of :math:`1` and :math:`0` elements
    represents the number of occupied and unoccupied spin orbitals, respectively. The string
    representation of the state is obtained by converting the occupation-number vector to a string,
    e.g., ``111000`` represents :math:`|1 1 1 0 0 0 \rangle. The number representation of the state
    is the integer corresonding to the binary number, e.g., ``111000`` is represented by
    :math:`int('111000', 2) = 56`.

    Args:
        electrons (int): number of electrons
        orbitals (int): number of spin orbitals
        excitation (int): number of excited electrons

    Returns:
        tuple(array, array): arrays of excited states and signs obtained by reordering of the creation operators

    **Example**

    >>> electrons = 2
    >>> orbitals = 5
    >>> excitation = 2
    >>> _excitated_states(electrons, orbitals, excitation)
    (array([28, 26, 25]), array([ 1, -1,  1]))
    """
    hf_state = qml.qchem.hf_state(electrons, orbitals)
    singles, doubles = _excitations(electrons, orbitals)
    states, signs = [], []

    if excitation == 1:
        for s in singles:
            state = hf_state.copy()
            state[s] = state[s[::-1]]
            states += [state]
            signs.append((-1) ** len(range(s[0], electrons - 1)))

    if excitation == 2:
        for d in doubles:
            state = hf_state.copy()
            state[d] = state[[d[2], d[3], d[0], d[1]]]
            states += [state]
            order_pq = len(range(d[0], electrons - 1))
            order_rs = len(range(d[1], electrons - 1))
            signs.append((-1) ** (order_pq + order_rs + 1))

    states_str = ["".join([str(i) for i in state]) for state in states]
    states_int = [int(state[::-1], 2) for state in states_str]

    return states_int, signs


def _ucisd_state(cisd_solver, tol=1e-15):
    r"""Construct a wavefunction from PySCF's `UCISD` Solver object.

    The wavefunction is represented as a dictionary where the keys are tuples representing a
    configuration, which corresponds to a Slater determinant, and the values are the CI coefficient
    corresponding to that Slater determinant. Each dictionary key is a tuple of two integers. The
    binary representation of these integers correspond to a specific configuration, or Slater
    determinant. The first number represents the configuration of alpha electrons and the second
    number represents the beta electrons. For instance, the Hartree-Fock state
    :math:`|1 1 0 0 \rangle` will be represented by the binary string `0011`. This string can be
    splited to `01` and `01` for the alpha and beta electrons. The integer corresponding to `01` is
    `1`. Then the dictionary representation of a state with `0.99` contribution of the Hartree-Fock
    state and `0.01` contribution from the doubly-excited state will be
    `{(1, 1): 0.99, (2, 2): 0.01}`.

    Args:
        cisd_solver (PySCF UCISD Class instance): the class object representing the UCISD calculation in PySCF
        tol (float):  the tolerance to which the wavefunction is being built -- Slater determinants
         with coefficients smaller than this are discarded. Default is 1e-15 (all coefficients are included).

    Returns:
        dict: Dictionary of the form `{(int_a, int_b) :coeff}`, with integers `int_a, int_b`
        having binary represention corresponding to the Fock occupation vector in alpha and beta
        spin sectors, respectively, and coeff being the CI coefficients of those configurations.

    **Example**

    >>> from pyscf import gto, scf, ci
    >>> mol = gto.M(atom=[['H', (0, 0, 0)], ['H', (0,0,0.71)]], basis='sto6g')
    >>> myhf = scf.UHF(mol).run()
    >>> myci = ci.UCISD(myhf).run()
    >>> wf_cisd = cisd_state(myci, tol=1e-1)
    >>> print(wf_cisd)
    {(1, 1): -0.9942969785398778, (2, 2): 0.10664669927602159}
    """
    mol = cisd_solver.mol
    cisdvec = cisd_solver.ci

    norb = mol.nao
    nelec = mol.nelectron
    nelec_a = int((nelec + mol.spin) / 2)
    nelec_b = int((nelec - mol.spin) / 2)

    nvir_a, nvir_b = norb - nelec_a, norb - nelec_b

    size_a, size_b = nelec_a * nvir_a, nelec_b * nvir_b
    size_aa = int(nelec_a * (nelec_a - 1) / 2) * int(nvir_a * (nvir_a - 1) / 2)
    size_bb = int(nelec_b * (nelec_b - 1) / 2) * int(nvir_b * (nvir_b - 1) / 2)
    size_ab = nelec_a * nelec_b * nvir_a * nvir_b

    sizes = [1, size_a, size_b, size_aa, size_ab, size_bb]
    cumul = np.cumsum(sizes)
    idxs = [0] + [slice(cumul[ii], cumul[ii + 1]) for ii in range(len(cumul) - 1)]
    c0, c1a, c1b, c2aa, c2ab, c2bb = [cisdvec[idx] for idx in idxs]

    # numbers representing the Hartree-Fock vector, e.g., bin(ref_a)[::-1] = 1111...10...0
    ref_a = int(2**nelec_a - 1)
    ref_b = int(2**nelec_b - 1)

    dict_fcimatr = dict(zip(list(zip([ref_a], [ref_b])), [c0]))

    # alpha -> alpha excitations
    c1a_configs, c1a_signs = _excited_configurations(nelec_a, norb, 1)
    dict_fcimatr.update(dict(zip(list(zip(c1a_configs, [ref_b] * size_a)), c1a * c1a_signs)))

    # beta -> beta excitations
    c1b_configs, c1b_signs = _excited_configurations(nelec_b, norb, 1)
    dict_fcimatr.update(dict(zip(list(zip([ref_a] * size_b, c1b_configs)), c1b * c1b_signs)))

    # alpha, alpha -> alpha, alpha excitations
    c2aa_configs, c2aa_signs = _excited_configurations(nelec_a, norb, 2)
    dict_fcimatr.update(dict(zip(list(zip(c2aa_configs, [ref_b] * size_aa)), c2aa * c2aa_signs)))

    # alpha, beta -> alpha, beta excitations
    rowvals, colvals = np.array(list(product(c1a_configs, c1b_configs)), dtype=int).T.numpy()
    dict_fcimatr.update(
        dict(zip(list(zip(rowvals, colvals)), c2ab * np.kron(c1a_signs, c1b_signs).numpy()))
    )

    # beta, beta -> beta, beta excitations
    c2bb_configs, c2bb_signs = _excited_configurations(nelec_b, norb, 2)
    dict_fcimatr.update(dict(zip(list(zip([ref_a] * size_bb, c2bb_configs)), c2bb * c2bb_signs)))

    # filter based on tolerance cutoff
    dict_fcimatr = {key: value for key, value in dict_fcimatr.items() if abs(value) > tol}

    return dict_fcimatr


def cisd_state(cisd_solver, hftype, tol=1e-15):
    r"""Construct a wavefunction from PySCF output.

    Args:
        cisd_solver (PySCF CISD or UCISD Class instance): the class object representing the
         CISD / UCISD calculation in PySCF

        hftype (str): keyword specifying whether restricted or unrestricted CISD was performed.
            The options are 'rhf' for restricted, and 'uhf' for unrestricted.

        tol (float): the tolerance to which the wavefunction is being built -- Slater
            determinants with coefficients smaller than this are discarded. Default
            is 1e-15 (all coefficients are included).

    Returns:
        dict: Dictionary of the form `{(int_a, int_b) :coeff}`, with integers `int_a, int_b`
        having binary represention corresponding to the Fock occupation vector in alpha and beta
        spin sectors, respectively, and coeff being the CI coefficients of those configurations.

    **Example**

    >>> from pyscf import gto, scf, ci
    >>> mol = gto.M(atom=[['H', (0, 0, 0)], ['H', (0,0,0.71)]], basis='sto6g')
    >>> myhf = scf.UHF(mol).run()
    >>> myci = ci.UCISD(myhf).run()
    >>> wf_cisd = cisd_state(myci, tol=1e-1)
    >>> print(wf_cisd)
    {(1, 1): -0.9942969785398778, (2, 2): 0.10664669927602159}
    """

    if hftype == "uhf":
        wf_dict = _ucisd_state(cisd_solver, tol=tol)
    else:
        raise ValueError(
            "The supported hftype options are 'rhf' for restricted and 'uhf' for unrestricted."
        )
    wf = wfdict_to_statevector(wf_dict, cisd_solver.mol.nao)

    return wf


def wfdict_to_statevector(wf_dict, norbs):
    r"""Convert a wavefunction in sparce dictionary format to a Pennylane's statevector.

    Args:
        wf_dict (wf_dict format): dictionary with keys (int_a,int_b) being integers whose binary representation
            shows the Fock occupation vector for alpha and beta electrons; and values being the CI
            coefficients.

        norbs (int): number of molecular orbitals in the system

    Returns:
        statevector (list): normalized statevector of length 2^(number_of_spinorbitals), standard in Pennylane

    """

    statevector = np.zeros(2 ** (2 * norbs), dtype=complex)

    for (int_a, int_b), coeff in wf_dict.items():
        bin_a, bin_b = bin(int_a)[2:], bin(int_b)[2:]
        bin_a, bin_b = bin_a[::-1], bin_b[::-1]
        bin_tot = "".join(i + j for i, j in zip(bin_a, bin_b))
        bin_tot_full = bin_tot + "0" * (2 * norbs - len(bin_tot))
        idx = int(bin_tot_full, 2)
        statevector[idx] += coeff

    statevector = statevector / np.sqrt(np.sum(statevector**2))

    return statevector
