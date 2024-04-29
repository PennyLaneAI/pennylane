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
import numpy as np

# pylint: disable= import-outside-toplevel,no-member,too-many-function-args
import pennylane as qml
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

    >>> q_op = 0.1 * QubitOperator('X0') + 0.2 * QubitOperator('Y0 Z2')
    >>> q_op
    0.1 [X0] +
    0.2 [Y0 Z2]
    >>> _openfermion_to_pennylane(q_op, wires=['w0','w1','w2','extra_wire'])
    (tensor([0.1, 0.2], requires_grad=False), [X('w0'), Y('w0') @ Z('w2')])

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

    xyz2pauli = {"X": qml.X, "Y": qml.Y, "Z": qml.Z}

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

    return np.array(coeffs).real, list(ops)


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
    ...     qml.operation.Tensor(qml.X('w0')),
    ...     qml.operation.Tensor(qml.Y('w0'), qml.Z('w2')),
    ...     qml.sum(qml.Z('w0'), qml.s_prod(-0.5, qml.X('w0'))),
    ...     qml.prod(qml.X('w0'), qml.Z('w1')),
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

        elif (ps := op.pauli_rep) is None:
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

    We currently support `OpenFermion <https://quantumai.google/openfermion>`__ operators: the function accepts most types of
    OpenFermion qubit operators, such as those corresponding to Pauli words and sums of Pauli words.

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
            Coefficients with imaginary part less than :math:`(2.22 \cdot 10^{-16}) \cdot \text{tol}` are considered to be real.

    Returns:
        (.Operator): PennyLane operator representing any operator expressed as linear combinations of
        Pauli words, e.g.,
        :math:`\sum_{k=0}^{N-1} c_k O_k`

    **Example**

    >>> assert qml.operation.active_new_opmath() == True
    >>> h_pl = import_operator(h_of, format='openfermion')
    >>> print(h_pl)
    (-0.0548 * X(0 @ X(1) @ Y(2) @ Y(3))) + (0.14297 * Z(0 @ Z(1)))

    If the new op-math is deactivated, a :class:`~Hamiltonian` is returned instead.

    >>> assert qml.operation.active_new_opmath() == False
    >>> from openfermion import QubitOperator
    >>> h_of = QubitOperator('X0 X1 Y2 Y3', -0.0548) + QubitOperator('Z0 Z1', 0.14297)
    >>> h_pl = import_operator(h_of, format='openfermion')
    >>> print(h_pl)
    (0.14297) [Z0 Z1]
    + (-0.0548) [X0 X1 Y2 Y3]
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

    This function is a more performant version of ``qchem.excitations``, where the order of the
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
    ([[0, 2], [0, 3], [1, 2], [1, 3]], [[0, 1, 2, 3]])
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
    r"""Generate excited configurations from a Hartree-Fock reference state.

    This function generates excited configurations in the form of integers representing a binary
    string, e.g., :math:`|1 1 0 1 0 0 \rangle` is represented by :math:`int('110100', 2) = 52`.

    The excited configurations are generated from a Hartree-Fock (HF) reference state. The HF state
    is assumed to have the form :math:`|1 1 ...1 0 ... 0 0 \rangle` where the number of :math:`1`
    and :math:`0` elements are the number of occupied and unoccupied spin orbitals, respectively.
    The string representation of the state is obtained by converting the occupation-number vector to
    a string, e.g., ``111000`` to represent :math:`|1 1 1 0 0 0 \rangle.

    Each excited configuration has a sign, :math:`+1` or :math:`-1`, that is obtained by reordering
    the creation operators.

    Args:
        electrons (int): number of electrons
        orbitals (int): number of spin orbitals
        excitation (int): number of excited electrons

    Returns:
        tuple(list, list): lists of excited configurations and signs obtained by reordering the
         creation operators

    **Example**

    >>> electrons = 3
    >>> orbitals = 5
    >>> excitation = 2
    >>> _excited_configurations(electrons, orbitals, excitation)
    ([28, 26, 25], [1, -1, 1])
    """
    if excitation not in [1, 2]:
        raise ValueError(
            "Only single (excitation = 1) and double (excitation = 2) excitations are supported."
        )

    hf_state = qml.qchem.hf_state(electrons, orbitals)
    singles, doubles = _excitations(electrons, orbitals)
    states, signs = [], []

    if excitation == 1:
        for s in singles:
            state = hf_state.copy()
            state[s] = state[[s[1], s[0]]]  # apply single excitation
            states += [state]
            signs.append((-1) ** len(range(s[0], electrons - 1)))

    if excitation == 2:
        for d in doubles:
            state = hf_state.copy()
            state[d] = state[[d[2], d[3], d[0], d[1]]]  # apply double excitation
            states += [state]
            order_pq = len(range(d[0], electrons - 1))
            order_rs = len(range(d[1], electrons - 1))
            signs.append((-1) ** (order_pq + order_rs + 1))

    states_str = ["".join([str(i) for i in state]) for state in states]
    states_int = [int(state[::-1], 2) for state in states_str]

    return states_int, signs


def import_state(solver, tol=1e-15):
    r"""Convert an external wavefunction to a state vector.

    The sources of wavefunctions that are currently accepted are listed below.

        * The PySCF library (the restricted and unrestricted CISD/CCSD
          methods are supported). The `solver` argument is then the associated PySCF CISD/CCSD Solver object.
        * The library Dice implementing the SHCI method. The `solver` argument is then the tuple(list[str], array[float]) of Slater determinants and their coefficients.
        * The library Block2 implementing the DMRG method. The `solver` argument is then the tuple(list[int], array[float]) of Slater determinants and their coefficients.

    Args:
        solver: external wavefunction object
        tol (float): the tolerance for discarding Slater determinants based on their coefficients

    Raises:
        ValueError: if external object type is not supported

    Returns:
        array: normalized state vector of length :math:`2^M`, where :math:`M` is the number of spin orbitals

    **Example**

    >>> from pyscf import gto, scf, ci
    >>> mol = gto.M(atom=[['H', (0, 0, 0)], ['H', (0,0,0.71)]], basis='sto6g')
    >>> myhf = scf.UHF(mol).run()
    >>> myci = ci.UCISD(myhf).run()
    >>> wf_cisd = qml.qchem.import_state(myci, tol=1e-1)
    >>> print(wf_cisd)
    [ 0.        +0.j  0.        +0.j  0.        +0.j  0.1066467 +0.j
      0.        +0.j  0.        +0.j  0.        +0.j  0.        +0.j
      0.        +0.j  0.        +0.j  0.        +0.j  0.        +0.j
     -0.99429698+0.j  0.        +0.j  0.        +0.j  0.        +0.j]
    """

    method = str(solver.__str__)

    if "CISD" in method and "UCISD" not in method:
        wf_dict = _rcisd_state(solver, tol=tol)
    elif "UCISD" in method:
        wf_dict = _ucisd_state(solver, tol=tol)
    elif "CCSD" in method and "UCCSD" not in method:
        wf_dict = _rccsd_state(solver, tol=tol)
    elif "UCCSD" in method:
        wf_dict = _uccsd_state(solver, tol=tol)
    elif "tuple" in method and len(solver) == 2:
        if isinstance(solver[0][0], str):
            wf_dict = _shci_state(solver, tol=tol)
        elif isinstance(solver[0][0][0], int):
            wf_dict = _dmrg_state(solver, tol=tol)
        else:
            raise ValueError(
                "For tuple input, the supported objects are"
                " tuple(list[str], array[float]) for SHCI calculations with Dice library and "
                "tuple(list[int], array[float]) for DMRG calculations with the Block2 library."
            )
    else:
        raise ValueError(
            "The supported objects are RCISD, UCISD, RCCSD, and UCCSD for restricted and"
            " unrestricted configuration interaction and coupled cluster calculations, and"
            " tuple(list[str], array[float]) for SHCI calculations with Dice library and "
            "tuple(list[int], array[float]) for DMRG calculations with the Block2 library."
        )
    if "tuple" in method:
        wf = _wfdict_to_statevector(wf_dict, len(solver[0][0]))
    else:
        wf = _wfdict_to_statevector(wf_dict, solver.mol.nao)

    return wf


def _wfdict_to_statevector(fcimatr_dict, norbs):
    r"""Convert a wavefunction in sparse dictionary format to a PennyLane statevector.

    In the sparse dictionary format, the keys ``(int_a, int_b)`` are integers whose binary
    representation shows the Fock occupation vector for alpha and beta electrons and values are the
    CI coefficients.

    Args:
        fcimatr_dict (dict[tuple(int,int),float]): the sparse dictionary format of a wavefunction
        norbs (int): number of molecular orbitals

    Returns:
        array: normalized state vector of length :math:`2^M`, where :math:`M` is the number of spin orbitals
    """
    statevector = np.zeros(2 ** (2 * norbs), dtype=complex)

    for (int_a, int_b), coeff in fcimatr_dict.items():
        bin_a = bin(int_a)[2:][::-1]
        bin_b = bin(int_b)[2:][::-1]
        bin_a += "0" * (norbs - len(bin_a))
        bin_b += "0" * (norbs - len(bin_b))
        bin_ab = "".join(i + j for i, j in zip(bin_a, bin_b))
        statevector[int(bin_ab, 2)] += coeff

    statevector = statevector / np.sqrt(np.sum(statevector**2))

    return statevector


def _rcisd_state(cisd_solver, tol=1e-15):
    r"""Construct a wavefunction from PySCF's ``RCISD`` solver object.

    The generated wavefunction is a dictionary where the keys represent a configuration, which
    corresponds to a Slater determinant, and the values are the CI coefficients of the Slater
    determinant. Each dictionary key is a tuple of two integers. The binary representation of these
    integers correspond to a specific configuration: the first number represents the
    configuration of the alpha electrons and the second number represents the configuration of the
    beta electrons. For instance, the Hartree-Fock state :math:`|1 1 0 0 \rangle` will be
    represented by the flipped binary string ``0011`` which is split to ``01`` and ``01`` for
    the alpha and beta electrons. The integer corresponding to ``01`` is ``1`` and the dictionary
    representation of the Hartree-Fock state will be ``{(1, 1): 1.0}``. The dictionary
    representation of a state with ``0.99`` contribution from the Hartree-Fock state and ``0.01``
    contribution from the doubly-excited state, i.e., :math:`|0 0 1 1 \rangle`, will be
    ``{(1, 1): 0.99, (2, 2): 0.01}``.

    Args:
        cisd_solver (PySCF CISD Class instance): the class object representing the CISD calculation in PySCF
        tol (float): the tolerance for discarding Slater determinants based on their coefficients

    Returns:
        fcimatr_dict (dict[tuple(int,int),float]): dictionary of the form ``{(int_a, int_b) :coeff}``, with integers ``int_a, int_b``
        having binary representation corresponding to the Fock occupation vector in alpha and beta
        spin sectors, respectively, and coeff being the CI coefficients of those configurations

    **Example**

    >>> from pyscf import gto, scf, ci
    >>> mol = gto.M(atom=[['H', (0, 0, 0)], ['H', (0,0,0.71)]], basis='sto6g', symmetry='d2h')
    >>> myhf = scf.RHF(mol).run()
    >>> myci = ci.CISD(myhf).run()
    >>> wf_cisd = _rcisd_state(myci, tol=1e-1)
    >>> print(wf_cisd)
    {(1, 1): -0.9942969785398775, (2, 2): 0.10664669927602162}
    """
    mol = cisd_solver.mol
    cisdvec = cisd_solver.ci

    norb = mol.nao
    nelec = mol.nelectron
    nocc, nvir = nelec // 2, norb - nelec // 2

    c0, c1, c2 = (
        cisdvec[0],
        cisdvec[1 : nocc * nvir + 1],
        cisdvec[nocc * nvir + 1 :].reshape(nocc, nocc, nvir, nvir),
    )
    c2ab = c2.transpose(0, 2, 1, 3).reshape(nocc * nvir, -1)

    # numbers representing the Hartree-Fock vector, e.g., bin(ref_a)[::-1] = 1111...10...0
    ref_a = int(2**nocc - 1)
    ref_b = ref_a

    fcimatr_dict = dict(zip(list(zip([ref_a], [ref_b])), [c0]))

    # alpha -> alpha excitations
    c1a_configs, c1a_signs = _excited_configurations(nocc, norb, 1)
    fcimatr_dict.update(
        dict(zip(list(zip(c1a_configs, [ref_b] * len(c1a_configs))), c1 * c1a_signs))
    )
    # beta -> beta excitations
    fcimatr_dict.update(
        dict(zip(list(zip([ref_a] * len(c1a_configs), c1a_configs)), c1 * c1a_signs))
    )

    # check if double excitations within one spin sector (aa->aa and bb->bb) are possible
    if nocc > 1 and nvir > 1:
        # get rid of excitations from identical orbitals, double-count the allowed ones
        c2_tr = c2 - c2.transpose(1, 0, 2, 3)
        # select only unique excitations, via lower triangle of matrix (already double-counted)
        ooidx, vvidx = np.tril_indices(nocc, -1), np.tril_indices(nvir, -1)
        c2aa = c2_tr[ooidx][:, vvidx[0], vvidx[1]].ravel()

        # alpha, alpha -> alpha, alpha excitations
        c2aa_configs, c2aa_signs = _excited_configurations(nocc, norb, 2)
        fcimatr_dict.update(
            dict(zip(list(zip(c2aa_configs, [ref_b] * len(c2aa_configs))), c2aa * c2aa_signs))
        )
        # beta, beta -> beta, beta excitations
        fcimatr_dict.update(
            dict(zip(list(zip([ref_a] * len(c2aa_configs), c2aa_configs)), c2aa * c2aa_signs))
        )

    # alpha, beta -> alpha, beta excitations
    # generate all possible pairwise combinations of _single_ excitations of alpha and beta sectors
    fcimatr_dict.update(
        dict(
            zip(
                list(product(c1a_configs, c1a_configs)),
                np.einsum("i,j,ij->ij", c1a_signs, c1a_signs, c2ab, optimize=True).ravel(),
            )
        )
    )

    # filter based on tolerance cutoff
    fcimatr_dict = {key: value for key, value in fcimatr_dict.items() if abs(value) > tol}

    # convert sign parity from chemist to physicist convention (interleaving spin operators
    # rather than commuting all spin-up operators to the left)
    fcimatr_dict = _sign_chem_to_phys(fcimatr_dict, norb)

    return fcimatr_dict


def _ucisd_state(cisd_solver, tol=1e-15):
    r"""Construct a wavefunction from PySCF's ``UCISD`` solver object.

    The generated wavefunction is a dictionary where the keys represent a configuration, which
    corresponds to a Slater determinant, and the values are the CI coefficients of the Slater
    determinant. Each dictionary key is a tuple of two integers. The binary representation of these
    integers correspond to a specific configuration: the first number represents the
    configuration of the alpha electrons and the second number represents the configuration of the
    beta electrons. For instance, the Hartree-Fock state :math:`|1 1 0 0 \rangle` will be
    represented by the flipped binary string ``0011`` which is split to ``01`` and ``01`` for
    the alpha and beta electrons. The integer corresponding to ``01`` is ``1`` and the dictionary
    representation of the Hartree-Fock state will be ``{(1, 1): 1.0}``. The dictionary
    representation of a state with `0.99` contribution from the Hartree-Fock state and ``0.01``
    contribution from the doubly-excited state, i.e., :math:`|0 0 1 1 \rangle`, will be
    ``{(1, 1): 0.99, (2, 2): 0.01}``.

    Args:
        cisd_solver (PySCF UCISD Class instance): the class object representing the UCISD calculation in PySCF
        tol (float): the tolerance for discarding Slater determinants based on their coefficients

    Returns:
        fcimatr_dict (dict[tuple(int,int),float]): dictionary of the form ``{(int_a, int_b) :coeff}``, with integers ``int_a, int_b``
        having binary representation corresponding to the Fock occupation vector in alpha and beta
        spin sectors, respectively, and coeff being the CI coefficients of those configurations

    **Example**

    >>> from pyscf import gto, scf, ci
    >>> mol = gto.M(atom=[['H', (0, 0, 0)], ['H', (0,0,0.71)]], basis='sto6g', symmetry='d2h')
    >>> myhf = scf.UHF(mol).run()
    >>> myci = ci.UCISD(myhf).run()
    >>> wf_cisd = _ucisd_state(myci, tol=1e-1)
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
    size_ab = size_a * size_b

    cumul = np.cumsum([0, 1, size_a, size_b, size_ab, size_aa, size_bb])
    c0, c1a, c1b, c2ab, c2aa, c2bb = [
        cisdvec[cumul[idx] : cumul[idx + 1]] for idx in range(len(cumul) - 1)
    ]
    c2ab = (
        c2ab.reshape(nelec_a, nelec_b, nvir_a, nvir_b)
        .transpose(0, 2, 1, 3)
        .reshape(nelec_a * nvir_a, -1)
    )

    # numbers representing the Hartree-Fock vector, e.g., bin(ref_a)[::-1] = 1111...10...0
    ref_a = int(2**nelec_a - 1)
    ref_b = int(2**nelec_b - 1)

    fcimatr_dict = dict(zip(list(zip([ref_a], [ref_b])), c0))

    # alpha -> alpha excitations
    c1a_configs, c1a_signs = _excited_configurations(nelec_a, norb, 1)
    fcimatr_dict.update(dict(zip(list(zip(c1a_configs, [ref_b] * size_a)), c1a * c1a_signs)))

    # beta -> beta excitations
    c1b_configs, c1b_signs = _excited_configurations(nelec_b, norb, 1)
    fcimatr_dict.update(dict(zip(list(zip([ref_a] * size_b, c1b_configs)), c1b * c1b_signs)))

    # alpha, alpha -> alpha, alpha excitations
    c2aa_configs, c2aa_signs = _excited_configurations(nelec_a, norb, 2)
    fcimatr_dict.update(dict(zip(list(zip(c2aa_configs, [ref_b] * size_aa)), c2aa * c2aa_signs)))

    # alpha, beta -> alpha, beta excitations
    fcimatr_dict.update(
        dict(
            zip(
                list(product(c1a_configs, c1b_configs)),
                np.einsum("i,j,ij->ij", c1a_signs, c1b_signs, c2ab, optimize=True).ravel(),
            )
        )
    )

    # beta, beta -> beta, beta excitations
    c2bb_configs, c2bb_signs = _excited_configurations(nelec_b, norb, 2)
    fcimatr_dict.update(dict(zip(list(zip([ref_a] * size_bb, c2bb_configs)), c2bb * c2bb_signs)))

    # filter based on tolerance cutoff
    fcimatr_dict = {key: value for key, value in fcimatr_dict.items() if abs(value) > tol}

    # convert sign parity from chemist to physicist convention (interleaving spin operators
    # rather than commuting all spin-up operators to the left)
    fcimatr_dict = _sign_chem_to_phys(fcimatr_dict, norb)

    return fcimatr_dict


def _rccsd_state(ccsd_solver, tol=1e-15):
    r"""Construct a wavefunction from PySCF's ``RCCSD`` Solver object.

    The generated wavefunction is a dictionary where the keys represent a configuration, which
    corresponds to a Slater determinant, and the values are the CI coefficients of the Slater
    determinant. Each dictionary key is a tuple of two integers. The binary representation of these
    integers correspond to a specific configuration: the first number represents the
    configuration of the alpha electrons and the second number represents the configuration of the
    beta electrons. For instance, the Hartree-Fock state :math:`|1 1 0 0 \rangle` will be
    represented by the flipped binary string ``0011` which is split to ``01`` and ``01`` for
    the alpha and beta electrons. The integer corresponding to ``01`` is ``1`` and the dictionary
    representation of the Hartree-Fock state will be ``{(1, 1): 1.0}``. The dictionary
    representation of a state with ``0.99`` contribution from the Hartree-Fock state and ``0.01``
    contribution from the doubly-excited state, i.e., :math:`|0 0 1 1 \rangle`, will be
    ``{(1, 1): 0.99, (2, 2): 0.01}``.

    In the current version, the exponential ansatz :math:`\exp(\hat{T}_1 + \hat{T}_2) \ket{\text{HF}}`
    is expanded to second order, with only single and double excitation terms collected and kept.
    In the future this will be amended to also collect terms from higher order. The expansion gives

    .. math::
        \exp(\hat{T}_1 + \hat{T}_2) \ket{\text{HF}} = \left[ 1 + \hat{T}_1 +
        \left( \hat{T}_2 + 0.5 * \hat{T}_1^2 \right) \right] \ket{\text{HF}}

    The coefficients in this expansion are the CI coefficients used to build the wavefunction
    representation.

    Args:
        ccsd_solver (PySCF RCCSD Class instance): the class object representing the RCCSD calculation in PySCF
        tol (float): the tolerance for discarding Slater determinants with small coefficients

    Returns:
        fcimatr_dict (dict[tuple(int,int),float]): dictionary of the form ``{(int_a, int_b) :coeff}``, with integers ``int_a, int_b``
        having binary represention corresponding to the Fock occupation vector in alpha and beta
        spin sectors, respectively, and coeff being the CI coefficients of those configurations

    **Example**

    >>> from pyscf import gto, scf, cc
    >>> mol = gto.M(atom=[['Li', (0, 0, 0)], ['Li', (0,0,0.71)]], basis='sto6g', symmetry="d2h")
    >>> myhf = scf.RHF(mol).run()
    >>> mycc = cc.CCSD(myhf).run()
    >>> wf_ccsd = _rccsd_state(mycc, tol=1e-1)
    >>> print(wf_ccsd)
    {(7, 7): -0.8886969878256522, (11, 11): 0.30584590248164206,
     (19, 19): 0.30584590248164145, (35, 35): 0.14507552651170982}
    """

    mol = ccsd_solver.mol

    norb = mol.nao
    nelec = mol.nelectron
    nelec_a = int((nelec + mol.spin) / 2)
    nelec_b = int((nelec - mol.spin) / 2)

    nvir_a, nvir_b = norb - nelec_a, norb - nelec_b

    # build the full, unrestricted representation of the coupled cluster amplitudes
    t1a = ccsd_solver.t1
    t1b = t1a
    t2aa = ccsd_solver.t2 - ccsd_solver.t2.transpose(1, 0, 2, 3)
    t2ab = ccsd_solver.t2.transpose(0, 2, 1, 3)
    t2bb = t2aa

    # add in the disconnected part ( + 0.5 T_1^2) of double excitations
    t2aa = t2aa - 0.5 * np.kron(t1a, t1a).reshape(nelec_a, nvir_a, nelec_a, nvir_a).transpose(
        0, 2, 1, 3
    )
    t2bb = t2bb - 0.5 * np.kron(t1b, t1b).reshape(nelec_b, nvir_b, nelec_b, nvir_b).transpose(
        0, 2, 1, 3
    )
    # align the entries with how the excitations are ordered when generated by _excited_configurations()
    t2ab = t2ab - 0.5 * np.kron(t1a, t1b).reshape(nelec_a, nvir_a, nelec_b, nvir_b)

    # numbers representing the Hartree-Fock vector, e.g., bin(ref_a)[::-1] = 1111...10...0
    ref_a = int(2**nelec_a - 1)
    ref_b = int(2**nelec_b - 1)

    fcimatr_dict = dict(zip(list(zip([ref_a], [ref_b])), [1.0]))

    # alpha -> alpha excitations
    t1a_configs, t1a_signs = _excited_configurations(nelec_a, norb, 1)
    fcimatr_dict.update(
        dict(zip(list(zip(t1a_configs, [ref_b] * len(t1a_configs))), t1a.ravel() * t1a_signs))
    )

    # beta -> beta excitations
    t1b_configs, t1b_signs = _excited_configurations(nelec_b, norb, 1)
    fcimatr_dict.update(
        dict(zip(list(zip([ref_a] * len(t1b_configs), t1b_configs)), t1b.ravel() * t1b_signs))
    )

    # alpha, alpha -> alpha, alpha excitations
    if nelec_a > 1 and nvir_a > 1:
        t2aa_configs, t2aa_signs = _excited_configurations(nelec_a, norb, 2)
        # select only unique excitations, via lower triangle of matrix
        ooidx = np.tril_indices(nelec_a, -1)
        vvidx = np.tril_indices(nvir_a, -1)
        t2aa = t2aa[ooidx][:, vvidx[0], vvidx[1]]
        fcimatr_dict.update(
            dict(
                zip(list(zip(t2aa_configs, [ref_b] * len(t2aa_configs))), t2aa.ravel() * t2aa_signs)
            )
        )

    if nelec_b > 1 and nvir_b > 1:
        t2bb_configs, t2bb_signs = _excited_configurations(nelec_b, norb, 2)
        # select only unique excitations, via lower triangle of matrix
        ooidx = np.tril_indices(nelec_b, -1)
        vvidx = np.tril_indices(nvir_b, -1)
        t2bb = t2bb[ooidx][:, vvidx[0], vvidx[1]]
        fcimatr_dict.update(
            dict(
                zip(list(zip([ref_a] * len(t2bb_configs), t2bb_configs)), t2bb.ravel() * t2bb_signs)
            )
        )

    # alpha, beta -> alpha, beta excitations
    fcimatr_dict.update(
        dict(
            zip(
                list(product(t1a_configs, t1b_configs)),
                np.einsum(
                    "i,j,ij->ij",
                    t1a_signs,
                    t1b_signs,
                    t2ab.reshape(nelec_a * nvir_a, -1),
                    optimize=True,
                ).ravel(),
            )
        )
    )

    # renormalize, to get the HF coefficient (CC wavefunction not normalized)
    norm = np.sqrt(np.sum(np.array(list(fcimatr_dict.values())) ** 2))
    fcimatr_dict = {key: value / norm for (key, value) in fcimatr_dict.items()}

    # filter based on tolerance cutoff
    fcimatr_dict = {key: value for key, value in fcimatr_dict.items() if abs(value) > tol}

    # convert sign parity from chemist to physicist convention (interleaving spin operators
    # rather than commuting all spin-up operators to the left)
    fcimatr_dict = _sign_chem_to_phys(fcimatr_dict, norb)

    return fcimatr_dict


def _uccsd_state(ccsd_solver, tol=1e-15):
    r"""Construct a wavefunction from PySCF's ``UCCSD`` Solver object.

    The generated wavefunction is a dictionary where the keys represent a configuration, which
    corresponds to a Slater determinant, and the values are the CI coefficients of the Slater
    determinant. Each dictionary key is a tuple of two integers. The binary representation of these
    integers correspond to a specific configuration: the first number represents the
    configuration of the alpha electrons and the second number represents the configuration of the
    beta electrons. For instance, the Hartree-Fock state :math:`|1 1 0 0 \rangle` will be
    represented by the flipped binary string ``0011`` which is split to ``01`` and ``01`` for
    the alpha and beta electrons. The integer corresponding to ``01`` is ``1`` and the dictionary
    representation of the Hartree-Fock state will be ``{(1, 1): 1.0}``. The dictionary
    representation of a state with ``0.99`` contribution from the Hartree-Fock state and ``0.01``
    contribution from the doubly-excited state, i.e., :math:`|0 0 1 1 \rangle`, will be
    ``{(1, 1): 0.99, (2, 2): 0.01}``.

    In the current version, the exponential ansatz :math:`\exp(\hat{T}_1 + \hat{T}_2) \ket{\text{HF}}`
    is expanded to second order, with only single and double excitation terms collected and kept.
    In the future this will be amended to also collect terms from higher order. The expansion gives

    .. math::
        \exp(\hat{T}_1 + \hat{T}_2) \ket{\text{HF}} = \left[ 1 + \hat{T}_1 +
                                    \left( \hat{T}_2 + 0.5 * \hat{T}_1^2 \right) \right] \ket{\text{HF}}

    The coefficients in this expansion are the CI coefficients used to build the wavefunction
    representation.

    Args:
        ccsd_solver (PySCF UCCSD Class instance): the class object representing the UCCSD calculation in PySCF
        tol (float): the tolerance for discarding Slater determinants with small coefficients

    Returns:
        fcimatr_dict (dict[tuple(int,int),float]): dictionary of the form `{(int_a, int_b) :coeff}`, with integers `int_a, int_b`
        having binary represention corresponding to the Fock occupation vector in alpha and beta
        spin sectors, respectively, and coeff being the CI coefficients of those configurations

    **Example**

    >>> from pyscf import gto, scf, cc
    >>> mol = gto.M(atom=[['Li', (0, 0, 0)], ['Li', (0,0,0.71)]], basis='sto6g', symmetry="d2h")
    >>> myhf = scf.UHF(mol).run()
    >>> mycc = cc.UCCSD(myhf).run()
    >>> wf_ccsd = _uccsd_state(mycc, tol=1e-1)
    >>> print(wf_ccsd)
    {(7, 7): -0.8886970081919591, (11, 11): 0.3058459002168582,
     (19, 19): 0.30584590021685887, (35, 35): 0.14507552387854625}
    """

    mol = ccsd_solver.mol

    norb = mol.nao
    nelec = mol.nelectron
    nelec_a = int((nelec + mol.spin) / 2)
    nelec_b = int((nelec - mol.spin) / 2)

    nvir_a, nvir_b = norb - nelec_a, norb - nelec_b

    t1a, t1b = ccsd_solver.t1
    t2aa, t2ab, t2bb = ccsd_solver.t2
    # add in the disconnected part ( + 0.5 T_1^2) of double excitations
    t2aa = t2aa - 0.5 * np.kron(t1a, t1a).reshape(nelec_a, nvir_a, nelec_a, nvir_a).transpose(
        0, 2, 1, 3
    )
    t2bb = t2bb - 0.5 * np.kron(t1b, t1b).reshape(nelec_b, nvir_b, nelec_b, nvir_b).transpose(
        0, 2, 1, 3
    )
    # align the entries with how the excitations are ordered when generated by _excited_configurations()
    t2ab = t2ab.transpose(0, 2, 1, 3) - 0.5 * np.kron(t1a, t1b).reshape(
        nelec_a, nvir_a, nelec_b, nvir_b
    )

    # numbers representing the Hartree-Fock vector, e.g., bin(ref_a)[::-1] = 1111...10...0
    ref_a = int(2**nelec_a - 1)
    ref_b = int(2**nelec_b - 1)

    fcimatr_dict = dict(zip(list(zip([ref_a], [ref_b])), [1.0]))

    # alpha -> alpha excitations
    t1a_configs, t1a_signs = _excited_configurations(nelec_a, norb, 1)
    fcimatr_dict.update(
        dict(zip(list(zip(t1a_configs, [ref_b] * len(t1a_configs))), t1a.ravel() * t1a_signs))
    )

    # beta -> beta excitations
    t1b_configs, t1b_signs = _excited_configurations(nelec_b, norb, 1)
    fcimatr_dict.update(
        dict(zip(list(zip([ref_a] * len(t1b_configs), t1b_configs)), t1b.ravel() * t1b_signs))
    )

    # alpha, alpha -> alpha, alpha excitations
    if nelec_a > 1 and nvir_a > 1:
        t2aa_configs, t2aa_signs = _excited_configurations(nelec_a, norb, 2)
        # select only unique excitations, via lower triangle of matrix
        ooidx = np.tril_indices(nelec_a, -1)
        vvidx = np.tril_indices(nvir_a, -1)
        t2aa = t2aa[ooidx][:, vvidx[0], vvidx[1]]
        fcimatr_dict.update(
            dict(
                zip(list(zip(t2aa_configs, [ref_b] * len(t2aa_configs))), t2aa.ravel() * t2aa_signs)
            )
        )

    # beta, beta -> beta, beta excitations
    if nelec_b > 1 and nvir_b > 1:
        t2bb_configs, t2bb_signs = _excited_configurations(nelec_b, norb, 2)
        # select only unique excitations, via lower triangle of matrix
        ooidx = np.tril_indices(nelec_b, -1)
        vvidx = np.tril_indices(nvir_b, -1)
        t2bb = t2bb[ooidx][:, vvidx[0], vvidx[1]]
        fcimatr_dict.update(
            dict(
                zip(list(zip([ref_a] * len(t2bb_configs), t2bb_configs)), t2bb.ravel() * t2bb_signs)
            )
        )

    # alpha, beta -> alpha, beta excitations
    fcimatr_dict.update(
        dict(
            zip(
                list(product(t1a_configs, t1b_configs)),
                np.einsum(
                    "i,j,ij->ij",
                    t1a_signs,
                    t1b_signs,
                    t2ab.reshape(nelec_a * nvir_a, -1),
                    optimize=True,
                ).ravel(),
            )
        )
    )

    # renormalize, to get the HF coefficient (CC wavefunction not normalized)
    norm = np.sqrt(np.sum(np.array(list(fcimatr_dict.values())) ** 2))
    fcimatr_dict = {key: value / norm for (key, value) in fcimatr_dict.items()}

    # filter based on tolerance cutoff
    fcimatr_dict = {key: value for key, value in fcimatr_dict.items() if abs(value) > tol}

    # convert sign parity from chemist to physicist convention (interleaving spin operators
    # rather than commuting all spin-up operators to the left)
    fcimatr_dict = _sign_chem_to_phys(fcimatr_dict, norb)

    return fcimatr_dict


def _dmrg_state(wavefunction, tol=1e-15):
    r"""Construct a wavefunction from the DMRG wavefunction obtained from the Block2 library.

    The generated wavefunction is a dictionary where the keys represent a configuration, which
    corresponds to a Slater determinant, and the values are the CI coefficients of the Slater
    determinant. Each dictionary key is a tuple of two integers. The binary representation of these
    integers correspond to a specific configuration: the first number represents the
    configuration of the alpha electrons and the second number represents the configuration of the
    beta electrons. For instance, the Hartree-Fock state :math:`|1 1 0 0 \rangle` will be
    represented by the flipped binary string ``0011`` which is split to ``01`` and ``01`` for
    the alpha and beta electrons. The integer corresponding to ``01`` is ``1`` and the dictionary
    representation of the Hartree-Fock state will be ``{(1, 1): 1.0}``. The dictionary
    representation of a state with ``0.99`` contribution from the Hartree-Fock state and ``0.01``
    contribution from the doubly-excited state, i.e., :math:`|0 0 1 1 \rangle`, will be
    ``{(1, 1): 0.99, (2, 2): 0.01}``.

    The determinants and coefficients should be supplied externally. They should be calculated by
    using Block2 DMRGDriver's `get_csf_coefficients()` method.

    Args:
        wavefunction tuple(list[int], array[float]): determinants and coefficients in physicist notation
        tol (float): the tolerance for discarding Slater determinants with small coefficients

    Returns:
        fcimatr_dict (dict[tuple(int,int),float]): dictionary of the form `{(int_a, int_b) : coeff}`, with integers `int_a, int_b`
        having binary representation corresponding to the Fock occupation vector in alpha and beta
        spin sectors, respectively, and coeff being the CI coefficients of those configurations

    **Example**

    >>> import numpy as np
    >>> wavefunction = ( [[0, 3], [3, 0]], np.array([-0.10660077,  0.9943019 ]))
    >>> wf_dmrg = _dmrg_state(wavefunction, tol=1e-1)
    >>> print(wf_dmrg)
    {(2, 2): -0.10660077, (1, 1): 0.9943019}
    """
    dets, coeffs = wavefunction

    row, col = [], []

    for det in dets:
        stra, strb = _sitevec_to_fock(det, format="dmrg")
        row.append(stra)
        col.append(strb)

        # slater determinant sign convention of block2 (physicist,
        # interleave spin-up/down operators) is consistent with pennylane

    ## create the FCI matrix as a dict
    fcimatr_dict = dict(zip(list(zip(row, col)), coeffs))

    # filter based on tolerance cutoff
    fcimatr_dict = {key: value for key, value in fcimatr_dict.items() if abs(value) > tol}

    return fcimatr_dict


def _shci_state(wavefunction, tol=1e-15):
    r"""Construct a wavefunction from the SHCI wavefunction obtained from the Dice library.

    The generated wavefunction is a dictionary where the keys represent a configuration, which
    corresponds to a Slater determinant, and the values are the CI coefficients of the Slater
    determinant. Each dictionary key is a tuple of two integers. The binary representation of these
    integers correspond to a specific configuration: the first number represents the
    configuration of the alpha electrons and the second number represents the configuration of the
    beta electrons. For instance, the Hartree-Fock state :math:`|1 1 0 0 \rangle` will be
    represented by the flipped binary string ``0011`` which is split to ``01`` and ``01`` for
    the alpha and beta electrons. The integer corresponding to ``01`` is ``1`` and the dictionary
    representation of the Hartree-Fock state will be ``{(1, 1): 1.0}``. The dictionary
    representation of a state with ``0.99`` contribution from the Hartree-Fock state and ``0.01``
    contribution from the doubly-excited state, i.e., :math:`|0 0 1 1 \rangle`, will be
    ``{(1, 1): 0.99, (2, 2): 0.01}``.

    The determinants and coefficients should be supplied externally. They are typically stored under
    SHCI.outputfile.

    Args:
        wavefunction tuple(list[str], array[float]): determinants and coefficients in chemist notation
        tol (float): the tolerance for discarding Slater determinants with small coefficients
    Returns:
        fcimatr_dict (dict[tuple(int,int),float]): dictionary of the form `{(int_a, int_b) : coeff}`, with integers `int_a, int_b`
        having binary representation corresponding to the Fock occupation vector in alpha and beta
        spin sectors, respectively, and coeff being the CI coefficients of those configurations

    **Example**

    >>> import numpy as np
    >>> wavefunction = ( ['20', '02'], np.array([-0.9943019036, 0.1066007711]))
    >>> wf_shci = _shci_state(wavefunction, tol=1e-1)
    >>> print(wf_shci)
    {(1, 1): -0.9943019036, (2, 2): 0.1066007711}
    """

    dets, coeffs = wavefunction
    norb = len(dets[0])

    xa = []
    xb = []

    for det in dets:
        bin_a, bin_b = _sitevec_to_fock(list(det), "shci")

        xa.append(bin_a)
        xb.append(bin_b)

    ## create the FCI matrix as a dict
    fcimatr_dict = dict(zip(list(zip(xa, xb)), coeffs))

    # filter based on tolerance cutoff
    fcimatr_dict = {key: value for key, value in fcimatr_dict.items() if abs(value) > tol}

    # convert sign parity from chemist to physicist convention (interleaving spin operators
    # rather than commuting all spin-up operators to the left)
    fcimatr_dict = _sign_chem_to_phys(fcimatr_dict, norb)

    return fcimatr_dict


def _sitevec_to_fock(det, format):
    r"""Convert a Slater determinant from site vector to occupation number vector representation.

    Args:
        det (list(int) or list(str)): determinant in site vector representation
        format (str): the format of the determinant

    Returns:
        tuple: tuple of integers representing binaries that correspond to occupation vectors in
            alpha and beta spin sectors

    **Example**

    >>> det = [1, 2, 1, 0, 0, 2]
    >>> _sitevec_to_fock(det, format = 'dmrg')
    (5, 34)

    >>> det = ["a", "b", "a", "0", "0", "b"]
    >>> _sitevec_to_fock(det, format = 'shci')
    (5, 34)
    """

    if format == "dmrg":
        format_map = {0: "00", 1: "10", 2: "01", 3: "11"}
    elif format == "shci":
        format_map = {"0": "00", "a": "10", "b": "01", "2": "11"}

    strab = [format_map[key] for key in det]

    stra = "".join(i[0] for i in strab)
    strb = "".join(i[1] for i in strab)

    inta = int(stra[::-1], 2)
    intb = int(strb[::-1], 2)

    return inta, intb


def _sign_chem_to_phys(fcimatr_dict, norb):
    r"""Convert the dictionary-form wavefunction from chemist sign convention
    for ordering the creation operators by spin (i.e. all spin-up operators
    on the left) to the physicist convention native to PennyLane, which
    storing spin operators as interleaved for the same spatial orbital index.

    Note that convention change in the opposite direction -- starting from physicist
    and going to chemist -- can be accomplished with the same function
    (the sign transformation is reversible).

    Args:
        fcimatr_dict (dict[tuple(int, int), float]): dictionary of the form `{(int_a, int_b) :coeff}`, with integers `int_a, int_b`
        having binary represention corresponding to the Fock occupation vector in alpha and beta
        spin sectors, respectively, and coeff being the CI coefficients of those configurations
        norb (int): total number of spatial orbitals of the underlying system

    Returns:
        signed_dict (dict): the same dictionary-type wavefunction with appropriate signs converted

    **Example**

    >>> fcimatr_dict = {(3, 1): 0.96, (6, 1): 0.1, \
                        (3, 4): 0.1, (6, 4): 0.14, (5, 2): 0.19}
    >>> _sign_chem_to_phys(fcimatr_dict, 3)
    {(3, 1): -0.96, (6, 1): 0.1, (3, 4): 0.1, (6, 4): 0.14, (5, 2): -0.19}
    """

    signed_dict = {}
    for key, elem in fcimatr_dict.items():
        lsta, lstb = bin(key[0])[2:][::-1], bin(key[1])[2:][::-1]
        # highest energy state is on the right -- pad to the right
        lsta = np.array([int(elem) for elem in lsta] + [0] * (norb - len(lsta)))
        lstb = np.array([int(elem) for elem in lstb] + [0] * (norb - len(lstb)))
        which_occ = np.where(lsta == 1)[0]
        parity = (-1) ** np.sum([np.sum(lstb[: int(ind)]) for ind in which_occ])
        signed_dict[key] = parity * elem
    return signed_dict
