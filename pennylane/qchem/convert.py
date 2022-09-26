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
# pylint: disable=import-outside-toplevel
import pennylane as qml
from pennylane import numpy as np
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
        tuple[array[float], Iterable[pennylane.operation.Observable]]: coefficients and their
        corresponding PennyLane observables in the Pauli basis

    **Example**

    >>> q_op = 0.1*QubitOperator('X0') + 0.2*QubitOperator('Y0 Z2')
    >>> q_op
    0.1 [X0] +
    0.2 [Y0 Z2]
    >>> _openfermion_to_pennylane(q_op, wires=['w0','w1','w2','extra_wire'])
    (array([0.1, 0.2]), [PauliX(wires=['w0']), PauliY(wires=['w0']) @ PauliZ(wires=['w2'])])
    """
    n_wires = (
        1 + max([max([i for i, _ in t]) if t else 1 for t in qubit_operator.terms])
        if qubit_operator.terms
        else 1
    )
    wires = _process_wires(wires, n_wires=n_wires)

    if not qubit_operator.terms:  # added since can't unpack empty zip to (coeffs, ops) below
        return np.array([0.0]), [qml.Identity(wires[0])]

    xyz2pauli = {"X": qml.PauliX, "Y": qml.PauliY, "Z": qml.PauliZ}

    coeffs, ops = zip(
        *[
            (
                coef,
                qml.operation.Tensor(*[xyz2pauli[q[1]](wires=wires[q[0]]) for q in term])
                if len(term) > 1
                else (
                    xyz2pauli[term[0][1]](wires=wires[term[0][0]])
                    if len(term) == 1
                    else qml.Identity(wires[0])
                )
                # example term: ((0,'X'), (2,'Z'), (3,'Y'))
            )
            for term, coef in qubit_operator.terms.items()
        ]
    )

    return np.real(np.array(coeffs, requires_grad=False)), list(ops)


def _pennylane_to_openfermion(coeffs, ops, wires=None):
    r"""Convert a 2-tuple of complex coefficients and PennyLane operations to
    OpenFermion ``QubitOperator``.

    This function is the inverse of ``_qubit_operator_to_terms``.

    Args:
        coeffs (array[complex]):
            coefficients for each observable, same length as ops
        ops (Iterable[pennylane.operation.Observable]): list of PennyLane observables as
            Tensor products of Pauli observables
        wires (Wires, list, tuple, dict): Custom wire mapping used to convert to qubit operator
            from an observable terms measurable in a PennyLane ansatz.
            For types Wires/list/tuple, each item in the iterable represents a wire label
            corresponding to the qubit number equal to its index.
            For type dict, only consecutive-int-valued dict (for wire-to-qubit conversion) is
            accepted. If None, will map sorted wires from all `ops` to consecutive int.

    Returns:
        QubitOperator: an instance of OpenFermion's ``QubitOperator``.

    **Example**

    >>> coeffs = np.array([0.1, 0.2])
    >>> ops = [
    ...     qml.operation.Tensor(qml.PauliX(wires=['w0'])),
    ...     qml.operation.Tensor(qml.PauliY(wires=['w0']), qml.PauliZ(wires=['w2']))
    ... ]
    >>> _pennylane_to_openfermion(coeffs, ops, wires=Wires(['w0', 'w1', 'w2']))
    0.1 [X0] +
    0.2 [Y0 Z2]
    """
    try:
        import openfermion
    except ImportError as Error:
        raise ImportError(
            "This feature requires openfermion."
            " It can be installed with: pip install openfermion"
        ) from Error

    all_wires = Wires.all_wires([op.wires for op in ops], sort=True)

    if wires is not None:
        qubit_indexed_wires = _process_wires(wires,)
        if not set(all_wires).issubset(set(qubit_indexed_wires)):
            raise ValueError("Supplied `wires` does not cover all wires defined in `ops`.")
    else:
        qubit_indexed_wires = all_wires

    q_op = openfermion.QubitOperator()
    for coeff, op in zip(coeffs, ops):

        if not isinstance(op, qml.operation.Tensor):
            op = qml.operation.Tensor(op)

        extra_obsvbs = set(op.name) - {"PauliX", "PauliY", "PauliZ", "Identity"}
        if extra_obsvbs != set():
            raise ValueError(
                f"Expected only PennyLane observables PauliX/Y/Z or Identity, "
                f"but also got {extra_obsvbs}."
            )

        # Pauli axis names, note s[-1] expects only 'Pauli{X,Y,Z}'
        pauli_names = [s[-1] if s != "Identity" else s for s in op.name]

        all_identity = all(obs.name == "Identity" for obs in op.obs)
        if (op.name == ["Identity"] and len(op.wires) == 1) or all_identity:
            term_str = ""
        else:
            term_str = " ".join(
                [
                    f"{pauli}{qubit_indexed_wires.index(wire)}"
                    for pauli, wire in zip(pauli_names, op.wires)
                    if pauli != "Identity"
                ]
            )

        # This is how one makes QubitOperator in OpenFermion
        q_op += complex(coeff) * openfermion.QubitOperator(term_str)

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


def import_operator(qubit_observable, format="openfermion", wires=None, tol=1e08):
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
        tol (float): Tolerance in machine epsilons for the imaginary part of the
            coefficients in ``qubit_observable``. Coefficients with imaginary part
            less than 2.22e-16*tol are considered to be real.

    Returns:
        (.Hamiltonian): PennyLane :class:`~.Hamiltonian`
        representing any operator expressed as linear combinations of observables, e.g.,
        :math:`\sum_{k=0}^{N-1} c_k O_k`

    **Example**

    >>> from openfermion import QubitOperator
    >>> h_of = QubitOperator('X0 X1 Y2 Y3', -0.0548) + QubitOperator('Z0 Z1', 0.14297)
    >>> h_pl = import_operator(h_of, format='openfermion')
    >>> print(h_pl)
    (0.14297) [Z0 Z1]
    + (-0.0548) [X0 X1 Y2 Y3]
    """
    if format not in "openfermion":
        raise TypeError(f"Converter does not exist for {format} format.")

    if any(
        np.iscomplex(np.real_if_close(coef, tol=tol)) for coef in qubit_observable.terms.values()
    ):
        raise TypeError(
            f"The coefficients entering the QubitOperator must be real;"
            f" got complex coefficients in the operator {qubit_observable}"
        )

    return qml.Hamiltonian(*_openfermion_to_pennylane(qubit_observable, wires=wires))
