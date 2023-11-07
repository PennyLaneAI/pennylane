    # Copyright 2018-2023 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Transform function for the Clifford+T decomposition."""

import math
from functools import partial
from itertools import product
from typing import Sequence, Callable

import pennylane as qml
from pennylane.queuing import QueuingManager
from pennylane.transforms.core import transform
from pennylane.tape import QuantumTape
from pennylane.transforms.optimization import (
    cancel_inverses,
    commute_controlled,
    merge_rotations,
    remove_barrier,
)
from pennylane.transforms.optimization.optimization_utils import find_next_gate, _fuse_global_phases
from .solovay_kitaev import sk_decomposition

# Single qubits Clifford+T gates in PL
_CLIFFORD_T_ONE_GATES = [
    qml.Identity,
    qml.PauliX,
    qml.PauliY,
    qml.PauliZ,
    qml.Hadamard,
    qml.S,
    qml.SX,
    qml.T,
]

# Two qubits Clifford+T gates in PL
_CLIFFORD_T_TWO_GATES = [
    qml.CNOT,
    qml.CY,
    qml.CZ,
    qml.SWAP,
    qml.ISWAP,
]

# Single-parameter gates in PL
_PARAMETER_GATES = [qml.RX, qml.RY, qml.RZ, qml.Rot, qml.PhaseShift, qml.GlobalPhase]

# Clifford+T gate set
_CLIFFORD_T_GATES = _CLIFFORD_T_ONE_GATES + _CLIFFORD_T_TWO_GATES


def check_clifford_op(op):
    r"""Check if an operator is Clifford or not.

    For a given unitary operator :math:`U` acting on :math:`N` qubits, this method checks that the
    transformation :math:`UPU^{\dagger}` maps the Pauli tensor products :math:`P = {I, X, Y, Z}^{\otimes N}`
    to Pauli tensor products with :math:`O(N \times 8^N)` time complexity when using naive matrix multiplication.

    Args:
        op: the operator that needs to be tested

    Returns:
        Bool that represents whether the provided operator is Clifford or not.
    """

    # Check if matrix can be calculated for the operator
    if not op.has_matrix:
        return False

    # Compute the LCUs for the operator in Pauli basis
    pauli_terms = qml.pauli_decompose(qml.matrix(op), wire_order=op.wires, check_hermitian=False)

    # Build Pauli tensor products P in Hamiltonian form
    def pauli_group(x):
        return [qml.Identity(x), qml.PauliX(x), qml.PauliY(x), qml.PauliZ(x)]

    # Build PauliSentence, PauliWord and Hamiltonians representations for set P
    pauli_sens = [
        qml.pauli.pauli_sentence(qml.prod(*pauli))
        for pauli in product(*(pauli_group(idx) for idx in op.wires))
    ]
    pauli_word = [list(word.keys())[0] for word in pauli_sens]
    pauli_hams = [pauli_sen.hamiltonian(wire_order=op.wires) for pauli_sen in pauli_sens]

    # Perform U@P@U^\dagger and check if the result exists in set P
    pauli_coves = []
    for prod in product([pauli_terms], pauli_hams, [pauli_terms]):
        # hopefully op_math.prod scales better than matrix multiplication, i.e., O((2^N)^3)
        upu = qml.pauli.pauli_sentence(qml.prod(*prod))
        upu.simplify()

        if not len(upu) == 1:  # Pauli sum always lie outisde set P
            return False  # early stopping

        op_word = list(upu.keys())[0]  # Get the Pauli Word in the sentence
        if op_word != pauli_word[0]:  # Identity always lie in set P
            pauli_coves.append(any(op_word == word for word in pauli_word[1:]))

    return all(pauli_coves)


def check_clifford_t(op):
    r"""Checks whether the gate is in the standard Clifford+T basis.

    For a given unitary operator :math:`U` acting on :math:`N` qubits, which is not a T-gate,
    this method checks that the transformation :math:`UPU^{\dagger}` maps the Pauli tensor products
    :math:`P = {I, X, Y, Z}^{\otimes N}` to Pauli tensor products with :math:`O(N \times 8^N)` time
    complexity when using naive matrix multiplication.

    Args:
        op: the operator that needs to be checked

    Returns:
        Bool that represents whether the provided operator is Clifford+T or not.
    """

    # Save time and check from the pre-computed list
    if any(
        (
            isinstance(op, gate) or isinstance(getattr(op, "base", None), gate)
            for gate in _CLIFFORD_T_GATES + [qml.GlobalPhase]
        )
    ):
        return True

    # Save time and check from the parameter of rotation gates
    if any(
        (
            isinstance(op, gate) or isinstance(getattr(op, "base", None), gate)
            for gate in _PARAMETER_GATES[:-1]  # don't check for GlobalPhase
        )
    ):
        return qml.math.isclose(op.data[0] % math.pi, 0.0)

    return check_clifford_op(op)


def _simplify_param(theta, gate):
    r"""Check if the parameter allows simplification for the rotation gate.

    For the cases where theta is an integer multiple of π: (a) returns a global phase
    when even, and (b) reutrns combination of provided gate with global phase when odd.
    In rest of the other cases it returns None.
    """
    if qml.math.isclose(theta, 0.0, atol=1e-6):
        return [qml.GlobalPhase(0.0)]

    rem, mod = theta / math.pi, theta % math.pi
    if qml.math.isclose(mod, 0.0, atol=1e-6):
        ops = [qml.GlobalPhase(theta / 2)]
        if qml.math.isclose(rem % 2, 1.0, atol=1e-6):
            ops.append(gate)
        return ops

    return None


# pylint: disable= too-many-branches,
def _rot_decompose(op):
    """Decompose a rotation operation using combination of RZ, S and Hadamard"""
    d_ops = []
    # Extend for Rot operation with RzRyRz decompositions
    if isinstance(op, qml.Rot):
        (phi, theta, omega), wires = op.parameters, op.wires
        for dec in [qml.RZ(phi, wires), qml.RY(theta, wires), qml.RZ(omega, wires)]:
            d_ops.extend(_rot_decompose(dec))
        return d_ops

    (theta,), wires = op.data, op.wires
    if isinstance(op, qml.ops.Adjoint):  # pylint: disable=no-member
        ops_ = _rot_decompose(op.base.adjoint())
    elif isinstance(op, qml.RX):
        ops_ = _simplify_param(theta, qml.PauliX(wires=wires))
        if ops_ is None:  # Use Rx = H @ Rz @ H
            ops_ = [qml.Hadamard(wires), qml.RZ(theta, wires), qml.Hadamard(wires)]
    elif isinstance(op, qml.RY):
        ops_ = _simplify_param(theta, qml.PauliY(wires=wires))
        if ops_ is None:  # Use Ry = S @ H @ Rz @ H @ S.dag
            ops_ = [
                qml.S(wires),
                qml.Hadamard(wires),
                qml.RZ(theta, wires),
                qml.Hadamard(wires),
                qml.adjoint(qml.S(wires)),
            ][::-1]
    elif isinstance(op, qml.RZ):
        ops_ = _simplify_param(theta, qml.PauliZ(wires=wires))
        if ops_ is None:
            ops_ = [qml.RZ(theta, wires)]
    elif isinstance(op, qml.PhaseShift):
        ops_ = _simplify_param(theta, qml.PauliZ(wires=wires))
        if ops_ is None:
            ops_ = [qml.RZ(theta, wires=wires), qml.GlobalPhase(theta / 2)]
        else:
            ops_.append(qml.GlobalPhase(-theta / 2))

    else:
        raise ValueError(
            f"Operation {op} is not a valid Pauli rotation: qml.RX, qml.RY, qml.RZ, qml.Rot and qml.PhaseShift"
        )

    d_ops.extend(ops_)
    return d_ops


def _one_qubit_decompose(op):
    """Decomposition for single qubit operations using combination of RZ and Hadamard"""

    sd_ops = qml.transforms.one_qubit_decomposition(
        qml.matrix(op), op.wires, "ZXZ", return_global_phase=True
    )

    d_ops = []
    for sd_op in sd_ops[:-1]:
        if sd_op.num_params:
            d_ops.extend(_rot_decompose(sd_op) if sd_op.num_params else [sd_op])

    return d_ops, sd_ops[-1]


def _two_qubit_decompose(op):
    """Decomposition for two qubit operations using combination of RZ, Hadamard, S and CNOT"""

    td_ops = qml.transforms.two_qubit_decomposition(qml.matrix(op), op.wires)

    d_ops = []
    for td_op in td_ops:
        d_ops.extend(
            _rot_decompose(td_op) if td_op.num_params and td_op.num_wires == 1 else [td_op]
        )

    return d_ops


def _merge_pauli_rotations(operations, merge_ops=None):
    """Merge the provided single qubit rotations gates on the same wires that are adjacent to each other"""

    copied_ops = operations.copy()
    merged_ops = []

    while len(copied_ops) > 0:
        curr_gate = copied_ops[0]

        # if gate is not to be merged, let it go
        if merge_ops is not None and curr_gate.name not in merge_ops:
            merged_ops.append(curr_gate)
            copied_ops.pop(0)
            continue

        # Find the next gate that acts on the same wires
        next_gate_idx = find_next_gate(curr_gate.wires, copied_ops[1:])
        if next_gate_idx is None:
            merged_ops.append(curr_gate)
            copied_ops.pop(0)
            continue

        # Initialize the current angle and iterate until end of merge
        cumulative_angles = 1.0 * qml.math.array(curr_gate.parameters)
        while next_gate_idx is not None:
            # If next gate is of the same type, we can merge the angles
            next_gate = copied_ops[next_gate_idx + 1]
            if curr_gate.name == next_gate.name and curr_gate.wires == next_gate.wires:
                copied_ops.pop(next_gate_idx + 1)
                cumulative_angles = cumulative_angles + qml.math.array(next_gate.parameters)
            else:
                break

            next_gate_idx = find_next_gate(curr_gate.wires, copied_ops[1:])

        # Replace the current gate, add it to merged list and pop it from orginal one
        merged_ops.append(curr_gate.__class__(*cumulative_angles, wires=curr_gate.wires))
        copied_ops.pop(0)

    return merged_ops


# pylint: disable= too-many-nested-blocks, too-many-branches, too-many-statements, unnecessary-lambda-assignment
@transform
def clifford_t_decomposition(
    tape: QuantumTape,
    max_depth=6,
    method="sk",
    **kwargs,
) -> (Sequence[QuantumTape], Callable):
    r"""Unrolls a circuit into the Clifford+T basis using the optimal ancilla-free approximation of :class:`~.RZ` operations.

    This method first decomposes the gate operations to a basis comprised of Clifford, :class:`~.RZ` and
    :class:`~.GlobalPhase` operations (and their adjoint). The Clifford gates include the following PennyLane operations:

    - Single qubit gates - :class:`~.Identity`, :class:`~.PauliX`, :class:`~.PauliY`, :class:`~.PauliZ`,
      :class:`~.SX`, :class:`~.S`, and :class:`~.Hadamard`.
    - Two qubit gates - :class:`~.CNOT`, :class:`~.CY`, :class:`~.CZ`, :class:`~.SWAP`, and :class:`~.ISWAP`.

    Then the leftover single qubit :class:`~.pennylane.RZ` operations are approximated in the Clifford+T basis with
    :math:`\epsilon > 0` error using the recursive Solovay-Kitaev algorithm described in
    `Dawson and Nielsen (2005) <https://arxiv.org/abs/quant-ph/0505030>`_.

    Args:
        qfunc (function): A quantum function
        max_depth (int): The depth to use for tape expansion before manual decomposition to Clifford+T basis is applied
        method (str): Method to be used for Clifford+T decomposition. Default value is ``"sk"`` for Solovay-Kitaev
        **kwargs: Keyword argument to pass options for the ``method`` used for decompositions

    Keyword Args:
        Options (*):

            * **depth** (int), **basis_set** (list(str)), **basis_length** (int):
              arguments for using the ``"sk"`` method, i.e., for performing Solovay-Kitaev decomposition using :func:`~.sk_decomposition`

    Returns:
        pennylane.QNode or qfunc or tuple[List[.QuantumTape], function]: If a QNode is passed,
        it returns a QNode with the transform added to its transform program.
        If a tape is passed, returns a tuple containing a list of
        quantum tapes to be evaluated, and a function to be applied to these
        tape executions.

    Raises:
        ValueError: If any gate operation does not have a rule for its decomposition
        NotImplementedError: If chosen decomposition is not supported

    .. seealso:: :func:`~.sk_decomposition` for Solovay-Kitaev decomposition.
    """
    with QueuingManager.stop_recording():
        # Build the basis set and the pipeline for intial compilation pass
        basis_set = [op.__name__ for op in _PARAMETER_GATES + _CLIFFORD_T_GATES]
        pipelines = [remove_barrier, commute_controlled, cancel_inverses, merge_rotations]

        # Expand the tape according to depth provided by the user and try simplifying it
        expanded_tape = tape.expand(depth=max_depth, stop_at=lambda op: op.name in basis_set)
        for transf in pipelines:
            [expanded_tape], _ = transf(expanded_tape)

        # Now iterate over the expanded tape operations
        decomp_ops, gphase_ops = [], []
        for op in expanded_tape.operations:
            # Check whether operation is to be skipped
            if any(
                (isinstance(op, skip_op) for skip_op in [qml.Barrier, qml.Snapshot, qml.WireCut])
            ):
                decomp_ops.append(op)

            # Check whether the operation is a global phase
            elif isinstance(op, qml.GlobalPhase):
                gphase_ops.append(op)

            # Check whether the operation is a Clifford or a T-gate
            elif op.name in basis_set and check_clifford_t(op):
                if op.num_params:
                    decomp_ops.extend(_rot_decompose(op))
                else:
                    decomp_ops.append(op)

            # Decompose and then iteratively go deeper via DFS
            else:
                # Single qubit unitary decomposition with ZXZ rotations
                if op.num_wires == 1:
                    if op.name in basis_set:
                        d_ops = _rot_decompose(op)
                    else:
                        d_ops, g_ops = _one_qubit_decompose(op)
                        gphase_ops.append(g_ops)
                    decomp_ops.extend(d_ops)

                # Two qubit unitary decomposition with SU(4) rotations
                elif op.num_wires == 2:
                    d_ops = _two_qubit_decompose(op)
                    decomp_ops.extend(d_ops)

                # For special multi-qubit gates and ones constructed from matrix
                else:
                    try:
                        # Attempt decomposing the operation
                        md_ops = op.decomposition()

                        idx = 0  # might not be fast but at least is not recursive
                        while idx < len(md_ops):
                            md_op = md_ops[idx]
                            if md_op.name not in basis_set or not check_clifford_t(md_op):
                                # For the gates acting on one qubit
                                if len(md_op.wires) == 1:
                                    if md_op.name in basis_set:  # For known recipe
                                        d_ops = _rot_decompose(md_op)
                                    else:  # Resort to decomposing manually
                                        d_ops, g_ops = _one_qubit_decompose(md_op)
                                        gphase_ops.append(g_ops)

                                # For the gates acting on two qubits
                                elif len(md_op.wires) == 2:
                                    # Resort to decomposing manually
                                    d_ops = _two_qubit_decompose(md_op)

                                # Final resort (should not enter in an ideal situtation)
                                else:  # pragma: no cover
                                    d_ops = md_op.decomposition()

                                # Expand the list and iterate over
                                del md_ops[idx]
                                md_ops[idx:idx] = d_ops
                            idx += 1

                        decomp_ops.extend(md_ops)

                    # If we don't know how to decompose the operation
                    except Exception as exc:
                        raise ValueError(
                            f"Cannot unroll {op} into the Clifford+T basis as no rule exists for its decomposition"
                        ) from exc

        # Merge RZ rotations together
        merged_ops = _merge_pauli_rotations(decomp_ops, merge_ops=["RZ"])

        # Squeeze global phases into a single global phase
        new_operations = _fuse_global_phases(gphase_ops + merged_ops)

        # Build the approximation set for Solovay-Kitaev decomposition
        if method == "sk":
            _depth = kwargs.get("depth", 3)
            _basis_set = kwargs.get("basis_set", ("T", "T*", "H"))
            _basis_length = kwargs.get("basis_length", 10)

            # Get the decompose function for sk
            decompose_fn = partial(
                sk_decomposition, depth=_depth, basis_set=_basis_set, basis_length=_basis_length
            )

        # TODO: Add RZ conversion with the newsynth method
        else:
            raise NotImplementedError(
                f"Currently we only support Solovay-Kitaev ('sk') decompostion, got {method}"
            )

        new_ops = []
        for op in new_operations:
            if isinstance(op, qml.RZ):
                clifford_ops = decompose_fn(op)
                new_ops.extend(clifford_ops)
            else:
                new_ops.append(op)

    # Construct a new tape with the expanded set of operations
    new_tape = type(tape)(new_ops, expanded_tape.measurements, shots=tape.shots)

    # Perform a final attempt of simplification before return
    [new_tape], _ = cancel_inverses(new_tape)

    def null_postprocessing(results):
        """A postprocesing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        """
        return results[0]

    return [new_tape], null_postprocessing
