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

from itertools import product
from typing import Sequence, Callable
import pennylane as qml
from pennylane.transforms.core import transform
from pennylane.tape import QuantumTape
from pennylane.transforms.optimization import (
    cancel_inverses,
    commute_controlled,
    merge_rotations,
    remove_barrier,
)

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

_PARAMETER_GATES = [qml.RX, qml.RY, qml.RZ, qml.Rot, qml.GlobalPhase]

_CLIFFORD_T_GATES = _CLIFFORD_T_ONE_GATES + _CLIFFORD_T_TWO_GATES


# pylint: disable= bare-except, unnecessary-lambda-assignment
def check_clifford_op(op):
    r"""Check if an operator is Clifford or not.

    For a given unitary operator :math:`U` acting on :math:`N` qubits, this method checks that the
    transformation :math:`UPU^{\dagger}` maps the Pauli tensor products :math:`P = {I, X, Y, Z}^{\otimes N}`
    to Pauli tensor products with O(N * 8^N) time complexity when using naive matrix multiplication.

    Args:
        op: the operator that needs to be tested

    Returns:
        Bool that represents whether the provided operator is Clifford or not.
    """

    num_qubits = len(op.wires)
    try:
        op_matrix = qml.matrix(op, wire_order=range(num_qubits))
    except:
        return False

    pauli_terms = qml.pauli_decompose(op_matrix, check_hermitian=False)
    pauli_group = lambda x: [qml.Identity(x), qml.PauliX(x), qml.PauliY(x), qml.PauliZ(x)]

    pauli_coves = []
    try:
        pauli_qubit = [
            qml.prod(*pauli) for pauli in product(*(pauli_group(idx) for idx in range(num_qubits)))
        ]
    except:
        pauli_qubit = [qml.Identity(0)]
    pauli_qubit = [
        qml.pauli.pauli_sentence(op).hamiltonian(wire_order=range(num_qubits)) for op in pauli_qubit
    ]

    for prod in product([pauli_terms], pauli_qubit, [pauli_terms]):
        # hopefully op_math.prod scales better than matrix multiplication, i.e., O((2^N)^3)
        upu = qml.pauli.pauli_sentence(qml.prod(*prod))
        upu.simplify()
        upu2 = upu.hamiltonian(wire_order=range(num_qubits))
        if len(upu2.ops) == 1:
            if not isinstance(upu2.ops[0], qml.Identity):
                pauli_coves.append(any((qml.equal(upu2.ops[0], tm) for tm in pauli_qubit)))
        else:
            pauli_coves.append(False)

    return all(pauli_coves)


def _check_clifford_t(op):
    """Checks whether the gate is in the standard Clifford+T basis"""
    # Save time and check from the pre-computed list
    if any(
        (
            isinstance(op, gate) or isinstance(getattr(op, "base", None), gate)
            for gate in _CLIFFORD_T_GATES
        )
    ):
        return True
    return check_clifford_op(op)


def _rot_decompose(op):
    """Decompose a rotation operation using combination of RZ, S and Hadamard"""
    d_ops = []
    if isinstance(op, qml.Rot):
        (phi, theta, omega), wires = op.data, op.wires
        d_ops.extend(
            [
                qml.RZ(phi, wires),
                qml.S(wires),
                qml.Hadamard(wires),
                qml.RZ(theta, wires),
                qml.Hadamard(wires),
                qml.adjoint(qml.S(wires)),
                qml.RZ(omega, wires),
            ]
        )
    elif isinstance(op, qml.RX):
        (theta,), wires = op.data, op.wires
        if theta:
            d_ops.extend([qml.Hadamard(wires), qml.RZ(theta, wires), qml.Hadamard(wires)])
    elif isinstance(op, qml.RY):
        (theta,), wires = op.data, op.wires
        if theta:
            d_ops.extend(
                [
                    qml.S(wires),
                    qml.Hadamard(wires),
                    qml.RZ(theta, wires),
                    qml.Hadamard(wires),
                    qml.adjoint(qml.S(wires)),
                ]
            )
    elif isinstance(op, qml.RZ):
        (theta,), wires = op.data, op.wires
        if theta:
            d_ops.extend([qml.RZ(theta, wires)])
    else:
        d_ops.append(op)
    return d_ops


def _one_qubit_decompose(op):
    """Decomposition for single qubit operations using combination of RZ and Hadamard"""

    sd_ops = qml.transforms.one_qubit_decomposition(
        qml.matrix(op), op.wires, "ZXZ", return_global_phase=True
    )

    d_ops = []
    for sd_op in sd_ops[:-1]:
        d_ops.extend(_rot_decompose(sd_op))

    return d_ops[:-1], d_ops[-1]


def _two_qubit_decompose(op):
    """Decomposition for two qubit operations using combination of RZ, Hadamard, S and CNOT"""

    td_ops = qml.transforms.two_qubit_decomposition(qml.matrix(op), op.wires)

    d_ops = []
    for td_op in td_ops:
        d_ops.extend(_rot_decompose(td_op))

    return d_ops


# pylint: disable= too-many-nested-blocks, too-many-branches
@transform
def clifford_t_decomposition(tape: QuantumTape, epsilon=1e-8) -> (Sequence[QuantumTape], Callable):
    r"""Unrolls the tape into Clifford+T basis"""

    # Build the basis set and the pipeline for intial compilation pass
    basis_set = [op.__name__ for op in _PARAMETER_GATES + _CLIFFORD_T_GATES]
    pipelines = [remove_barrier, commute_controlled, cancel_inverses, merge_rotations]
    [tape], _ = qml.compile(tape, basis_set=basis_set, pipeline=pipelines)

    # Now iterate over the compiled pipeline
    decomp_ops, gphase_ops = [], []
    for op in tape.operations:
        # Check whether operation is to be skipped
        if any((isinstance(op, skip_op) for skip_op in [qml.Barrier, qml.Snapshot, qml.WireCut])):
            decomp_ops.append(op)

        # Check whether the operation is a global phase
        elif isinstance(op, qml.GlobalPhase):
            gphase_ops.append(op)

        # Check whether the operation is a Clifford or a T agte
        elif _check_clifford_t(op):
            decomp_ops.append(op)

        # Decompose and then iteratively go deeper via DFS
        else:
            if isinstance(op, qml.operation.Operation):
                # Single qubit unitary decomposition with ZXZ rotations
                if len(op.wires) == 1:
                    d_ops, g_ops = _one_qubit_decompose(op)
                    decomp_ops.extend(d_ops)
                    gphase_ops.append(g_ops)

                # Two qubit unitary decomposition with SU(4) rotations
                elif len(op.wires) == 2:
                    d_ops = _two_qubit_decompose(op)
                    decomp_ops.extend(d_ops)

                else:
                    try:
                        # Attempt decomposing the operation
                        md_ops = op.decomposition()

                        idx = 0  # might not be fast but at least is not recursive
                        while idx < len(md_ops):
                            md_op = md_ops[idx]

                            if not _check_clifford_t(md_op):
                                if len(md_op.wires) == 1:
                                    d_ops, g_ops = _one_qubit_decompose(md_op)
                                    gphase_ops.append(g_ops)
                                elif len(md_op.wires) == 2:
                                    d_ops = _two_qubit_decompose(md_op)
                                else:
                                    d_ops = md_op.decomposition()

                                # Expand the list and iterate over
                                del md_ops[idx]
                                md_ops[idx:idx] = d_ops
                            idx += 1

                        decomp_ops.extend(md_ops)

                    except Exception as exc:
                        raise ValueError(
                            f"Cannot unroll {op} into the Clifford+T basis as no rule exists for its decomposition"
                        ) from exc

    # Squeeze global phases into a single global phase
    gphase_op = qml.GlobalPhase(qml.math.sum(pd.data[0] for pd in gphase_ops))

    # TODO: Replace it with the real newsynth routine
    rz_to_clifford_plus_t = lambda op, epsilon: ([op], 0.0)

    new_ops, error = [gphase_op], 0
    for op in decomp_ops:
        if isinstance(op, qml.RZ):
            clifford_ops, err = rz_to_clifford_plus_t(op, epsilon)
            new_ops.extend(clifford_ops)
            error += err
        else:
            new_ops.append(op)

    new_tape = QuantumTape(new_ops, tape.measurements, shots=tape.shots)
    setattr(new_tape, "_qfunc_output", getattr(tape, "_qfunc_output", None))

    tapes, processing_fn = qml.compile(new_tape, pipeline=pipelines)

    return tapes, processing_fn
