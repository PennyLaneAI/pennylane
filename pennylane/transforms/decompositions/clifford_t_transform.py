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
import warnings
from itertools import product
from typing import Sequence, Callable

import pennylane as qml
from pennylane.ops import Adjoint
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
from pennylane.ops.op_math.decompositions.solovay_kitaev import sk_decomposition

# Single qubits Clifford+T gates in PL
_CLIFFORD_T_ONE_GATES = [
    qml.Identity,
    qml.X,
    qml.Y,
    qml.Z,
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
# note: _simplify_param makes use of their periodic nature,
# any additions to this, should be reflected there as well.
_PARAMETER_GATES = (qml.RX, qml.RY, qml.RZ, qml.Rot, qml.PhaseShift)

# Clifford+T gate set
_CLIFFORD_T_GATES = tuple(_CLIFFORD_T_ONE_GATES + _CLIFFORD_T_TWO_GATES) + (qml.GlobalPhase,)

# Gates to be skipped during decomposition
_SKIP_OP_TYPES = (qml.Barrier, qml.Snapshot, qml.WireCut)


def _check_clifford_op(op, use_decomposition=False):
    r"""Checks if an operator is Clifford or not.

    For a given unitary operator :math:`U` acting on :math:`N` qubits, this method checks that the
    transformation :math:`UPU^{\dagger}` maps the Pauli tensor products :math:`P = {I, X, Y, Z}^{\otimes N}`
    to Pauli tensor products using the decomposition of the matrix for :math:`U` in the Pauli basis.

    Args:
        op (~pennylane.operation.Operation): the operator that needs to be tested
        use_decomposition (bool): if ``True``, use operator's decomposition to compute the matrix, in case
            it doesn't define a ``compute_matrix`` method. Default is ``False``.

    Returns:
        Bool that represents whether the provided operator is Clifford or not.
    """

    # Check if matrix can be calculated for the operator
    if (not op.has_matrix and not use_decomposition) or (
        use_decomposition and not op.expand().wires
    ):
        return False

    # Compute the LCUs for the operator in Pauli basis
    pauli_terms = qml.pauli_decompose(qml.matrix(op), wire_order=op.wires, check_hermitian=False)
    pauli_terms_adj = qml.Hamiltonian(qml.math.conj(pauli_terms.coeffs), pauli_terms.ops)

    # Build Pauli tensor products P in the Hamiltonian form
    def pauli_group(x):
        return [qml.Identity(x), qml.X(x), qml.Y(x), qml.Z(x)]

    # Build PauliSentence and Hamiltonians representations for set P
    pauli_sens = [
        qml.pauli.pauli_sentence(qml.prod(*pauli))
        for pauli in product(*(pauli_group(idx) for idx in op.wires))
    ]
    pauli_hams = (pauli_sen.hamiltonian(wire_order=op.wires) for pauli_sen in pauli_sens)

    # Perform U@P@U^\dagger and check if the result exists in set P
    for pauli_prod in product([pauli_terms], pauli_hams, [pauli_terms_adj]):
        # hopefully op_math.prod scales better than matrix multiplication, i.e., O((2^N)^3)
        upu = qml.pauli.pauli_sentence(qml.prod(*pauli_prod))
        upu.simplify()
        # Pauli sum always lie outside set P
        if len(upu) != 1:
            return False

    return True


def check_clifford_t(op, use_decomposition=False):
    r"""Checks whether the gate is in the standard Clifford+T basis.

    For a given unitary operator :math:`U` acting on :math:`N` qubits, which is not a T-gate,
    this method checks that the transformation :math:`UPU^{\dagger}` maps the Pauli tensor products
    :math:`P = {I, X, Y, Z}^{\otimes N}` to Pauli tensor products using the decomposition of the
    matrix for :math:`U` in the Pauli basis.

    Args:
        op (~pennylane.operation.Operation): the operator that needs to be checked
        use_decomposition (bool): if ``True``, use operator's decomposition to compute the matrix, in case
            it doesn't define a ``compute_matrix`` method. Default is ``False``.

    Returns:
        Bool that represents whether the provided operator is Clifford+T or not.
    """
    # Get the base operation for an adjointed operation
    if isinstance(op, Adjoint):
        base = op.base
    else:
        base = None

    # Save time and check from the pre-computed list
    if isinstance(op, _CLIFFORD_T_GATES) or isinstance(base, _CLIFFORD_T_GATES):
        return True

    # Save time and check from the parameter of rotation gates
    if isinstance(op, _PARAMETER_GATES) or isinstance(base, _PARAMETER_GATES):
        theta = op.data[0]
        return (
            False
            if qml.math.is_abstract(theta)
            else qml.math.allclose(qml.math.mod(theta, math.pi), 0.0)
        )

    return _check_clifford_op(op, use_decomposition=use_decomposition)


def _simplify_param(theta, gate):
    r"""Check if the parameter allows simplification for the rotation gate.

    For the cases where theta is an integer multiple of π: (a) returns a global phase
    when even, and (b) returns combination of provided gate with global phase when odd.
    In rest of the other cases it would return None.
    """
    if qml.math.is_abstract(theta):  # pragma: no cover
        return None

    if qml.math.allclose(theta, 0.0, atol=1e-6):
        return [qml.GlobalPhase(0.0)]

    rem_, mod_ = qml.math.divide(theta, math.pi), qml.math.mod(theta, math.pi)
    if qml.math.allclose(mod_, 0.0, atol=1e-6):
        ops = [qml.GlobalPhase(theta / 2)]
        if qml.math.allclose(qml.math.mod(rem_, 2), 1.0, atol=1e-6):
            ops.append(gate)
        return ops

    return None


# pylint: disable= too-many-branches,
def _rot_decompose(op):
    r"""Decomposes a rotation operation: :class:`~.Rot`, :class:`~.RX`, :class:`~.RY`, :class:`~.RZ`,
    :class:`~.PhaseShift` into a basis composed of :class:`~.RZ`, :class:`~.S`, and :class:`~.Hadamard`.
    """
    d_ops = []
    # Extend for Rot operation with Rz.Ry.Rz decompositions
    if isinstance(op, qml.Rot):
        (phi, theta, omega), wires = op.parameters, op.wires
        for dec in [qml.RZ(phi, wires), qml.RY(theta, wires), qml.RZ(omega, wires)]:
            d_ops.extend(_rot_decompose(dec))
        return d_ops

    (theta,), wires = op.data, op.wires
    if isinstance(op, qml.ops.Adjoint):  # pylint: disable=no-member
        ops_ = _rot_decompose(op.base.adjoint())
    elif isinstance(op, qml.RX):
        ops_ = _simplify_param(theta, qml.X(wires))
        if ops_ is None:  # Use Rx = H @ Rz @ H
            ops_ = [qml.Hadamard(wires), qml.RZ(theta, wires), qml.Hadamard(wires)]
    elif isinstance(op, qml.RY):
        ops_ = _simplify_param(theta, qml.Y(wires))
        if ops_ is None:  # Use Ry = S @ H @ Rz @ H @ S.adjoint()
            ops_ = [
                qml.S(wires),
                qml.Hadamard(wires),
                qml.RZ(theta, wires),
                qml.Hadamard(wires),
                qml.adjoint(qml.S(wires)),
            ][::-1]
    elif isinstance(op, qml.RZ):
        ops_ = _simplify_param(theta, qml.Z(wires))
        if ops_ is None:
            ops_ = [qml.RZ(theta, wires)]
    elif isinstance(op, qml.PhaseShift):
        ops_ = _simplify_param(theta, qml.Z(wires))
        if ops_ is None:
            ops_ = [qml.RZ(theta, wires=wires), qml.GlobalPhase(-theta / 2)]
        else:
            ops_.append(qml.GlobalPhase(-theta / 2))

    else:
        raise ValueError(
            f"Operation {op} is not a supported Pauli rotation: qml.RX, qml.RY, qml.RZ, qml.Rot and qml.PhaseShift."
        )

    d_ops.extend(ops_)
    return d_ops


def _one_qubit_decompose(op):
    r"""Decomposition for single qubit operations using combination of :class:`~.RZ`, :class:`~.S`, and
    :class:`~.Hadamard`."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sd_ops = qml.ops.one_qubit_decomposition(
            qml.matrix(op), op.wires, "ZXZ", return_global_phase=True
        )
    # Get the global phase
    gphase_op = sd_ops.pop()

    # Decompose the rotation gates
    d_ops = []
    for sd_op in sd_ops:
        d_ops.extend(_rot_decompose(sd_op) if sd_op.num_params else [sd_op])

    return d_ops, gphase_op


def _two_qubit_decompose(op):
    r"""Decomposition for two qubit operations using combination of :class:`~.RZ`, :class:`~.S`,
    :class:`~.Hadamard`, and :class:`~.CNOT`."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        td_ops = qml.ops.two_qubit_decomposition(qml.matrix(op), op.wires)

    d_ops = []
    for td_op in td_ops:
        d_ops.extend(
            _rot_decompose(td_op) if td_op.num_params and td_op.num_wires == 1 else [td_op]
        )

    return d_ops


def _merge_param_gates(operations, merge_ops=None):
    """Merge the provided parameterized gates on the same wires that are adjacent to each other"""

    copied_ops = operations.copy()
    merged_ops, number_ops = [], 0

    while len(copied_ops) > 0:
        curr_gate = copied_ops.pop(0)

        # If gate is not to be merged, let it go
        if merge_ops is not None and curr_gate.name not in merge_ops:
            merged_ops.append(curr_gate)
            continue

        # If gate is in the merge_ops, update counter
        if curr_gate.name in merge_ops:
            number_ops += 1

        # Find the next gate that acts on the same wires
        next_gate_idx = find_next_gate(curr_gate.wires, copied_ops)
        if next_gate_idx is None:
            merged_ops.append(curr_gate)
            continue

        # Initialize the current angle and iterate until end of merge
        curr_params = curr_gate.parameters
        curr_intrfc = qml.math.get_deep_interface(curr_gate.parameters)
        cumulative_angles = qml.math.array(curr_params, dtype=float, like=curr_intrfc)
        next_gate = copied_ops[next_gate_idx]
        while curr_gate.name == next_gate.name and curr_gate.wires == next_gate.wires:
            cumulative_angles += qml.math.array(next_gate.parameters, like=curr_intrfc)
            # Check if the subsequent gate exists in the vicinity
            copied_ops.pop(next_gate_idx)
            next_gate_idx = find_next_gate(curr_gate.wires, copied_ops)
            if next_gate_idx is None:
                break
            next_gate = copied_ops[next_gate_idx]

        # Replace the current gate, add it to merged list and pop it from orginal one
        merged_ops.append(curr_gate.__class__(*cumulative_angles, wires=curr_gate.wires))

    return merged_ops, number_ops


# pylint: disable= too-many-nested-blocks, too-many-branches, too-many-statements, unnecessary-lambda-assignment
@transform
def clifford_t_decomposition(
    tape: QuantumTape,
    epsilon=1e-4,
    max_expansion=6,
    method="sk",
    **method_kwargs,
) -> (Sequence[QuantumTape], Callable):
    r"""Decomposes a circuit into the Clifford+T basis.

    This method first decomposes the gate operations to a basis comprised of Clifford, :class:`~.T`, :class:`~.RZ` and
    :class:`~.GlobalPhase` operations (and their adjoints). The Clifford gates include the following PennyLane operations:

    - Single qubit gates - :class:`~.Identity`, :class:`~.PauliX`, :class:`~.PauliY`, :class:`~.PauliZ`,
      :class:`~.SX`, :class:`~.S`, and :class:`~.Hadamard`.
    - Two qubit gates - :class:`~.CNOT`, :class:`~.CY`, :class:`~.CZ`, :class:`~.SWAP`, and :class:`~.ISWAP`.

    Then, the leftover single qubit :class:`~.RZ` operations are approximated in the Clifford+T basis with
    :math:`\epsilon > 0` error. By default, we use the Solovay-Kitaev algorithm described in
    `Dawson and Nielsen (2005) <https://arxiv.org/abs/quant-ph/0505030>`_ for this.

    Args:
        tape (QNode or QuantumTape or Callable): The quantum circuit to be decomposed.
        epsilon (float): The maximum permissible operator norm error of the complete circuit decomposition. Defaults to ``0.0001``.
        max_expansion (int): The depth to be used for tape expansion before manual decomposition to Clifford+T basis is applied.
        method (str): Method to be used for Clifford+T decomposition. Default value is ``"sk"`` for Solovay-Kitaev.
        **method_kwargs: Keyword argument to pass options for the ``method`` used for decompositions.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]: The transformed circuit as described
        in the :func:`qml.transform <pennylane.transform>`.

    **Keyword Arguments**

    - Solovay-Kitaev decomposition --
        **max_depth** (int), **basis_set** (list[str]), **basis_length** (int) -- arguments for the ``"sk"`` method,
        where the decomposition is performed using the :func:`~.sk_decomposition` method.

    Raises:
        ValueError: If a gate operation does not have a decomposition when required.
        NotImplementedError: If chosen decomposition ``method`` is not supported.

    .. seealso:: :func:`~.sk_decomposition` for Solovay-Kitaev decomposition.

    **Example**

    .. code-block:: python3

        @qml.qnode(qml.device("default.qubit"))
        def circuit(x, y):
            qml.RX(x, 0)
            qml.CNOT([0, 1])
            qml.RY(y, 0)
            return qml.expval(qml.Z(0))

        x, y = 1.1, 2.2
        decomposed_circuit = qml.transforms.clifford_t_decomposition(circuit)
        result = circuit(x, y)
        approx = decomposed_circuit(x, y)

    >>> qml.math.allclose(result, approx, atol=1e-4)
    True
    """
    with QueuingManager.stop_recording():
        # Build the basis set and the pipeline for intial compilation pass
        basis_set = [op.__name__ for op in _PARAMETER_GATES + _CLIFFORD_T_GATES]
        pipelines = [remove_barrier, commute_controlled, cancel_inverses, merge_rotations]

        # Compile the tape according to depth provided by the user and expand it
        [compiled_tape], _ = qml.compile(
            tape, pipelines, basis_set=basis_set, expand_depth=max_expansion
        )

        # Now iterate over the expanded tape operations
        decomp_ops, gphase_ops = [], []
        for op in compiled_tape.operations:
            # Check whether operation is to be skipped
            if isinstance(op, _SKIP_OP_TYPES):
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
                        d_ops, g_op = _one_qubit_decompose(op)
                        gphase_ops.append(g_op)
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
                                        d_ops, g_op = _one_qubit_decompose(md_op)
                                        gphase_ops.append(g_op)

                                # For the gates acting on two qubits
                                elif len(md_op.wires) == 2:
                                    # Resort to decomposing manually
                                    d_ops = _two_qubit_decompose(md_op)

                                # Final resort (should not enter in an ideal situtation)
                                else:
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
        merged_ops, number_ops = _merge_param_gates(decomp_ops, merge_ops=["RZ"])

        # Squeeze global phases into a single global phase
        new_operations = _fuse_global_phases(merged_ops + gphase_ops)

        # Compute the per-gate epsilon value
        epsilon /= number_ops or 1

        # Every decomposition implementation should have the following shape:
        # def decompose_fn(op: Operator, epsilon: float, **method_kwargs) -> List[Operator]
        # note: the last operator in the decomposition must be a GlobalPhase

        # Build the approximation set for Solovay-Kitaev decomposition
        if method == "sk":
            decompose_fn = sk_decomposition

        else:
            raise NotImplementedError(
                f"Currently we only support Solovay-Kitaev ('sk') decompostion, got {method}"
            )

        decomp_ops = []
        phase = new_operations.pop().data[0]
        for op in new_operations:
            if isinstance(op, qml.RZ):
                clifford_ops = decompose_fn(op, epsilon, **method_kwargs)
                phase += qml.math.convert_like(clifford_ops.pop().data[0], phase)
                decomp_ops.extend(clifford_ops)
            else:
                decomp_ops.append(op)

        # check if phase is non-zero for non jax-jit cases
        if qml.math.is_abstract(phase) or not qml.math.allclose(phase, 0.0):
            decomp_ops.append(qml.GlobalPhase(phase))

    # Construct a new tape with the expanded set of operations
    new_tape = type(tape)(decomp_ops, compiled_tape.measurements, shots=tape.shots)

    # Perform a final attempt of simplification before return
    [new_tape], _ = cancel_inverses(new_tape)

    def null_postprocessing(results):
        """A postprocesing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        """
        return results[0]

    return [new_tape], null_postprocessing
