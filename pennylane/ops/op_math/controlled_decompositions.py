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
"""
This submodule defines functions to decompose controlled operations
"""

from copy import copy
from typing import Literal, Optional

import numpy as np
import numpy.linalg as npl

import pennylane as qml
from pennylane import math
from pennylane.math.decomposition import zyz_rotation_angles
from pennylane.operation import Operation, Operator
from pennylane.wires import Wires, WiresLike


def _is_single_qubit_special_unitary(op):
    mat = op.matrix()
    det = mat[0, 0] * mat[1, 1] - mat[0, 1] * mat[1, 0]
    return qml.math.allclose(det, 1)


def _convert_to_real_diagonal(q: np.ndarray) -> np.ndarray:
    """
    Change the phases of Q so the main diagonal is real, and return the modified Q.
    """
    exp_angles = np.angle(np.diag(q))
    return q * np.exp(-1j * exp_angles).reshape((1, 2))


def _param_su2(ar: float, ai: float, br: float, bi: float):
    """
    Create a matrix in the SU(2) form from complex parameters a, b.
    The resulting matrix is not guaranteed to be in SU(2), unless |a|^2 + |b|^2 = 1.
    """
    return np.array([[ar + 1j * ai, -br + 1j * bi], [br + 1j * bi, ar + 1j * -ai]])


def _bisect_compute_a(u: np.ndarray):
    """
    Given the U matrix, compute the A matrix such that
    At x A x At x A x = U
    where At is the adjoint of A
    and x is the Pauli X matrix.
    """
    x = np.real(u[0, 1])
    z = u[1, 1]
    zr = np.real(z)
    zi = np.imag(z)
    if np.isclose(zr, -1):
        # special case [[-1, 0], [0, -1]]
        # would cause divide by 0 with the other formula, so we use hardcoded solution
        return np.array([[1, -1], [1, 1]]) * 2**-0.5
    ar = np.sqrt((np.sqrt((zr + 1) / 2) + 1) / 2)
    mul = 1 / (2 * np.sqrt((zr + 1) * (np.sqrt((zr + 1) / 2) + 1)))
    ai = zi * mul
    br = x * mul
    bi = 0
    return _param_su2(ar, ai, br, bi)


def _bisect_compute_b(u: np.ndarray):
    """
    Given the U matrix, compute the B matrix such that
    H Bt x B x H = U
    where Bt is the adjoint of B,
    H is the Hadamard matrix,
    and x is the Pauli X matrix.
    """
    sqrt = np.sqrt
    Abs = np.abs
    w = np.real(u[0, 0])
    s = np.real(u[1, 0])
    t = np.imag(u[1, 0])
    if np.isclose(s, 0):
        b = 0
        if np.isclose(t, 0):
            if w < 0:
                c = 0
                d = sqrt(-w)
            else:
                c = sqrt(w)
                d = 0
        else:
            c = sqrt(2 - 2 * w) * (-w / 2 - 1 / 2) / t
            d = sqrt(2 - 2 * w) / 2
    elif np.isclose(t, 0):
        b = (1 / 2 - w / 2) * sqrt(2 * w + 2) / s
        c = sqrt(2 * w + 2) / 2
        d = 0
    else:
        b = sqrt(2) * s * sqrt((1 - w) / (s**2 + t**2)) * Abs(t) / (2 * t)
        c = sqrt(2) * sqrt((1 - w) / (s**2 + t**2)) * (w + 1) * Abs(t) / (2 * t)
        d = -sqrt(2) * sqrt((1 - w) / (s**2 + t**2)) * Abs(t) / 2
    return _param_su2(c, d, b, 0)


def _multi_controlled_zyz(
    rot_angles,
    global_phase,
    target_wire: Wires,
    control_wires: Wires,
    work_wires: Optional[Wires] = None,
) -> list[Operator]:
    # The decomposition of zyz for special unitaries with multiple control wires
    # defined in Lemma 7.9 of https://arxiv.org/pdf/quant-ph/9503016

    if not qml.math.allclose(0.0, global_phase, atol=1e-6, rtol=0):
        raise ValueError(f"The global_phase should be zero, instead got: {global_phase}.")

    # Unpack the rotation angles
    phi, theta, omega = rot_angles

    # We use the conditional statements to account when decomposition is ran within a queue
    decomp = []

    cop_wires = (control_wires[-1], target_wire[0])

    # Add operator A
    if not qml.math.allclose(0.0, phi, atol=1e-8, rtol=0):
        decomp.append(qml.CRZ(phi, wires=cop_wires))
    if not qml.math.allclose(0.0, theta / 2, atol=1e-8, rtol=0):
        decomp.append(qml.CRY(theta / 2, wires=cop_wires))

    decomp.append(qml.ctrl(qml.X(target_wire), control=control_wires[:-1], work_wires=work_wires))

    # Add operator B
    if not qml.math.allclose(0.0, theta / 2, atol=1e-8, rtol=0):
        decomp.append(qml.CRY(-theta / 2, wires=cop_wires))
    if not qml.math.allclose(0.0, -(phi + omega) / 2, atol=1e-6, rtol=0):
        decomp.append(qml.CRZ(-(phi + omega) / 2, wires=cop_wires))

    decomp.append(qml.ctrl(qml.X(target_wire), control=control_wires[:-1], work_wires=work_wires))

    # Add operator C
    if not qml.math.allclose(0.0, (omega - phi) / 2, atol=1e-8, rtol=0):
        decomp.append(qml.CRZ((omega - phi) / 2, wires=cop_wires))

    return decomp


def _single_control_zyz(rot_angles, global_phase, target_wire, control_wires: Wires):
    # The zyz decomposition of a general unitary with single control wire
    # defined in Lemma 7.9 of https://arxiv.org/pdf/quant-ph/9503016

    # Unpack the rotation angles
    phi, theta, omega = rot_angles
    # We use the conditional statements to account when decomposition is ran within a queue
    decomp = []
    # Add negative of global phase. Compare definition of qml.GlobalPhase and Ph(delta) from section 4.1 of Barenco et al.
    if not qml.math.allclose(0.0, global_phase, atol=1e-8, rtol=0):
        decomp.append(
            qml.ctrl(qml.GlobalPhase(phi=-global_phase, wires=target_wire), control=control_wires)
        )
    # Add operator A
    if not qml.math.allclose(0.0, phi, atol=1e-8, rtol=0):
        decomp.append(qml.RZ(phi, wires=target_wire))
    if not qml.math.allclose(0.0, theta / 2, atol=1e-8, rtol=0):
        decomp.append(qml.RY(theta / 2, wires=target_wire))

    decomp.append(qml.ctrl(qml.X(target_wire), control=control_wires))

    # Add operator B
    if not qml.math.allclose(0.0, theta / 2, atol=1e-8, rtol=0):
        decomp.append(qml.RY(-theta / 2, wires=target_wire))
    if not qml.math.allclose(0.0, -(phi + omega) / 2, atol=1e-6, rtol=0):
        decomp.append(qml.RZ(-(phi + omega) / 2, wires=target_wire))

    decomp.append(qml.ctrl(qml.X(target_wire), control=control_wires))

    # Add operator C
    if not qml.math.allclose(0.0, (omega - phi) / 2, atol=1e-8, rtol=0):
        decomp.append(qml.RZ((omega - phi) / 2, wires=target_wire))

    return decomp


def ctrl_decomp_zyz(
    target_operation: Operator, control_wires: Wires, work_wires: Optional[Wires] = None
) -> list[Operator]:
    """Decompose the controlled version of a target single-qubit operation

    This function decomposes both single and multiple controlled single-qubit
    target operations using the decomposition defined in Lemma 4.3 and Lemma 5.1
    for single `controlled_wires`, and Lemma 7.9 for multiple `controlled_wires`
    from `Barenco et al. (1995) <https://arxiv.org/abs/quant-ph/9503016>`_.

    Args:
        target_operation (~.operation.Operator): the target operation to decompose
        control_wires (~.wires.Wires): the control wires of the operation.

    Returns:
        list[Operation]: the decomposed operations

    Raises:
        ValueError: if ``target_operation`` is not a single-qubit operation

    **Example**

    We can create a controlled operation using `qml.ctrl`, or by creating the
    decomposed controlled version of using `qml.ctrl_decomp_zyz`.

    .. code-block:: python

        import pennylane as qml

        dev = qml.device("default.qubit", wires=2)

        @qml.qnode(dev)
        def expected_circuit(op):
            qml.Hadamard(wires=0)
            qml.ctrl(op, [0])
            return qml.probs()

        @qml.qnode(dev)
        def decomp_circuit(op):
            qml.Hadamard(wires=0)
            qml.ops.ctrl_decomp_zyz(op, [0])
            return qml.probs()

    Measurements on both circuits will give us the same results:

    >>> op = qml.RX(0.123, wires=1)
    >>> expected_circuit(op)
    tensor([0.5       , 0.        , 0.49811126, 0.00188874], requires_grad=True)

    >>> decomp_circuit(op)
    tensor([0.5       , 0.        , 0.49811126, 0.00188874], requires_grad=True)

    """
    if len(target_operation.wires) != 1:
        raise ValueError(
            "The target operation must be a single-qubit operation, instead "
            f"got {target_operation.__class__.__name__}."
        )
    control_wires = Wires(control_wires)

    target_wire = target_operation.wires

    if isinstance(target_operation, Operation):
        try:
            rot_angles = target_operation.single_qubit_rot_angles()
        except NotImplementedError:
            rot_angles = zyz_rotation_angles(qml.matrix(target_operation))
    else:
        rot_angles = zyz_rotation_angles(qml.matrix(target_operation))

    _, global_phase = math.convert_to_su2(qml.matrix(target_operation), return_global_phase=True)

    return (
        _multi_controlled_zyz(rot_angles, global_phase, target_wire, control_wires, work_wires)
        if len(control_wires) > 1
        else _single_control_zyz(rot_angles, global_phase, target_wire, control_wires)
    )


def _ctrl_decomp_bisect_od(
    u: np.ndarray,
    target_wire: Wires,
    control_wires: Wires,
):
    """Decompose the controlled version of a target single-qubit operation

    Not backpropagation compatible (as currently implemented). Use only with numpy.

    This function decomposes a controlled single-qubit target operation using the
    decomposition defined in section 3.1, Theorem 1 of
    `Vale et al. (2023) <https://arxiv.org/abs/2302.06377>`_.

    The target operation's matrix must have a real off-diagonal for this specialized method to work.

    .. warning:: This method will add a global phase for target operations that do not
        belong to the SU(2) group.

    Args:
        u (np.ndarray): the target operation's matrix
        target_wire (~.wires.Wires): the target wire of the controlled operation
        control_wires (~.wires.Wires): the control wires of the operation

    Returns:
        list[Operation]: the decomposed operations

    Raises:
        ValueError: if ``u`` does not have a real off-diagonal

    """
    ui = np.imag(u)
    if not np.isclose(ui[1, 0], 0) or not np.isclose(ui[0, 1], 0):
        raise ValueError(f"Target operation's matrix must have real off-diagonal, but it is {u}")

    a = _bisect_compute_a(u)

    mid = (len(control_wires) + 1) // 2  # for odd n, make control_k1 bigger
    control_k1 = control_wires[:mid]
    control_k2 = control_wires[mid:]

    def component():
        return [
            qml.ctrl(qml.X(target_wire), control=control_k1, work_wires=control_k2),
            qml.QubitUnitary(a, target_wire),
            qml.ctrl(qml.X(target_wire), control=control_k2, work_wires=control_k1),
            qml.adjoint(qml.QubitUnitary(a, target_wire)),
        ]

    return component() + component()


def _ctrl_decomp_bisect_md(
    u: np.ndarray,
    target_wire: Wires,
    control_wires: Wires,
):
    """Decompose the controlled version of a target single-qubit operation

    Not backpropagation compatible (as currently implemented). Use only with numpy.

    This function decomposes a controlled single-qubit target operation using the
    decomposition defined in section 3.1, Theorem 2 of
    `Vale et al. (2023) <https://arxiv.org/abs/2302.06377>`_.

    The target operation's matrix must have a real main-diagonal for this specialized method to work.

    .. warning:: This method will add a global phase for target operations that do not
        belong to the SU(2) group.

    Args:
        u (np.ndarray): the target operation's matrix
        target_wire (~.wires.Wires): the target wire of the controlled operation
        control_wires (~.wires.Wires): the control wires of the operation

    Returns:
        list[Operation]: the decomposed operations

    Raises:
        ValueError: if ``u`` does not have a real main-diagonal

    """
    ui = np.imag(u)
    if not np.isclose(ui[0, 0], 0) or not np.isclose(ui[1, 1], 0):
        raise ValueError(f"Target operation's matrix must have real main-diagonal, but it is {u}")

    h_matrix = qml.Hadamard.compute_matrix()
    mod_u = h_matrix @ u @ h_matrix

    decomposition = [qml.Hadamard(target_wire)]
    decomposition += _ctrl_decomp_bisect_od(mod_u, target_wire, control_wires)
    decomposition.append(qml.Hadamard(target_wire))

    return decomposition


def _ctrl_decomp_bisect_general(
    u: np.ndarray,
    target_wire: Wires,
    control_wires: Wires,
):
    """Decompose the controlled version of a target single-qubit operation

    Not backpropagation compatible (as currently implemented). Use only with numpy.

    This function decomposes a controlled single-qubit target operation using the
    decomposition defined in section 3.2 of
    `Vale et al. (2023) <https://arxiv.org/abs/2302.06377>`_.

    .. warning:: This method will add a global phase for target operations that do not
        belong to the SU(2) group.

    Args:
        u (np.ndarray): the target operation's matrix
        target_wire (~.wires.Wires): the target wire of the controlled operation
        control_wires (~.wires.Wires): the control wires of the operation

    Returns:
        list[Operation]: the decomposed operations
    """
    x_matrix = qml.X.compute_matrix()
    h_matrix = qml.Hadamard.compute_matrix()
    alternate_h_matrix = x_matrix @ h_matrix @ x_matrix

    d, q = npl.eig(u)
    d = np.diag(d)
    q = _convert_to_real_diagonal(q)
    b = _bisect_compute_b(q)
    c1 = b @ alternate_h_matrix
    c2t = b @ h_matrix

    mid = (len(control_wires) + 1) // 2  # for odd n, make control_k1 bigger
    control_k1 = control_wires[:mid]
    control_k2 = control_wires[mid:]

    component = [
        qml.QubitUnitary(c2t, target_wire),
        qml.ctrl(qml.X(target_wire), control=control_k2, work_wires=control_k1),
        qml.adjoint(qml.QubitUnitary(c1, target_wire)),
        qml.ctrl(qml.X(target_wire), control=control_k1, work_wires=control_k2),
    ]

    od_decomp = _ctrl_decomp_bisect_od(d, target_wire, control_wires)

    # cancel two identical multicontrolled x gates
    qml.QueuingManager.remove(component[3])
    qml.QueuingManager.remove(od_decomp[0])

    adjoint_component = [qml.adjoint(copy(op), lazy=False) for op in reversed(component)]

    return component[0:3] + od_decomp[1:] + adjoint_component


def ctrl_decomp_bisect(
    target_operation: Operator,
    control_wires: Wires,
):
    """Decompose the controlled version of a target single-qubit operation

    Not backpropagation compatible (as currently implemented). Use only with numpy.

    Automatically selects the best algorithm based on the matrix (uses specialized more efficient
    algorithms if the matrix has a certain form, otherwise falls back to the general algorithm).
    These algorithms are defined in section 3.1 and 3.2 of
    `Vale et al. (2023) <https://arxiv.org/abs/2302.06377>`_.

    .. warning:: This method will add a global phase for target operations that do not
        belong to the SU(2) group.

    Args:
        target_operation (~.operation.Operator): the target operation to decompose
        control_wires (~.wires.Wires): the control wires of the operation

    Returns:
        list[Operation]: the decomposed operations

    Raises:
        ValueError: if ``target_operation`` is not a single-qubit operation

    **Example:**

    >>> op = qml.T(0) # uses OD algorithm
    >>> print(qml.draw(ctrl_decomp_bisect, wire_order=(0,1,2,3,4,5), show_matrices=False)(op, (1,2,3,4,5)))
    0: ─╭X──U(M0)─╭X──U(M0)†─╭X──U(M0)─╭X──U(M0)†─┤
    1: ─├●────────│──────────├●────────│──────────┤
    2: ─├●────────│──────────├●────────│──────────┤
    3: ─╰●────────│──────────╰●────────│──────────┤
    4: ───────────├●───────────────────├●─────────┤
    5: ───────────╰●───────────────────╰●─────────┤
    >>> op = qml.QubitUnitary([[0,1j],[1j,0]], 0) # uses MD algorithm
    >>> print(qml.draw(ctrl_decomp_bisect, wire_order=(0,1,2,3,4,5), show_matrices=False)(op, (1,2,3,4,5)))
    0: ──H─╭X──U(M0)─╭X──U(M0)†─╭X──U(M0)─╭X──U(M0)†──H─┤
    1: ────├●────────│──────────├●────────│─────────────┤
    2: ────├●────────│──────────├●────────│─────────────┤
    3: ────╰●────────│──────────╰●────────│─────────────┤
    4: ──────────────├●───────────────────├●────────────┤
    5: ──────────────╰●───────────────────╰●────────────┤
    >>> op = qml.Hadamard(0) # uses general algorithm
    >>> print(qml.draw(ctrl_decomp_bisect, wire_order=(0,1,2,3,4,5), show_matrices=False)(op, (1,2,3,4,5)))
    0: ──U(M0)─╭X──U(M1)†──U(M2)─╭X──U(M2)†─╭X──U(M2)─╭X──U(M2)†─╭X──U(M1)─╭X──U(M0)─┤
    1: ────────│─────────────────│──────────├●────────│──────────├●────────│─────────┤
    2: ────────│─────────────────│──────────├●────────│──────────├●────────│─────────┤
    3: ────────│─────────────────│──────────╰●────────│──────────╰●────────│─────────┤
    4: ────────├●────────────────├●───────────────────├●───────────────────├●────────┤
    5: ────────╰●────────────────╰●───────────────────╰●───────────────────╰●────────┤

    """
    if len(target_operation.wires) > 1:
        raise ValueError(
            "The target operation must be a single-qubit operation, instead "
            f"got {target_operation}."
        )
    target_matrix = target_operation.matrix()
    target_wire = target_operation.wires

    target_matrix = math.convert_to_su2(target_matrix)
    target_matrix_imag = np.imag(target_matrix)

    if np.isclose(target_matrix_imag[1, 0], 0) and np.isclose(target_matrix_imag[0, 1], 0):
        # Real off-diagonal specialized algorithm - 16n+O(1) CNOTs
        return _ctrl_decomp_bisect_od(target_matrix, target_wire, control_wires)
    if np.isclose(target_matrix_imag[0, 0], 0) and np.isclose(target_matrix_imag[1, 1], 0):
        # Real main-diagonal specialized algorithm - 16n+O(1) CNOTs
        return _ctrl_decomp_bisect_md(target_matrix, target_wire, control_wires)
    # General algorithm - 20n+O(1) CNOTs
    return _ctrl_decomp_bisect_general(target_matrix, target_wire, control_wires)


def decompose_mcx(
    control_wires, target_wire, work_wires, work_wire_type: Literal["clean", "dirty"] = "clean"
):
    """Decomposes the multi-controlled PauliX"""

    n_ctrl_wires, n_work_wires = len(control_wires), len(work_wires)
    if n_ctrl_wires == 1:
        return [qml.CNOT(wires=control_wires + Wires(target_wire))]
    if n_ctrl_wires == 2:
        return qml.Toffoli.compute_decomposition(wires=control_wires + Wires(target_wire))

    if n_work_wires >= n_ctrl_wires - 2:
        # Lemma 7.2 of `Barenco et al. (1995) <https://arxiv.org/abs/quant-ph/9503016>`_
        return _decompose_mcx_with_many_workers(control_wires, target_wire, work_wires)
    if n_work_wires >= 2:
        return _decompose_mcx_with_two_workers(
            control_wires, target_wire, work_wires[0:2], work_wire_type
        )
    if n_work_wires == 1:
        return _decompose_mcx_with_one_worker_kg24(
            control_wires, target_wire, work_wires[0], work_wire_type
        )

    # Lemma 7.5
    with qml.QueuingManager.stop_recording():
        op = qml.X(target_wire)
    return _decompose_multicontrolled_unitary(op, control_wires)


def _decompose_multicontrolled_unitary(op, control_wires):
    """Decomposes general multi controlled unitary with no work wires
    Follows approach from Lemma 7.5 combined with 7.3 and 7.2 of
    https://arxiv.org/abs/quant-ph/9503016.

    We are assuming this decomposition is used only in the general cases
    """
    if not op.has_matrix or len(op.wires) != 1:
        raise ValueError(
            "The target operation must be a single-qubit operation with a matrix representation"
        )

    target_wire = op.wires
    if len(control_wires) == 0:
        return [op]
    if len(control_wires) == 1:
        return ctrl_decomp_zyz(op, control_wires)
    if _is_single_qubit_special_unitary(op):
        return ctrl_decomp_bisect(op, control_wires)
    # use recursive decomposition of general gate
    return _decompose_recursive(op, 1.0, control_wires, target_wire, Wires([]))


def _decompose_recursive(op, power, control_wires, target_wire, work_wires):
    """Decompose multicontrolled operator recursively using lemma 7.5
    Number of gates in decomposition are: O(len(control_wires)^2)
    """
    if len(control_wires) == 1:
        with qml.QueuingManager.stop_recording():
            powered_op = qml.pow(op, power, lazy=True)
        return ctrl_decomp_zyz(powered_op, control_wires)

    with qml.QueuingManager.stop_recording():
        cnots = decompose_mcx(
            control_wires=control_wires[:-1],
            target_wire=control_wires[-1],
            work_wires=work_wires + target_wire,
            work_wire_type="dirty",
        )
    with qml.QueuingManager.stop_recording():
        powered_op = qml.pow(op, 0.5 * power, lazy=True)
        powered_op_adj = qml.adjoint(powered_op, lazy=True)

    if qml.QueuingManager.recording():
        decomposition = [
            *ctrl_decomp_zyz(powered_op, control_wires[-1]),
            *(qml.apply(o) for o in cnots),
            *ctrl_decomp_zyz(powered_op_adj, control_wires[-1]),
            *(qml.apply(o) for o in cnots),
            *_decompose_recursive(
                op, 0.5 * power, control_wires[:-1], target_wire, control_wires[-1] + work_wires
            ),
        ]
    else:
        decomposition = [
            *ctrl_decomp_zyz(powered_op, control_wires[-1]),
            *cnots,
            *ctrl_decomp_zyz(powered_op_adj, control_wires[-1]),
            *cnots,
            *_decompose_recursive(
                op, 0.5 * power, control_wires[:-1], target_wire, control_wires[-1] + work_wires
            ),
        ]
    return decomposition


def _decompose_mcx_with_many_workers(control_wires, target_wire, work_wires):
    """Decomposes the multi-controlled PauliX gate using the approach in Lemma 7.2 of
    https://arxiv.org/abs/quant-ph/9503016, which requires a suitably large register of
    work wires"""
    num_work_wires_needed = len(control_wires) - 2
    work_wires = work_wires[:num_work_wires_needed]

    work_wires_reversed = list(reversed(work_wires))
    control_wires_reversed = list(reversed(control_wires))

    gates = []

    for i in range(len(work_wires)):
        ctrl1 = control_wires_reversed[i]
        ctrl2 = work_wires_reversed[i]
        t = target_wire if i == 0 else work_wires_reversed[i - 1]
        gates.append(qml.Toffoli(wires=[ctrl1, ctrl2, t]))

    gates.append(qml.Toffoli(wires=[*control_wires[:2], work_wires[0]]))

    for i in reversed(range(len(work_wires))):
        ctrl1 = control_wires_reversed[i]
        ctrl2 = work_wires_reversed[i]
        t = target_wire if i == 0 else work_wires_reversed[i - 1]
        gates.append(qml.Toffoli(wires=[ctrl1, ctrl2, t]))

    for i in range(len(work_wires) - 1):
        ctrl1 = control_wires_reversed[i + 1]
        ctrl2 = work_wires_reversed[i + 1]
        t = work_wires_reversed[i]
        gates.append(qml.Toffoli(wires=[ctrl1, ctrl2, t]))

    gates.append(qml.Toffoli(wires=[*control_wires[:2], work_wires[0]]))

    for i in reversed(range(len(work_wires) - 1)):
        ctrl1 = control_wires_reversed[i + 1]
        ctrl2 = work_wires_reversed[i + 1]
        t = work_wires_reversed[i]
        gates.append(qml.Toffoli(wires=[ctrl1, ctrl2, t]))

    return gates


def _decompose_mcx_with_one_worker_b95(control_wires, target_wire, work_wire):
    """
    Decomposes the multi-controlled PauliX gate using the approach in Lemma 7.3 of
    `Barenco et al. (1995) <https://arxiv.org/abs/quant-ph/9503016>`_, which requires
    a single work wire. This approach requires O(16k) CX gates, where k is the number of control wires.
    """
    tot_wires = len(control_wires) + 2
    partition = int(np.ceil(tot_wires / 2))

    first_part = control_wires[:partition]
    second_part = control_wires[partition:]

    gates = [
        qml.ctrl(qml.X(work_wire), control=first_part, work_wires=second_part + target_wire),
        qml.ctrl(qml.X(target_wire), control=second_part + work_wire, work_wires=first_part),
        qml.ctrl(qml.X(work_wire), control=first_part, work_wires=second_part + target_wire),
        qml.ctrl(qml.X(target_wire), control=second_part + work_wire, work_wires=first_part),
    ]

    return gates


def _linear_depth_ladder_ops(wires: WiresLike) -> tuple[list[Operator], int]:
    r"""
    Helper function to create linear-depth ladder operations used in Khattar and Gidney's MCX synthesis.
    In particular, this implements Step-1 and Step-2 on Fig. 3 of [1] except for the first and last
    CCX gates.

    Preconditions:
        - The number of wires must be greater than 2.

    Args:
        wires (Wires): Wires to apply the ladder operations on.

    Returns:
        tuple[list[Operator], int]: Linear-depth ladder circuit and the index of control qubit to
        apply the final CCX gate.

    References:
        1. Khattar and Gidney, Rise of conditionally clean ancillae for optimizing quantum circuits
        `arXiv:2407.17966 <https://arxiv.org/abs/2407.17966>`__
    """

    n = len(wires)
    assert n > 2, "n_ctrls > 2 to use MCX ladder. Otherwise, use CCX"

    gates = []
    # up-ladder
    for i in range(1, n - 2, 2):
        gates.append(qml.Toffoli(wires=[wires[i + 1], wires[i + 2], wires[i]]))
        gates.append(qml.PauliX(wires=wires[i]))

    # down-ladder
    if n % 2 == 0:
        ctrl_1, ctrl_2, target = n - 3, n - 5, n - 6
    else:
        ctrl_1, ctrl_2, target = n - 1, n - 4, n - 5

    if target >= 0:
        gates.append(qml.Toffoli(wires=[wires[ctrl_1], wires[ctrl_2], wires[target]]))
        gates.append(qml.PauliX(wires=wires[target]))

    for i in range(target, 1, -2):
        gates.append(qml.Toffoli(wires=[wires[i], wires[i - 1], wires[i - 2]]))
        gates.append(qml.PauliX(wires=wires[i - 2]))

    final_ctrl = max(0, 5 - n)

    return gates, final_ctrl


def _decompose_mcx_with_one_worker_kg24(
    control_wires: WiresLike,
    target_wire: int,
    work_wire: int,
    work_wire_type: Literal["clean", "dirty"] = "clean",
) -> list[Operator]:
    r"""
    Synthesise a multi-controlled X gate with :math:`k` controls using :math:`1` ancillary qubit. It
    produces a circuit with :math:`2k-3` Toffoli gates and depth :math:`O(k)` if the ancilla is clean
    and :math:`4k-3` Toffoli gates and depth :math:`O(k)` if the ancilla is dirty as described in
    Sec. 5.1 of [1].

    Args:
        control_wires (Wires): the control wires
        target_wire (int): the target wire
        work_wires (Wires): the work wires used to decompose the gate
        work_wire_type (string): If "dirty", perform un-computation. Default is "clean".

    Returns:
        list[Operator]: the synthesized quantum circuit

    References:
        1. Khattar and Gidney, Rise of conditionally clean ancillae for optimizing quantum circuits
        `arXiv:2407.17966 <https://arxiv.org/abs/2407.17966>`__
    """

    gates = []
    gates.append(qml.Toffoli(wires=[control_wires[0], control_wires[1], work_wire]))
    ladder_ops, final_ctrl = _linear_depth_ladder_ops(control_wires)
    gates += ladder_ops
    gates.append(qml.Toffoli(wires=[work_wire, control_wires[final_ctrl], target_wire]))
    gates += ladder_ops[::-1]
    gates.append(qml.Toffoli(wires=[control_wires[0], control_wires[1], work_wire]))

    if work_wire_type == "dirty":
        # perform toggle-detection if ancilla is dirty
        gates += ladder_ops
        gates.append(qml.Toffoli(wires=[work_wire, control_wires[final_ctrl], target_wire]))
        gates += ladder_ops[::-1]

    return gates


def _n_parallel_ccx_x(
    control_wires_x: WiresLike, control_wires_y: WiresLike, target_wires: WiresLike
) -> list[Operation]:
    r"""
    Construct a quantum circuit for creating n-condionally clean ancillae using 3n qubits. This
    implements Fig. 4a of [1]. Each wire is of the same size :math:`n`.

    Args:
        control_wires_x (Wires): The control wires for register 1.
        control_wires_y (Wires): The control wires for register 2.
        target_wires (Wires): The wires for target register.

    Returns:
        list[Operation]: The quantum circuit for creating n-condionally clean ancillae.

    References:
        1. Khattar and Gidney, Rise of conditionally clean ancillae for optimizing quantum circuits
        `arXiv:2407.17966 <https://arxiv.org/abs/2407.17966>`__
    """

    assert (
        len(control_wires_x) == len(control_wires_y) == len(target_wires)
    ), "The number of wires must be the same for x, y, and target."

    gates = []
    for i, ctrl_x in enumerate(control_wires_x):
        gates.append(qml.X(wires=target_wires[i]))
        gates.append(qml.Toffoli(wires=[ctrl_x, control_wires_y[i], target_wires[i]]))

    return gates


def _build_logn_depth_ccx_ladder(
    work_wire: int, control_wires: WiresLike
) -> tuple[list[Operator], list[Operator]]:
    r"""
    Helper function to build a log-depth ladder compose of CCX and X gates as shown in Fig. 4b of [1].

    Args:
        work_wire (int): The work wire.
        control_wires (list[Wire]): The control wires.

    Returns:
        tuple[list[Operator], WiresLike: log-depth ladder circuit of cond. clean ancillae and
        control_wires to apply the linear-depth MCX gate on.

    References:
        1. Khattar and Gidney, Rise of conditionally clean ancillae for optimizing quantum circuits
        `arXiv:2407.17966 <https://arxiv.org/abs/2407.17966>`__
    """

    gates = []
    anc = [work_wire]
    final_ctrls = []

    while len(control_wires) > 1:
        next_batch_len = min(len(anc) + 1, len(control_wires))
        control_wires, nxt_batch = control_wires[next_batch_len:], control_wires[:next_batch_len]
        new_anc = []
        while len(nxt_batch) > 1:
            ccx_n = len(nxt_batch) // 2
            st = int(len(nxt_batch) % 2)
            ccx_x, ccx_y, ccx_t = (
                nxt_batch[st : st + ccx_n],
                nxt_batch[st + ccx_n :],
                anc[-ccx_n:],
            )
            assert len(ccx_x) == len(ccx_y) == len(ccx_t) == ccx_n >= 1
            if ccx_t != [work_wire]:
                gates += _n_parallel_ccx_x(ccx_x, ccx_y, ccx_t)
            else:
                gates.append(qml.Toffoli(wires=[ccx_x[0], ccx_y[0], ccx_t[0]]))
            new_anc += nxt_batch[st:]  # newly created cond. clean ancilla
            nxt_batch = ccx_t + nxt_batch[:st]
            anc = anc[:-ccx_n]

        anc = sorted(anc + new_anc)
        final_ctrls += nxt_batch

    final_ctrls += control_wires
    final_ctrls = sorted(final_ctrls)
    final_ctrls.remove(work_wire)  #                        # exclude ancilla
    return gates, final_ctrls


def _decompose_mcx_with_two_workers(
    control_wires: WiresLike,
    target_wire: int,
    work_wires: WiresLike,
    work_wire_type: Literal["clean", "dirty"] = "clean",
) -> list[Operator]:
    r"""
    Synthesise a multi-controlled X gate with :math:`k` controls using :math:`2` ancillary qubits.
    It produces a circuit with :math:`2k-3` Toffoli gates and depth :math:`O(\log(k))` if using
    clean ancillae, and :math:`4k-8` Toffoli gates and depth :math:`O(\log(k))` if using dirty
    ancillae as described in Sec. 5 of [1].

    Args:
        control_wires (Wires): The control wires.
        target_wire (int): The target wire.
        work_wires (Wires): The work wires.
        work_wire_type (string): If "dirty" perform uncomputation after we're done. Default is "clean".

    Returns:
        list[Operator]: The synthesized quantum circuit.

    References:
        1. Khattar and Gidney, Rise of conditionally clean ancillae for optimizing quantum circuits
        `arXiv:2407.17966 <https://arxiv.org/abs/2407.17966>`__
    """

    if len(work_wires) < 2:
        raise ValueError("At least 2 work wires are needed for this decomposition.")

    gates = []
    ladder_ops, final_ctrls = _build_logn_depth_ccx_ladder(work_wires[0], control_wires)
    gates += ladder_ops
    if len(final_ctrls) == 1:  # Already a toffoli
        gates.append(qml.Toffoli(wires=[work_wires[0], final_ctrls[0], target_wire]))
    else:
        mid_mcx = _decompose_mcx_with_one_worker_kg24(
            work_wires[0:1] + final_ctrls, target_wire, work_wires[1], work_wire_type="clean"
        )
        gates += mid_mcx
    gates += ladder_ops[::-1]

    if work_wire_type == "dirty":
        # perform toggle-detection if ancilla is dirty
        gates += ladder_ops[1:]
        if len(final_ctrls) == 1:
            gates.append(qml.Toffoli(wires=[work_wires[0], final_ctrls[0], target_wire]))
        else:
            gates += mid_mcx
        gates += ladder_ops[1:][::-1]

    return gates
