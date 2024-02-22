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
from typing import Tuple
import numpy as np
import numpy.linalg as npl
import pennylane as qml
from pennylane.operation import Operation, Operator
from pennylane.wires import Wires
from pennylane import math


def _convert_to_su2(U):
    r"""Convert a 2x2 unitary matrix to :math:`SU(2)`.

    Args:
        U (array[complex]): A matrix, presumed to be :math:`2 \times 2` and unitary.
        return_global_phase (bool): If `True`, the return will include
        the global phase. If `False`, only the :math:`SU(2)` representative
        is returned.

    Returns:
        array[complex]: A :math:`2 \times 2` matrix in :math:`SU(2)` that is
        equivalent to U up to a global phase. If ``return_global_phase=True``,
        a 2-element tuple is returned, with the first element being the
        :math:`SU(2)` equivalent and the second, the global phase.
    """
    # Compute the determinants
    dets = math.linalg.det(U)

    exp_angles = math.cast_like(math.angle(dets), 1j) / 2
    U_SU2 = math.cast_like(U, dets) * math.exp(-1j * exp_angles)
    return U_SU2


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


def ctrl_decomp_zyz(target_operation: Operator, control_wires: Wires):
    """Decompose the controlled version of a target single-qubit operation

    This function decomposes a controlled single-qubit target operation using the
    decomposition defined in section 5 of
    `Barenco et al. (1995) <https://arxiv.org/abs/quant-ph/9503016>`_.

    .. warning:: This method will add a global phase for target operations that do not
        belong to the SU(2) group.

    Args:
        target_operation (~.operation.Operator): the target operation to decompose
        control_wires (~.wires.Wires): the control wires of the operation

    Returns:
        list[Operation]: the decomposed operations

    Raises:
        ValueError: if ``target_operation`` is not a single-qubit operation

    **Example**

    We can create a controlled operation using `qml.ctrl`, or by creating the
    decomposed controlled version of using `qml.ctrl_decomp_zyz`.

    .. code-block:: python

        dev = qml.device("default.qubit", wires=3)

        @qml.qnode(dev)
        def expected_circuit(op):
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=1)
            qml.ctrl(op, [0,1])
            return qml.probs()

        @qml.qnode(dev)
        def decomp_circuit(op):
            qml.Hadamard(wires=0)
            qml.Hadamard(wires=1)
            qml.ops.ctrl_decomp_zyz(op, [0,1])
            return qml.probs()

    Measurements on both circuits will give us the same results:

    >>> op = qml.RX(0.123, wires=2)
    >>> expected_circuit(op)
    tensor([0.25      , 0.        , 0.25      , 0.        , 0.25      ,
        0.        , 0.24905563, 0.00094437], requires_grad=True)
    >>> decomp_circuit(op)
    tensor([0.25      , 0.        , 0.25      , 0.        , 0.25      ,
        0.        , 0.24905563, 0.00094437], requires_grad=True)

    """
    if len(target_operation.wires) != 1:
        raise ValueError(
            "The target operation must be a single-qubit operation, instead "
            f"got {target_operation.__class__.__name__}."
        )

    target_wire = target_operation.wires

    def get_single_qubit_rot_angles_via_matrix() -> Tuple[float, float, float]:
        """Returns a triplet of angles representing the single-qubit decomposition
        of the matrix of the target operation using ZYZ rotations.
        """
        with qml.QueuingManager.stop_recording():
            zyz_decomp = qml.ops.one_qubit_decomposition(
                qml.matrix(target_operation),
                wire=target_wire,
                rotations="ZYZ",
            )
        return tuple(gate.parameters[0] for gate in zyz_decomp)  # type: ignore

    if isinstance(target_operation, Operation):
        try:
            phi, theta, omega = target_operation.single_qubit_rot_angles()
        except NotImplementedError:
            phi, theta, omega = get_single_qubit_rot_angles_via_matrix()
    else:
        phi, theta, omega = get_single_qubit_rot_angles_via_matrix()

    decomp = []

    if not qml.math.allclose(0.0, phi, atol=1e-8, rtol=0):
        decomp.append(qml.RZ(phi, wires=target_wire))
    if not qml.math.allclose(0.0, theta / 2, atol=1e-8, rtol=0):
        decomp.extend(
            [
                qml.RY(theta / 2, wires=target_wire),
                qml.ctrl(qml.X(target_wire), control=control_wires),
                qml.RY(-theta / 2, wires=target_wire),
            ]
        )
    else:
        decomp.append(qml.ctrl(qml.X(target_wire), control=control_wires))
    if not qml.math.allclose(0.0, -(phi + omega) / 2, atol=1e-6, rtol=0):
        decomp.append(qml.RZ(-(phi + omega) / 2, wires=target_wire))
    decomp.append(qml.ctrl(qml.PauliX(wires=target_wire), control=control_wires))
    if not qml.math.allclose(0.0, (omega - phi) / 2, atol=1e-8, rtol=0):
        decomp.append(qml.RZ((omega - phi) / 2, wires=target_wire))

    return decomp


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

    target_matrix = _convert_to_su2(target_matrix)
    target_matrix_imag = np.imag(target_matrix)

    if np.isclose(target_matrix_imag[1, 0], 0) and np.isclose(target_matrix_imag[0, 1], 0):
        # Real off-diagonal specialized algorithm - 16n+O(1) CNOTs
        return _ctrl_decomp_bisect_od(target_matrix, target_wire, control_wires)
    if np.isclose(target_matrix_imag[0, 0], 0) and np.isclose(target_matrix_imag[1, 1], 0):
        # Real main-diagonal specialized algorithm - 16n+O(1) CNOTs
        return _ctrl_decomp_bisect_md(target_matrix, target_wire, control_wires)
    # General algorithm - 20n+O(1) CNOTs
    return _ctrl_decomp_bisect_general(target_matrix, target_wire, control_wires)


def decompose_mcx(control_wires, target_wire, work_wires):
    """Decomposes the multi-controlled PauliX gate"""

    num_work_wires_needed = len(control_wires) - 2

    if len(work_wires) >= num_work_wires_needed:
        return _decompose_mcx_with_many_workers(control_wires, target_wire, work_wires)

    return _decompose_mcx_with_one_worker(control_wires, target_wire, work_wires[0])


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


def _decompose_mcx_with_one_worker(control_wires, target_wire, work_wire):
    """Decomposes the multi-controlled PauliX gate using the approach in Lemma 7.3 of
    https://arxiv.org/abs/quant-ph/9503016, which requires a single work wire"""
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
