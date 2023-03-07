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
import typing
import numpy as np
import numpy.linalg as npl
import pennylane as qml
from pennylane.operation import Operator
from pennylane.wires import Wires
from pennylane import math
from pennylane.typing import TensorLike
from pennylane.transforms.decompositions.single_qubit_unitary import _convert_to_su2 as _convert_to_su2_batched


def _matrix_adjoint(matrix: np.ndarray):
    return math.transpose(math.conj(matrix))


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
    return _convert_to_su2_batched([U])[0]


def _convert_to_real_diagonal(q: TensorLike) -> TensorLike:
    """
    Change the phases of Q so the main diagonal is real, and return the modified Q.
    """
    exp_angles = math.angle(math.diag(q))
    return q * math.exp(-1j * exp_angles).reshape((1, 2))


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
    x = math.real(u[0, 1])
    z = u[1, 1]
    zr = math.real(z)
    zi = math.imag(z)
    ar = math.sqrt((math.sqrt((zr + 1) / 2) + 1) / 2)
    mul = 1 / (2 * math.sqrt((zr + 1) * (math.sqrt((zr + 1) / 2) + 1)))
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
    sqrt = math.sqrt
    Abs = math.abs
    w = math.real(u[0, 0])
    s = math.real(u[1, 0])
    t = math.imag(u[1, 0])
    if math.isclose(s, 0):
        b = 0
        if math.isclose(t, 0):
            c = sqrt(w)
            d = 0
        else:
            c = sqrt(2 - 2 * w) * (-w / 2 - 1 / 2) / t
            d = sqrt(2 - 2 * w) / 2
    elif math.isclose(t, 0):
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

    try:
        phi, theta, omega = target_operation.single_qubit_rot_angles()
    except NotImplementedError:
        with qml.QueuingManager.stop_recording():
            zyz_decomp = qml.transforms.zyz_decomposition(
                qml.matrix(target_operation), target_wire
            )[0]
        phi, theta, omega = zyz_decomp.single_qubit_rot_angles()

    decomp = []

    if not qml.math.isclose(phi, 0.0, atol=1e-8, rtol=0):
        decomp.append(qml.RZ(phi, wires=target_wire))
    if not qml.math.isclose(theta / 2, 0.0, atol=1e-8, rtol=0):
        decomp.extend(
            [
                qml.RY(theta / 2, wires=target_wire),
                qml.MultiControlledX(wires=control_wires + target_wire),
                qml.RY(-theta / 2, wires=target_wire),
            ]
        )
    else:
        decomp.append(qml.MultiControlledX(wires=control_wires + target_wire))
    if not qml.math.isclose(-(phi + omega) / 2, 0.0, atol=1e-6, rtol=0):
        decomp.append(qml.RZ(-(phi + omega) / 2, wires=target_wire))
    decomp.append(qml.MultiControlledX(wires=control_wires + target_wire))
    if not qml.math.isclose((omega - phi) / 2, 0.0, atol=1e-8, rtol=0):
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
    decomposition defined in section 3.1 of
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

    mid = (len(control_wires) + 1) // 2  # for odd n, make lk bigger
    lower_controls = control_wires[:mid]
    upper_controls = control_wires[mid:]

    def component():
            return [ qml.MultiControlledX(wires=lk+target_wire, work_wires=rk),
            qml.QubitUnitary(a, target_wire),
            qml.MultiControlledX(wires=rk+target_wire, work_wires=lk), 
            qml.QubitUnitary(_matrix_adjoint(a), target_wire)]

    return component() + component()


def _ctrl_decomp_bisect_md(
    u: np.ndarray,
    target_wire: Wires,
    control_wires: Wires,
):
    """Decompose the controlled version of a target single-qubit operation

    Not backpropagation compatible (as currently implemented). Use only with numpy.

    This function decomposes a controlled single-qubit target operation using the
    decomposition defined in section 3.1 of
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

    sh = qml.Hadamard.compute_matrix()
    mod_u = sh @ u @ sh

    decomposition = [qml.Hadamard(target_wire)]
    decomposition +=  _ctrl_decomp_bisect_od(mod_u, target_wire, control_wires)
    decomposition.append(qml.Hadamard(target_wire)

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
    x_matrix = qml.PauliX.compute_matrix()
    h_matrix = qml.Hadamard.compute_matrix()
    alternate_h_matrix = x_matrix @ h_matrix @ x_matrix

    d, q = npl.eig(u)
    d = np.diag(d)
    q = _convert_to_real_diagonal(q)
    b = _bisect_compute_b(q)
    c1 = b @ sh_alt
    c2t = b @ sh

    mid = (len(control_wires) + 1) // 2  # for odd n, make lk bigger
    lk = control_wires[:mid]
    rk = control_wires[mid:]

    component = [qml.QubitUnitary(c2t, target_wire),
                qml.MultiControlledX(wires=rk+target_wire, work_wires=lk),
                qml.adjoint(qml.QubitUnitary(c1, target_wire)),
                qml.MultiControlledX(wires=lk+target_wire, work_wires=rk)]

    od_decomp = _ctrl_decomp_bisect_od(d, target_wire, control_wires)

    # cancel two identical multicontrolled x gates
    qml.QueuingManager.remove(component[3])
    qml.QueuingManager.remove(od_decomp[0])

    adjoint_component =  [qml.adjoint(copy(op), lazy=False) for op in reversed(component)]

    return component[0:3] + od_decomp[1:] + adjoint_compoent


def ctrl_decomp_bisect(
    target_operation: typing.Union[Operator, typing.Tuple[np.ndarray, Wires]],
    control_wires: Wires,
):
    """Decompose the controlled version of a target single-qubit operation

    Not backpropagation compatible (as currently implemented). Use only with numpy.

    Automatically selects the best algorithm based on the matrix (uses specialized more efficient
    algorithms if the matrix has a certain form, otherwise falls back to the general algorithm).

    .. warning:: This method will add a global phase for target operations that do not
        belong to the SU(2) group.

    Args:
        target_operation (~.operation.Operator): the target operation to decompose
        control_wires (~.wires.Wires): the control wires of the operation

    Returns:
        list[Operation]: the decomposed operations

    Raises:
        ValueError: if ``target_operation`` is not a single-qubit operation

    """
if len(target_operation.wires) > 1:
    raise ValueError(
        "The target operation must be a single-qubit operation, instead " f"got {target_operation}."
    )
target_matrix = target_operation.matrix()
target_wire = target_operation.wires

    u = _convert_to_su2(u)
    ui = np.imag(u)

    with qml.QueuingManager.stop_recording():
        if np.isclose(ui[1, 0], 0) and np.isclose(ui[0, 1], 0):
            # Real off-diagonal specialized algorithm - 16n+O(1) CNOTs
            result = _ctrl_decomp_bisect_od(u, target_wire, control_wires)
        elif np.isclose(ui[0, 0], 0) and np.isclose(ui[1, 1], 0):
            # Real main-diagonal specialized algorithm - 16n+O(1) CNOTs
            result = _ctrl_decomp_bisect_md(u, target_wire, control_wires)
        else:
            # General algorithm - 20n+O(1) CNOTs
            result = _ctrl_decomp_bisect_general(u, target_wire, control_wires)

    if qml.QueuingManager.recording():
        for op in result:
            qml.apply(op)

    return result
