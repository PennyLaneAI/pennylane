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

import numpy as np
import numpy.linalg as npl
import pennylane as qml
import typing
from pennylane.operation import Operator
from pennylane.wires import Wires
from pennylane import math

def _convert_to_su2(U, return_global_phase=False):
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
    return (U_SU2, exp_angles) if return_global_phase else U_SU2

def _ensure_real_diagonal_q(q: np.ndarray):
    """
    Change the phases of Q so the main diagonal is real, and return the modified Q.
    """
    exp_angles = np.angle(np.diag(q))
    return q * np.exp(-1j * exp_angles).reshape((2, 1))

def _matrix_adjoint(matrix: np.ndarray):
    return np.transpose(np.conj(matrix))

def _param_su2(ar: float, ai: float, br: float, bi: float):
    """
    Create a matrix in the SU(2) form from complex parameters a, b.
    The resulting matrix is not guaranteed to be in SU(2), unless |a|^2 + |b|^2 = 1.
    """
    return np.array([[complex(ar, ai), complex(-br, bi)],
                     [complex(br, bi), complex(ar, -ai)]])

def _bisect_compute_a(u: np.ndarray):
    """
    Given the U matrix, compute the A matrix such that
    At x A x At x A x = U
    where At is the adjoint of A
    and x is the Pauli X matrix.
    """
    x = np.real(u[0,1])
    z = u[1,1]
    zr = np.real(z)
    zi = np.imag(z)
    ar = np.sqrt((np.sqrt((zr+1)/2)+1)/2)
    mul = 1/(2*np.sqrt((zr+1)*(np.sqrt((zr+1)/2)+1)))
    ai = zi*mul
    br = x*mul
    bi = 0
    return _param_su2(ar,ai,br,bi)

def _bisect_compute_b(u: np.ndarray):
    """
    Given the U matrix, compute the B matrix such that
    H Bt x B x H
    where Bt is the adjoint of B,
    H is the Hadamard matrix,
    and x is the Pauli X matrix.
    """
    x = u[0,1]
    zr = np.real(x)
    zi = np.imag(x)
    x = np.real(u[1,1])
    ar = np.sqrt((zr+1)/2)
    mul = 1/np.sqrt(2*(zr+1))
    ai = zi*mul
    br = x*mul
    bi = 0
    return _param_su2(ar,ai,br,bi)

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

def ctrl_decomp_bisect_od(target_operation: typing.Union[Operator, tuple[np.ndarray, Wires]], control_wires: Wires, later: bool = False):
    """Decompose the controlled version of a target single-qubit operation

    This function decomposes a controlled single-qubit target operation using the
    decomposition defined in section 3.1 of
    `Vale et al. (2023) <https://arxiv.org/abs/2302.06377>`_.

    The target operation's matrix must have a real off-diagonal for this specialized method to work.

    .. warning:: This method will add a global phase for target operations that do not
        belong to the SU(2) group.

    Args:
        target_operation (~.operation.Operator): the target operation to decompose
        control_wires (~.wires.Wires): the control wires of the operation

    Returns:
        list[Operation]: the decomposed operations

    Raises:
        ValueError: if ``target_operation`` is not a single-qubit operation
            or its matrix does not have a real off-diagonal

    """
    orig_target_operation = target_operation
    if isinstance(target_operation, Operator):
        target_operation = target_operation.matrix(), target_operation.wires

    if len(target_operation[1]) != 1:
        raise ValueError(
            "The target operation must be a single-qubit operation, instead "
            f"got {orig_target_operation.__class__.__name__}."
        )

    target_wire = target_operation[1]

    u = target_operation[0]
    u = _convert_to_su2(u)
    u = np.array(u)

    ui = np.imag(u)
    if not np.isclose(ui[1,0], 0) or not np.isclose(ui[0,1], 0):
        raise ValueError(f"Target operation's matrix must have real off-diagonal, but it is {u}")
    
    a = _bisect_compute_a(u)

    mid = len(control_wires) // 2
    lk = control_wires[:mid]
    rk = control_wires[mid:]

    def mcx(_lk, _rk):
        return qml.MultiControlledX(control_wires = _lk, wires = target_wire, work_wires = _rk)
    op_mcx1 = lambda:mcx(lk,rk)
    op_mcx2 = lambda:mcx(rk,lk)
    op_a = lambda:qml.QubitUnitary(a, target_wire)
    op_at = lambda:qml.adjoint(op_a())

    result = [op_mcx1, op_a, op_mcx2, op_at, op_mcx1, op_a, op_mcx2, op_at]
    if not later:
        result = [func() for func in result]
    return result


def ctrl_decomp_bisect_md(target_operation: typing.Union[Operator, tuple[np.ndarray, Wires]], control_wires: Wires, later: bool = False):
    """Decompose the controlled version of a target single-qubit operation

    This function decomposes a controlled single-qubit target operation using the
    decomposition defined in section 3.1 of
    `Vale et al. (2023) <https://arxiv.org/abs/2302.06377>`_.

    The target operation's matrix must have a real main-diagonal for this specialized method to work.

    .. warning:: This method will add a global phase for target operations that do not
        belong to the SU(2) group.

    Args:
        target_operation (~.operation.Operator): the target operation to decompose
        control_wires (~.wires.Wires): the control wires of the operation

    Returns:
        list[Operation]: the decomposed operations

    Raises:
        ValueError: if ``target_operation`` is not a single-qubit operation
            or its matrix does not have a real main-diagonal

    """
    orig_target_operation = target_operation
    if isinstance(target_operation, Operator):
        target_operation = target_operation.matrix(), target_operation.wires

    if len(target_operation[1]) != 1:
        raise ValueError(
            "The target operation must be a single-qubit operation, instead "
            f"got {orig_target_operation.__class__.__name__}."
        )

    target_wire = target_operation[1]

    u = target_operation[0]
    u = _convert_to_su2(u)
    u = np.array(u)

    ui = np.imag(u)
    if not np.isclose(ui[0,0], 0) or not np.isclose(ui[1,1], 0):
        raise ValueError(f"Target operation's matrix must have real main-diagonal, but it is {u}")
    
    sh = qml.Hadamard.compute_matrix()
    mod_u = sh @ u @ sh
    mod_op = mod_u, target_wire

    op_h = lambda:qml.Hadamard(target_wire)
    inner = ctrl_decomp_bisect_od(mod_op, control_wires, later=True)
    result = [op_h] + inner + [op_h]
    if not later:
        result = [func() for func in result]
    return result


def ctrl_decomp_bisect_general(target_operation: typing.Union[Operator, tuple[np.ndarray, Wires]], control_wires: Wires, later: bool = False):
    """Decompose the controlled version of a target single-qubit operation

    This function decomposes a controlled single-qubit target operation using the
    decomposition defined in section 3.2 of
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

    """
    orig_target_operation = target_operation
    if isinstance(target_operation, Operator):
        target_operation = target_operation.matrix(), target_operation.wires

    if len(target_operation[1]) != 1:
        raise ValueError(
            "The target operation must be a single-qubit operation, instead "
            f"got {orig_target_operation.__class__.__name__}."
        )

    target_wire = target_operation[1]

    u = target_operation[0]
    u = _convert_to_su2(u)
    u = np.array(u)

    sx = qml.PauliX.compute_matrix()
    sh = qml.Hadamard.compute_matrix()
    
    d, q = npl.eig(u)
    d = np.diag(d)
    q = _ensure_real_diagonal_q(q)
    b = _bisect_compute_b(q)
    c1 = b @ sx @ sh
    c2t = b @ sh

    mid = len(control_wires) // 2
    lk = control_wires[:mid]
    rk = control_wires[mid:]

    def mcx(_lk, _rk):
        return qml.MultiControlledX(control_wires = _lk, wires = target_wire, work_wires = _rk)
    op_mcx1 = lambda:mcx(lk,rk)
    op_mcx2 = lambda:mcx(rk,lk)
    op_c1 = lambda:qml.QubitUnitary(c1, target_wire)
    op_c1t = lambda:qml.adjoint(op_c1())
    op_c2t = lambda:qml.QubitUnitary(c2t, target_wire)
    op_c2 = lambda:qml.adjoint(op_c2t())

    inner = ctrl_decomp_bisect_od((d, target_wire), control_wires, later=True)
    result = [op_c2t, op_mcx2, op_c1t, op_mcx1] + inner + [op_mcx1, op_c1, op_mcx2, op_c2]
    # cancel out adjacent op_mcx1
    result = result[:3] + result[5:]
    if not later:
        result = [func() for func in result]
    return result



