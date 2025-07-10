# Copyright 2025 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Ross-Selinger (arXiv:1403.2975v3) implementation for approximate Pauli-Z rotation gate decomposition."""
import math

import pennylane as qml
from pennylane.ops.op_math.decompositions.grid_problems import GridIterator
from pennylane.ops.op_math.decompositions.norm_solver import _solve_diophantine
from pennylane.ops.op_math.decompositions.normal_forms import _ma_normal_form
from pennylane.ops.op_math.decompositions.rings import DyadicMatrix, SO3Matrix, ZOmega, ZSqrtTwo
from pennylane.queuing import QueuingManager


def _domain_correction(theta: float) -> tuple[float, ZOmega]:
    r"""Return shifts for the angle :math:`\theta` for it to be in the interval :math:`[-\pi/4, \pi/4]` and the corresponding scaling factor for the matrix elements.

    Args:
        theta (float): The angle to shift.

    Returns:
        tuple[float, ZOmega]: The domain shift and the scaling factor.
    """
    pi_vals = [(2 * i + 1) * math.pi / 4 for i in range(4)]  # pi/4, 3pi/4, 5pi/4, 7pi/4

    sign = 1
    theta %= 4 * math.pi
    if theta > 2 * math.pi:
        sign = -1
        theta -= math.pi * 4
    abs_theta = abs(theta)

    PI_8 = math.pi / 8
    div_ = round(abs_theta / PI_8, 12)  # Find and round the quotient
    ivl_ = int(div_)  # Integer part of the quotient
    mod_ = round(abs(ivl_ * PI_8 - abs_theta), 12)  # Find and round the remainder

    # Check if abs_theta is an odd multiple of PI_8
    if div_ == ivl_ and mod_ == 0.0 and ivl_ % 2 == 1.0:
        scale_vals = (
            [
                (ZOmega(d=1), ZOmega(d=1)),
                (ZOmega(b=-1), ZOmega(c=1)),
                (ZOmega(d=-1), ZOmega(b=1)),
                (ZOmega(b=1), ZOmega(a=1)),
            ]
            if sign == -1
            else [
                (ZOmega(b=1), ZOmega(b=1)),
                (ZOmega(d=-1), ZOmega(d=1)),
                (ZOmega(b=-1), ZOmega(b=-1)),
                (ZOmega(d=1), ZOmega(d=-1)),
            ]
        )
        return (sign, ivl_), scale_vals[ivl_ // 2]

    if pi_vals[0] <= abs_theta < pi_vals[1]:  # pi/4 <= |theta| < 3pi/4
        return -sign * math.pi / 2, ZOmega(b=sign)

    if pi_vals[1] <= abs_theta < pi_vals[2]:  # 3pi/4 <= |theta| < 5pi/4
        return -sign * math.pi, ZOmega(d=-1)

    if pi_vals[2] <= abs_theta < pi_vals[3]:  # 5pi/4 <= |theta| < 7pi/4
        return -sign * 3 * math.pi / 2, ZOmega(b=-sign)

    # TODO: Handle the |theta| == pi/4 case better.
    return 0.0, ZOmega(d=1)  # -pi/4 <= |theta| < pi/4 / 7pi/4 <= |theta| < 8pi/4


def rs_decomposition(op, epsilon, *, max_search_trials=20, max_factoring_trials=1000):
    r"""Approximate a phase shift rotation gate in the Clifford+T basis using the `Ross-Selinger algorithm <https://arxiv.org/abs/1403.2975>`_.

    This method implements the Ross-Selinger decomposition algorithm that approximates any arbitrary
    phase shift rotation gate with :math:`\epsilon > 0` error. The procedure exits when the approximation error
    becomes less than :math:`\epsilon`, or when ``max_search_trials`` attempts have been made for solution search.
    In the latter case, the approximation error could be :math:`\geq \epsilon`.

    This algorithm produces a decomposition with :math:`O(3\text{log}_2(1/\epsilon)) + O(\text{log}_2(\text{log}_2(1/\epsilon)))` operations.

    Args:
        op (~pennylane.RZ | ~pennylane.PhaseShift): A :class:`~.RZ` or :class:`~.PhaseShift` gate operation.
        epsilon (float): The maximum permissible error.

    Keyword Args:
        max_search_trials (int): The maximum number of attempts to find a solution
            while performing the grid search according to the Algorithm 7.6.1, in the
            `arXiv:1403.2975v3 <https://arxiv.org/abs/1403.2975>`_. Default is ``20``.
        max_factoring_trials (int): The maximum number of attempts to find a prime factor
            while performing the factoring to solve the Diophantine equation (Algorithm 7.6.2b)
            for the solution found in the grid search. Default is ``1000``.

    Returns:
        list[~pennylane.operation.Operation]: A list of gates in the Clifford+T basis set that approximates the given
        operation along with a final global phase operation. The operations are in the circuit-order.

    Raises:
        ValueError: If the given operator is not a :class:`~.RZ` or :class:`~.PhaseShift` gate.

    **Example**

    Suppose one would like to decompose :class:`~.RZ` with rotation angle :math:`\phi = \pi/3`:

    .. code-block:: python3

        import numpy as np
        import pennylane as qml

        op  = qml.RZ(np.pi/3, wires=0)
        ops = qml.ops.rs_decomposition(op, epsilon=1e-3)

        # Get the approximate matrix from the ops
        matrix_rs = qml.prod(*reversed(ops)).matrix()

    When the function is run for a sufficient ``max_search_trials``, the output gate sequence
    should implement the same operation approximately, up to an :math:`\epsilon`-error.

    >>> qml.math.allclose(op.matrix(), matrix_rs, atol=1e-3)
    True

    """

    with QueuingManager.stop_recording():

        # Check for length of wires in the operation
        if not isinstance(op, (qml.RZ, qml.PhaseShift)):
            raise ValueError(f"Operator must be a RZ or PhaseShift gate, got {op}")

        # Get the implemented angle with the domain correction and scaling factor for it.
        angle = -op.data[0] / 2
        shift, scale = _domain_correction(angle)
        interface = qml.math.get_interface(angle)
        phase = 0.0 if isinstance(op, qml.RZ) else angle

        u, t, k = ZOmega(d=1), ZOmega(), 0
        if not isinstance(scale, ZOmega):  # Get solution for the ± (2k + 1) . PI / 4 case.
            t_mat = DyadicMatrix(u, t, t, ZOmega(c=1)) @ DyadicMatrix(scale[0], t, t, u)
            dyd_mat = t_mat * scale[1]
            phase += math.pi / 8 if shift[0] == -1 else (8 - shift[1]) * math.pi / 8

        else:  # Get the solution from the grid search solver.
            u_solutions = GridIterator(angle + shift, epsilon, max_trials=max_search_trials)
            for u_sol, k_val in u_solutions:
                xi = ZSqrtTwo(2**k_val) - u_sol.norm().to_sqrt_two()
                if (t_sol := _solve_diophantine(xi, max_trials=max_factoring_trials)) is not None:
                    u, t, k = u_sol * scale, t_sol * scale, k_val
                    break

            dyd_mat = DyadicMatrix(u, -t.conj(), t, u.conj(), k=k)

        # Get the normal form of the decomposition.
        so3_mat = SO3Matrix(dyd_mat)
        decomposition, g_phase = _ma_normal_form(so3_mat)

        # Remove inverses if any in the decomposition and handle trivial case
        new_tape = qml.tape.QuantumScript(decomposition)

    # Map the wires to that of the operation and queue
    if queuing := QueuingManager.recording():
        QueuingManager.remove(op)

    if (op_wire := op.wires[0]) != 0:
        [new_tape], _ = qml.map_wires(new_tape, wire_map={0: op_wire}, queue=True)
    else:
        if queuing:
            _ = [qml.apply(op) for op in new_tape.operations]

    # TODO: Improve the global phase information to the decomposition.
    phase += qml.math.mod(g_phase, 2) * math.pi
    global_phase = qml.GlobalPhase(qml.math.array(phase, like=interface))

    # Return the gates from the mapped tape and global phase
    return new_tape.operations + [global_phase]
