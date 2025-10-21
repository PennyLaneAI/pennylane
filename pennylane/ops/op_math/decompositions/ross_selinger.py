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
from pennylane.compiler.compiler import AvailableCompilers
from pennylane.ops.op_math.decompositions.grid_problems import GridIterator
from pennylane.ops.op_math.decompositions.norm_solver import _solve_diophantine
from pennylane.ops.op_math.decompositions.normal_forms import (
    _clifford_keys_unwired,
    _ma_normal_form,
)
from pennylane.ops.op_math.decompositions.rings import DyadicMatrix, SO3Matrix, ZOmega, ZSqrtTwo
from pennylane.queuing import QueuingManager

is_jax = True
try:
    import jax
    import jax.numpy as jnp
    from jax.core import ShapedArray
except (ModuleNotFoundError, ImportError):  # pragma: no cover
    is_jax = False


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
        if ivl_ > 8:
            ivl_, sign = 16 - ivl_, -sign
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


def apply_clifford_from_idx(idx, wire):
    """Apply a Clifford gate sequence by index on the specified wire.

    This function maps an integer index to one of the standard Clifford sequences
    defined in `clifford_keys_unwired`. The returned function uses `qml.cond`
    to select and apply the correct sequence in QJIT-compatible form.

    Args:
        idx (int): Index into the Clifford sequence list.
        wire (int): Target wire.

    Returns:
        Callable: A conditional function that applies the indexed Clifford sequence.
    """
    keys = _clifford_keys_unwired()

    def make_fn(seq):
        def fn():
            for g in seq:
                g(wire)

        return fn

    # Build conditional cases: for each index, associate it with a function
    cases = [(idx == i, make_fn(seq)) for i, seq in enumerate(keys)]

    # Extract the first condition
    head_cond, head_fn = cases[0]

    # Remaining cases handled as "elif"
    elifs = cases[1:]

    # TODO: use qml.switch once available
    return qml.cond(head_cond, head_fn, elifs=elifs)


# pylint: disable=no-value-for-parameter
def _jit_rs_decomposition(wire, decomposition_info):
    """Apply the Ross-Selinger decomposition with QJIT to the given decomposition.

    Matsumoto-Amano normal form: (T|ε)(HT|SHT)*C
    - (T|ε): Optional leading T gate
    - (HT|SHT): Middle sequence of HT or SHT syllables
    - C: Right most Clifford operator

    Args:
        wire (int): The wire to apply the decomposition to.
        decomposition_info (tuple): The decomposition information.

    Returns:
        list[~pennylane.operation.Operation]: A Clifford+T gate implementing the instructions from `decomposition_info`.
    """
    ops = []
    has_leading_t, syllable_sequence, clifford_op_idx = decomposition_info

    # Optional leading T gate
    leading_t_cond = qml.cond(has_leading_t, qml.T)
    leading_t_cond(wire)
    ops.append(leading_t_cond.operation)

    # Middle sequence of HT or SHT syllables.
    if syllable_sequence.shape[0] > 0:

        @qml.for_loop(start=0, stop=syllable_sequence.shape[0])
        def syllable_sequence_loop(i):
            is_HT = syllable_sequence[i]

            def compose_HT():
                qml.H(wire)
                qml.T(wire)

            def compose_SHT():
                qml.S(wire)
                qml.H(wire)
                qml.T(wire)

            #  syllable_sequence can be 0, 1 or -1.
            qml.cond(
                is_HT != -1,
                true_fn=qml.cond(is_HT.astype(bool), true_fn=compose_SHT, false_fn=compose_HT),
            )()

        syllable_sequence_loop()
        ops.append(syllable_sequence_loop.operation)

    # Rightmost Clifford operator
    # TODO: Whenever subroutines is supported in QJIT, make this function as subroutines.
    rightmost_cliff_op_cond = apply_clifford_from_idx(clifford_op_idx, wire)
    rightmost_cliff_op_cond()
    ops.append(rightmost_cliff_op_cond.operation)

    return ops


def rs_decomposition(
    op, epsilon, is_qjit=False, *, max_search_trials=20, max_factoring_trials=1000
):
    r"""Approximate a phase shift rotation gate in the Clifford+T basis using the `Ross-Selinger algorithm <https://arxiv.org/abs/1403.2975>`_.

    This method implements the Ross-Selinger decomposition algorithm that approximates any arbitrary
    phase shift rotation gate with :math:`\epsilon > 0` error. The procedure exits when the approximation error
    becomes less than :math:`\epsilon`, or when ``max_search_trials`` attempts have been made for solution search.
    In the latter case, the approximation error could be :math:`\geq \epsilon`.

    This algorithm produces a decomposition with :math:`O(3\text{log}_2(1/\epsilon)) + O(\text{log}_2(\text{log}_2(1/\epsilon)))` operations.

    Args:
        op (~pennylane.RZ | ~pennylane.PhaseShift): A :class:`~.RZ` or :class:`~.PhaseShift` gate operation.
        epsilon (float): The maximum permissible error.
        is_qjit (bool): Whether the decomposition is being performed with QJIT enabled.

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

    .. code-block:: python

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

        angle = -op.data[0] / 2

        def eval_ross_algorithm(angle: float, upper_bounded_size: int = None):

            # Get the implemented angle with the domain correction and scaling factor for it.
            shift, scale = _domain_correction(angle)
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
                    if (
                        t_sol := _solve_diophantine(xi, max_trials=max_factoring_trials)
                    ) is not None:
                        u, t, k = u_sol * scale, t_sol * scale, k_val
                        break

                dyd_mat = DyadicMatrix(u, -t.conj(), t, u.conj(), k=k)

            # Get the normal form of the decomposition.
            so3_mat = SO3Matrix(dyd_mat)

            decomposition_info, g_phase = _ma_normal_form(
                so3_mat, compressed=is_qjit, upper_bounded_size=upper_bounded_size
            )

            return (decomposition_info, g_phase, phase)

        # If QJIT is active, use the compressed normal form.
        if not is_qjit:
            decomposed_gates, g_phase, phase = eval_ross_algorithm(angle)
        else:
            if not is_jax:
                raise ImportError(
                    "QJIT mode requires JAX. Please install it with `pip install jax jaxlib`."
                )  # pragma: no cover

            # circular import issue when import outside of the function
            api_extensions = AvailableCompilers.names_entrypoints["catalyst"]["ops"].load()

            # JAX arrays are static, so we need to specify the maximum size of the output array.
            # This is a heuristic to ensure the output array is large enough.
            # If the output array is smaller than the actual size,
            # the decomposition will be padded with -1s.
            upper_bounded_size = int(10 * math.log2(1 / epsilon))

            # Wrap the pure Python algorithm to cast outputs to JAX types
            def eval_ross_algorithm_jitted(angle_val, upper_bounded_size_val):
                decomp_info, gph, ph = eval_ross_algorithm(angle_val, upper_bounded_size_val)
                return (decomp_info, jnp.float64(gph), jnp.float64(ph))

            result_type = (
                (jnp.int32, ShapedArray((upper_bounded_size,), jnp.int32), jnp.int32),
                jnp.float64,  # g_phase
                jnp.float64,  # phase
            )

            decomposed_info, g_phase, phase = api_extensions.pure_callback(
                eval_ross_algorithm_jitted, result_type=result_type
            )(angle, upper_bounded_size)

            decomposed_gates = _jit_rs_decomposition(op.wires[0], decomposed_info)

        # Remove inverses if any in the decomposition and handle trivial case
        new_tape = qml.tape.QuantumScript(decomposed_gates)

    # Map the wires to that of the operation and queue
    if queuing := QueuingManager.recording():
        QueuingManager.remove(op)

    if not is_qjit and (op_wire := op.wires[0]) != 0:
        [new_tape], _ = qml.map_wires(new_tape, wire_map={0: op_wire}, queue=True)
    else:
        if queuing:
            _ = [qml.apply(op) for op in new_tape.operations]

    interface = qml.math.get_interface(angle)
    phase += qml.math.mod(g_phase, 2) * math.pi
    if is_qjit:
        if not is_jax:
            raise ImportError(
                "QJIT mode requires JAX. Please install it with `pip install jax jaxlib`."
            )  # pragma: no cover
        with jax.ensure_compile_time_eval():
            global_phase = qml.GlobalPhase(phase)
    else:
        global_phase = qml.GlobalPhase(qml.math.array(phase, like=interface))

    # Return the gates from the mapped tape and global phase
    return new_tape.operations + [global_phase]
