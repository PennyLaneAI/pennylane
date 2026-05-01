# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
A transform for decomposing arbitrary single-qubit QubitUnitary gates into elementary gates.
"""

from functools import lru_cache, partial

from pennylane import capture, math
from pennylane.operation import Operator
from pennylane.ops.op_math.decompositions import one_qubit_decomposition, two_qubit_decomposition
from pennylane.ops.qubit.matrix_ops import QubitUnitary
from pennylane.queuing import QueuingManager
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.transforms import transform
from pennylane.typing import PostprocessingFn


@lru_cache
def _get_plxpr_unitary_to_rot():
    try:
        # pylint: disable=import-outside-toplevel
        from pennyjax5 import make_jaxpr

        from pennylane.capture import PlxprInterpreter
    except ImportError:  # pragma: no cover
        return None, None

    # pylint: disable=redefined-outer-name, too-few-public-methods
    class UnitaryToRotInterpreter(PlxprInterpreter):
        """Plxpr Interpreter for applying the ``unitary_to_rot``
        transform when program capture is enabled."""

        def interpret_operation(self, op: Operator):
            """Decompose a PennyLane operation instance if it is a QubitUnitary.

            Args:
                op (Operator): a pennylane operator instance

            Returns:
                list: The decomposed operations.

            This method is only called when the operator's output is a dropped variable,
            so the output will not affect later equations in the circuit.

            See also: :meth:`~.interpret_operation_eqn`, :meth:`~.interpret_operation`.
            """
            if isinstance(op, QubitUnitary):
                ops = []
                with capture.pause():
                    matrix_shape = math.shape(op.parameters[0])
                    if matrix_shape == (2, 2):
                        ops = one_qubit_decomposition(op.parameters[0], op.wires[0])
                    elif matrix_shape == (4, 4):
                        ops = two_qubit_decomposition(op.parameters[0], op.wires)
                    else:
                        ops = [op]
                # List comprehensions are run in a separate scope.
                # The automatic insertion of __class__ and self for zero-argument super does not work in such a nested scope.
                # pylint: disable=super-with-arguments
                return [super(UnitaryToRotInterpreter, self).interpret_operation(o) for o in ops]

            return super().interpret_operation(op)

    def unitary_to_rot_plxpr_to_plxpr(jaxpr, consts, targs, tkwargs, *args):
        """Function for applying the ``unitary_to_rot`` transform on plxpr."""
        # Restore tkwargs from hashable tuple to dict
        tkwargs = dict(tkwargs)

        interpreter = UnitaryToRotInterpreter(*targs, **tkwargs)

        def wrapper(*inner_args):
            return interpreter.eval(jaxpr, consts, *inner_args)

        return make_jaxpr(wrapper)(*args)

    return UnitaryToRotInterpreter, unitary_to_rot_plxpr_to_plxpr


UnitaryToRotInterpreter, unitary_to_rot_plxpr_to_plxpr = _get_plxpr_unitary_to_rot()


@partial(transform, plxpr_transform=unitary_to_rot_plxpr_to_plxpr)
def unitary_to_rot(tape: QuantumScript) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    r"""Decompose all single-qubit and two-qubit :class:`~.QubitUnitary` operations to parametrized single-qubit operations and CNOTs.

    Single-qubit gates will be converted to a sequence of Y and Z rotations in the form
    :math:`RZ(\omega) RY(\theta) RZ(\phi)` that implements the original operation up
    to a global phase. Two-qubit gates will be decomposed according to the
    :func:`pennylane.transforms.two_qubit_decomposition` function.

    .. warning::

        This transform is not fully differentiable for 2-qubit ``QubitUnitary``
        operations. See usage details below.

    Args:
        tape (QNode or QuantumTape or Callable): A quantum circuit.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]: The transformed circuit as described in :func:`qp.transform <pennylane.transform>`.

    **Example**

    Suppose we would like to apply the following unitary operation:

    .. code-block:: python

        U = np.array([
            [-0.17111489+0.58564875j, -0.69352236-0.38309524j],
            [ 0.25053735+0.75164238j,  0.60700543-0.06171855j]
        ])

    The ``unitary_to_rot`` transform enables us to decompose such numerical
    operations while preserving differentiability.

    .. code-block:: python

        @qp.transforms.unitary_to_rot
        @qp.qnode(qp.device("default.qubit"))
        def circuit():
            qp.QubitUnitary(U, wires=0)
            return qp.expval(qp.Z(0))

    The original circuit is:

    >>> print(qp.draw(circuit, level=0)())
    0: ──U(M0)─┤  <Z>
    M0 =
    [[-0.171...+0.5856...j -0.693...-0.383...j]
    [ 0.250...+0.751...j  0.607...-0.061...j]]

    We can use the transform to decompose the gate:

    >>> print(qp.draw(circuit, level=1)())
    0: ──RZ(11.22)──RY(1.83)──RZ(11.96)─┤  <Z>


    .. details::
        :title: Usage Details

        This decomposition is not fully differentiable. We **can** differentiate
        with respect to input QNode parameters when they are not used to
        explicitly construct a :math:`4 \times 4` unitary matrix being
        decomposed. So for example, the following will work:

        .. code-block:: python

            import scipy
            import pennylane.numpy as pnp

            U = scipy.stats.unitary_group.rvs(4, random_state=12345)

            @qp.transforms.unitary_to_rot
            @qp.qnode(qp.device("default.qubit"))
            def circuit(angles):
                qp.QubitUnitary(U, wires=["a", "b"])
                qp.RX(angles[0], wires="a")
                qp.RY(angles[1], wires="b")
                qp.CNOT(wires=["b", "a"])
                return qp.expval(qp.Z("a"))

        >>> g = qp.grad(circuit)
        >>> params = pnp.array([0.2, 0.3], requires_grad=True)
        >>> g(params)
        array([ 0.342..., -0.077...])

        However, the following example will **not** be differentiable:

        .. code-block:: python

            @qp.transforms.unitary_to_rot
            @qp.qnode(qp.device("default.qubit"))
            def circuit(angles):
                z = angles[0]
                x = angles[1]

                Z_mat = pnp.array([[pnp.exp(-1j * z / 2), 0.0], [0.0, pnp.exp(1j * z / 2)]])

                c = pnp.cos(x / 2)
                s = pnp.sin(x / 2) * 1j
                X_mat = pnp.array([[c, -s], [-s, c]])

                U = pnp.kron(Z_mat, X_mat)

                qp.Hadamard(wires="a")

                # U depends on the input parameters
                qp.QubitUnitary(U, wires=["a", "b"])

                qp.CNOT(wires=["b", "a"])
                return qp.expval(qp.X("a"))

        >>> g = qp.grad(circuit)
        >>> params = pnp.array([0.2, 0.3], requires_grad=True)
        >>> g(params)
        array([nan, nan])
    """
    operations = []
    for op in tape.operations:
        if isinstance(op, QubitUnitary):
            with QueuingManager.stop_recording():
                decomposition = _recursively_decompose_qubit_unitary(op)
            operations.extend(decomposition)
        else:
            operations.append(op)

    new_tape = tape.copy(operations=operations)

    def null_postprocessing(results):
        """A postprocessing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        """
        return results[0]

    return [new_tape], null_postprocessing


def _recursively_decompose_qubit_unitary(op: QubitUnitary) -> list[Operator]:
    """Recursively decomposes single or two-qubit QubitUnitary operators.

    Args:
        op (QubitUnitary): The operator to decompose.

    Returns:
        list[Operator]: The decomposition that does not contain any single or two-qubit unitaries.

    """
    decomposition = [op]

    while True:
        keep_decomposing = False
        next_decomposition = []
        for op_ in decomposition:
            if not isinstance(op_, QubitUnitary):
                next_decomposition.append(op_)
                continue

            shape = math.shape(op_.parameters[0])
            if shape == (2, 2):
                # Single-qubit unitary operations
                next_decomposition.extend(one_qubit_decomposition(op_.parameters[0], op_.wires[0]))
            elif shape == (4, 4):
                # Two-qubit unitary operations
                next_decomposition.extend(two_qubit_decomposition(op_.parameters[0], op_.wires))
                # NOTE: Keep decomposing just in case we generated two single qubit unitaries
                keep_decomposing = True
            else:
                # NOTE: Ensure we add 3+ QubitUnitary operators back
                next_decomposition.append(op_)

        decomposition = next_decomposition
        if not keep_decomposing:
            break

    return decomposition
