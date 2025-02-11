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
"""Transform for merging adjacent rotations of the same type in a quantum circuit."""
# pylint: disable=too-many-branches

from functools import lru_cache, partial

import pennylane as qml
from pennylane.ops.op_math import Adjoint
from pennylane.ops.qubit.attributes import composable_rotations
from pennylane.queuing import QueuingManager
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.transforms import transform
from pennylane.transforms.decompose import decompose_plxpr_to_plxpr
from pennylane.typing import PostprocessingFn

from .optimization_utils import find_next_gate, fuse_rot_angles


@lru_cache
def _get_plxpr_merge_rotations():
    try:
        # pylint: disable=import-outside-toplevel
        from jax import make_jaxpr
        from jax.core import Jaxpr

        from pennylane.capture import PlxprInterpreter
        from pennylane.operation import Operator
    except ImportError:
        return None, None

    # pylint: disable=redefined-outer-name, too-few-public-methods
    class MergeRotationsInterpreter(PlxprInterpreter):
        """Plxpr Interpreter for applying the ``merge_rotations``
        transform when program capture is enabled."""

        def __init__(self, atol=1e-8, include_gates=None):
            super().__init__()
            self.atol = atol
            self.include_gates = include_gates

            # dict[wire, operator]
            self.seen_operators = {}

        def interpret_operation(self, op: Operator):
            """Interpret a PennyLane operation instance.

            Args:
                op (Operator): a pennylane operator instance

            Returns:
                Any

            This method is only called when the operator's output is a dropped variable,
            so the output will not affect later equations in the circuit.

            See also: :meth:`~.interpret_operation_eqn`.

            """
            last_seen_op_on_same_wire = self.seen_operators.get(op.wires[0], None)
            if last_seen_op_on_same_wire is None:
                for w in op.wires:
                    self.seen_operators[w] = op
                return

            with qml.capture.pause():
                exact_wires_match = op.wires == last_seen_op_on_same_wire.wires
                are_same_type = isinstance(op, type(last_seen_op_on_same_wire))

                if exact_wires_match:
                    if are_same_type:
                        if isinstance(op, qml.Rot):
                            cumulative_angles = fuse_rot_angles(
                                qml.math.stack(op.parameters),
                                qml.math.stack(last_seen_op_on_same_wire.parameters),
                            )
                        else:
                            cumulative_angles = qml.math.stack(op.parameters) + qml.math.stack(
                                last_seen_op_on_same_wire.parameters
                            )

                        # delete last seen operators
                        for w in op.wires:
                            self.seen_operators.pop(w)

                        # replace with the newly merged one
                        for w in op.wires:
                            self.seen_operators[w] = op.__class__(
                                *cumulative_angles, wires=op.wires
                            )
                        return
            previous_ops_on_wires = set(self.seen_operators.get(w) for w in op.wires)
            for o in previous_ops_on_wires:
                if o is not None:
                    for w in o.wires:
                        self.seen_operators.pop(w)
            for w in op.wires:
                self.seen_operators[w] = op

            if len(previous_ops_on_wires) != 0:
                return [
                    super(MergeRotationsInterpreter, self).interpret_operation(o)
                    for o in previous_ops_on_wires
                ]

            if self.include_gates is not None and op.name not in self.include_gates:
                return super().interpret_operation(op)

            if op not in composable_rotations:
                return super().interpret_operation(op)

        def purge_seen_operators(self):
            ops_remaining = set(self.seen_operators.values())
            for op in ops_remaining:
                super().interpret_operation(op)

            all_wires = tuple(self.seen_operators.keys())
            for w in all_wires:
                self.seen_operators.pop(w)

        def eval(self, jaxpr: Jaxpr, consts: list, *args) -> list:
            """Evaluate a jaxpr.

            Args:
                jaxpr (jax.core.Jaxpr): the jaxpr to evaluate
                consts (list[TensorLike]): the constant variables for the jaxpr
                *args (tuple[TensorLike]): The arguments for the jaxpr.

            Returns:
                list[TensorLike]: the results of the execution.

            """
            self._env = {}
            self.setup()

            for arg, invar in zip(args, jaxpr.invars, strict=True):
                self._env[invar] = arg
            for const, constvar in zip(consts, jaxpr.constvars, strict=True):
                self._env[constvar] = const

            for eqn in jaxpr.eqns:
                primitive = eqn.primitive
                custom_handler = self._primitive_registrations.get(primitive, None)

                if custom_handler:
                    self.purge_seen_operators()
                    invals = [self.read(invar) for invar in eqn.invars]
                    outvals = custom_handler(self, *invals, **eqn.params)
                elif getattr(primitive, "prim_type", "") == "operator":
                    outvals = self.interpret_operation_eqn(eqn)
                elif getattr(primitive, "prim_type", "") == "measurement":
                    self.purge_seen_operators()
                    outvals = self.interpret_measurement_eqn(eqn)
                else:
                    invals = [self.read(invar) for invar in eqn.invars]
                    subfuns, params = primitive.get_bind_params(eqn.params)
                    outvals = primitive.bind(*subfuns, *invals, **params)

                if not primitive.multiple_results:
                    outvals = [outvals]
                for outvar, outval in zip(eqn.outvars, outvals, strict=True):
                    self._env[outvar] = outval

            self.purge_seen_operators()

            # Read the final result of the Jaxpr from the environment
            outvals = []
            for var in jaxpr.outvars:
                outval = self.read(var)
                if isinstance(outval, qml.operation.Operator):
                    outvals.append(super().interpret_operation(outval))
                else:
                    outvals.append(outval)
            self.cleanup()
            self._env = {}
            return outvals

    # pylint: disable=redefined-outer-name
    def merge_rotations_plxpr_to_plxpr(
        jaxpr, consts, targs, tkwargs, *args
    ):  # pylint: disable=unused-argument
        merge_rotations = MergeRotationsInterpreter(
            atol=tkwargs.pop("atol", 1e-8), include_gates=tkwargs.pop("include_gates", None)
        )

        # Flattens any adjoints in the JAXPR
        expanded_jaxpr = decompose_plxpr_to_plxpr(jaxpr, consts, targs, tkwargs, *args)

        def wrapper(*inner_args):
            return merge_rotations.eval(expanded_jaxpr.jaxpr, expanded_jaxpr.consts, *inner_args)

        return make_jaxpr(wrapper)(*args)

    return MergeRotationsInterpreter, merge_rotations_plxpr_to_plxpr


MergeRotationsInterpreter, merge_rotations_plxpr_to_plxpr = _get_plxpr_merge_rotations()


@partial(transform, plxpr_transform=merge_rotations_plxpr_to_plxpr)
def merge_rotations(
    tape: QuantumScript, atol=1e-8, include_gates=None
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    r"""Quantum transform to combine rotation gates of the same type that act sequentially.

    If the combination of two rotation produces an angle that is close to 0,
    neither gate will be applied.

    Args:
        tape (QNode or QuantumTape or Callable): A quantum circuit.
        atol (float): After fusion of gates, if the fused angle :math:`\theta` is such that
            :math:`|\theta|\leq \text{atol}`, no rotation gate will be applied.
        include_gates (None or list[str]): A list of specific operations to merge. If
            set to ``None`` (default), all operations in the
            `~.pennylane.ops.qubit.attributes.composable_rotations` attribute will be merged. Otherwise,
            only the operations whose names match those in the list will undergo merging.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], function]: The transformed circuit as described in :func:`qml.transform <pennylane.transform>`.

    **Example**

    >>> dev = qml.device('default.qubit', wires=3)

    You can apply the transform directly on :class:`QNode`

    .. code-block:: python

        @merge_rotations
        @qml.qnode(device=dev)
        def circuit(x, y, z):
            qml.RX(x, wires=0)
            qml.RX(y, wires=0)
            qml.CNOT(wires=[1, 2])
            qml.RY(y, wires=1)
            qml.Hadamard(wires=2)
            qml.CRZ(z, wires=[2, 0])
            qml.RY(-y, wires=1)
            return qml.expval(qml.Z(0))

    >>> circuit(0.1, 0.2, 0.3)
    0.9553364891256055

    .. details::
        :title: Details on merging ``Rot`` gates
        :href: details-on-rot

        When merging two :class:`~.pennylane.Rot` gates, there are a number of details to consider:

        First, the output angles are not always defined uniquely, because Euler angles are not
        unique for some rotations. ``merge_rotations`` makes a particular choice in
        this case.

        Second, ``merge_rotations`` is not differentiable everywhere when used on ``Rot``.
        It has singularities for specific rotation angles where the derivative will be NaN.

        Finally, this function can be numerically unstable near singular points.
        It is therefore recommended to use it with 64-bit floating point precision angles.

        For a mathematical derivation of the fusion of two ``Rot`` gates, see the documentation
        of :func:`~.pennylane.transforms.single_qubit_fusion`.

    .. details::
        :title: Usage Details

        You can also apply ``merge_rotations`` to a quantum function.

        .. code-block:: python

            def qfunc(x, y, z):
                qml.RX(x, wires=0)
                qml.RX(y, wires=0)
                qml.CNOT(wires=[1, 2])
                qml.RY(y, wires=1)
                qml.Hadamard(wires=2)
                qml.CRZ(z, wires=[2, 0])
                qml.RY(-y, wires=1)
                return qml.expval(qml.Z(0))

        The circuit before optimization:

        >>> qnode = qml.QNode(qfunc, dev)
        >>> print(qml.draw(qnode)(1, 2, 3))
        0: ──RX(1.00)──RX(2.00)─╭RZ(3.00)────────────┤  <Z>
        1: ─╭●─────────RY(2.00)─│──────────RY(-2.00)─┤
        2: ─╰X─────────H────────╰●───────────────────┤

        By inspection, we can combine the two ``RX`` rotations on the first qubit.
        On the second qubit, we have a cumulative angle of 0, and the gates will cancel.

        >>> optimized_qfunc = merge_rotations()(qfunc)
        >>> optimized_qnode = qml.QNode(optimized_qfunc, dev)
        >>> print(qml.draw(optimized_qnode)(1, 2, 3))
        0: ──RX(3.00)────╭RZ(3.00)─┤  <Z>
        1: ─╭●───────────│─────────┤
        2: ─╰X─────────H─╰●────────┤

        It is also possible to explicitly specify which rotations ``merge_rotations`` should
        merge using the ``include_gates`` argument. For example, if in the above
        circuit we wanted only to merge the "RX" gates, we could do so as follows:

        >>> optimized_qfunc = merge_rotations(include_gates=["RX"])(qfunc)
        >>> optimized_qnode = qml.QNode(optimized_qfunc, dev)
        >>> print(qml.draw(optimized_qnode)(1, 2, 3))
        0: ──RX(3.00)───────────╭RZ(3.00)────────────┤  <Z>
        1: ─╭●─────────RY(2.00)─│──────────RY(-2.00)─┤
        2: ─╰X─────────H────────╰●───────────────────┤

    """

    # Expand away adjoint ops
    def stop_at(obj):
        return not isinstance(obj, Adjoint)

    [expanded_tape], _ = qml.devices.preprocess.decompose(
        tape,
        stopping_condition=stop_at,
        name="merge_rotations",
        error=qml.operation.DecompositionUndefinedError,
    )
    list_copy = expanded_tape.operations
    new_operations = []
    while len(list_copy) > 0:
        current_gate = list_copy[0]

        # If a specific list of operations is specified, check and see if our
        # op is in it, then try to merge. If not, queue and move on.
        if include_gates is not None:
            if current_gate.name not in include_gates:
                new_operations.append(current_gate)
                list_copy.pop(0)
                continue

        # Check if the rotation is composable; if it is not, move on.
        if not current_gate in composable_rotations:
            new_operations.append(current_gate)
            list_copy.pop(0)
            continue

        # Find the next gate that acts on the same wires
        next_gate_idx = find_next_gate(current_gate.wires, list_copy[1:])

        # If no such gate is found (either there simply is none, or there are other gates
        # "in the way", queue the operation and move on
        if next_gate_idx is None:
            new_operations.append(current_gate)
            list_copy.pop(0)
            continue

        # We need to use stack to get this to work and be differentiable in all interfaces
        cumulative_angles = qml.math.stack(current_gate.parameters)
        interface = qml.math.get_interface(cumulative_angles)
        # As long as there is a valid next gate, check if we can merge the angles
        while next_gate_idx is not None:
            # Get the next gate
            next_gate = list_copy[next_gate_idx + 1]

            # If next gate is of the same type, we can merge the angles
            if isinstance(current_gate, type(next_gate)) and current_gate.wires == next_gate.wires:
                list_copy.pop(next_gate_idx + 1)
                next_params = qml.math.stack(next_gate.parameters, like=interface)
                # jax-jit does not support cast_like
                if not qml.math.is_abstract(cumulative_angles):
                    next_params = qml.math.cast_like(next_params, cumulative_angles)

                # The Rot gate must be treated separately
                if isinstance(current_gate, qml.Rot):
                    cumulative_angles = fuse_rot_angles(cumulative_angles, next_params)
                # Other, single-parameter rotation gates just have the angle summed
                else:
                    cumulative_angles = cumulative_angles + next_params
            # If it is not, we need to stop
            else:
                break

            # If we did merge, look now at the next gate
            next_gate_idx = find_next_gate(current_gate.wires, list_copy[1:])

        # If we are tracing/jitting or differentiating, don't perform any conditional checks and
        # apply the operation regardless of the angles. Otherwise, only apply if
        # the rotation angle is non-trivial.
        if (
            qml.math.is_abstract(cumulative_angles)
            or qml.math.requires_grad(cumulative_angles)
            or not qml.math.allclose(cumulative_angles, 0.0, atol=atol, rtol=0)
        ):
            with QueuingManager.stop_recording():
                new_operations.append(
                    current_gate.__class__(*cumulative_angles, wires=current_gate.wires)
                )

        # Remove the first gate from the working list
        list_copy.pop(0)

    new_tape = tape.copy(operations=new_operations)

    def null_postprocessing(results):
        """A postprocesing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        """
        return results[0]

    return [new_tape], null_postprocessing
