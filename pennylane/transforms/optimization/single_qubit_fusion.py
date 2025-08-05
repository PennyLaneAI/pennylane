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
"""Transform for fusing sequences of single-qubit gates."""


from functools import lru_cache, partial

import pennylane as qml
from pennylane import math
from pennylane.ops.qubit import Rot
from pennylane.queuing import QueuingManager
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.transforms import transform
from pennylane.typing import PostprocessingFn, TensorLike

from .optimization_utils import find_next_gate, fuse_rot_angles


@lru_cache
def _get_plxpr_single_qubit_fusion():  # pylint: disable=too-many-statements
    try:
        # pylint: disable=import-outside-toplevel
        from jax import make_jaxpr

        from pennylane.capture import PlxprInterpreter
        from pennylane.capture.primitives import measure_prim
        from pennylane.operation import Operator

    except ImportError:  # pragma: no cover
        return None, None

    # pylint: disable=redefined-outer-name

    class SingleQubitFusionInterpreter(PlxprInterpreter):
        """Plxpr Interpreter for applying the ``single_qubit_fusion`` transform to callables or jaxpr
        when program capture is enabled.

        .. note::

            In the process of transforming plxpr, this interpreter may reorder operations that do
            not share any wires. This will not impact the correctness of the circuit.
        """

        def __init__(self, atol: float | None = 1e-8, exclude_gates: list[str] | None = None):
            """Initialize the interpreter."""
            self.atol = atol
            self.exclude_gates = set(exclude_gates) if exclude_gates is not None else set()
            self.previous_ops = {}
            self._env = {}

        def setup(self) -> None:
            """Initialize the instance before interpreting equations."""
            self.previous_ops = {}
            self._env.clear()

        def cleanup(self) -> None:
            """Clean up the instance after interpreting equations."""
            self.previous_ops.clear()
            self._env.clear()

        def _retrieve_prev_ops_same_wire(self, op: Operator):
            """Retrieve and remove all previous operations that act on the same wire(s) as the given operation."""

            # The order might not be deterministic if the wires (keys) are abstract.
            # However, this only impacts operators without any shared wires,
            # which does not affect the correctness of the result.

            # If the wires are concrete, the order of the keys (wires)
            # and thus the values should reflect the order in which they are iterated
            # because in Python 3.7+ dictionaries maintain insertion order.
            previous_ops_on_wires = {
                w: self.previous_ops.pop(w) for w in op.wires if w in self.previous_ops
            }

            return previous_ops_on_wires.values()

        def _handle_non_fusible_op(self, op: Operator) -> list:
            """Handle an operation that cannot be fused into a Rot gate."""

            previous_ops_on_wires = self._retrieve_prev_ops_same_wire(op)
            dyn_wires = {w for w in op.wires if math.is_abstract(w)}
            other_saved_wires = set(self.previous_ops.keys()) - dyn_wires

            if dyn_wires and other_saved_wires:
                # We cannot guarantee that ops on other static or dynamic wires won't have wire
                # overlap with the current op, so we need to interpret all of them.
                self.interpret_all_previous_ops()

            res = []
            for prev_op in previous_ops_on_wires:
                with qml.capture.pause():
                    rot = qml.Rot(*prev_op.single_qubit_rot_angles(), wires=prev_op.wires)
                res.append(super().interpret_operation(rot))

            res.append(super().interpret_operation(op))

            return res

        def _handle_fusible_op(self, op: Operator, cumulative_angles: TensorLike) -> list:
            """Handle an operation that can be potentially fused into a Rot gate."""

            # Only single-qubit gates are considered for fusion
            op_wire = op.wires[0]

            prev_op = self.previous_ops.get(op.wires[0], None)
            dyn_wires = {w for w in op.wires if math.is_abstract(w)}
            other_saved_wires = set(self.previous_ops.keys()) - dyn_wires

            if prev_op is None or (dyn_wires and other_saved_wires):
                # If there are dynamic wires, we need to make sure that there are no
                # other wires in `self.previous_ops`, otherwise we can't fuse. If
                # there are other wires but no other op on the same dynamic wire(s),
                # there isn't anything to fuse, so we just add the current op to
                # `self.previous_ops` and return.
                if dyn_wires and (prev_op is None or other_saved_wires):
                    self.interpret_all_previous_ops()
                for w in op.wires:
                    self.previous_ops[w] = op
                return []

            prev_op_angles = math.stack(prev_op.single_qubit_rot_angles())
            cumulative_angles = fuse_rot_angles(prev_op_angles, cumulative_angles)

            if (
                math.is_abstract(cumulative_angles)
                or math.requires_grad(cumulative_angles)
                or not math.allclose(
                    math.stack([cumulative_angles[0] + cumulative_angles[2], cumulative_angles[1]]),
                    0.0,
                    atol=self.atol,
                    rtol=0,
                )
            ):
                with qml.capture.pause():
                    new_rot = qml.Rot(*cumulative_angles, wires=op.wires)
                self.previous_ops[op_wire] = new_rot
            else:
                del self.previous_ops[op_wire]

            return []

        def interpret_operation(self, op: Operator):
            """Interpret a PennyLane operation instance."""

            # Operators like Identity() have no wires
            if len(op.wires) == 0:
                return super().interpret_operation(op)

            # We interpret directly if the gate is explicitly excluded,
            # after interpreting all previous operations on the same wires.
            if op.name in self.exclude_gates:

                previous_ops_on_wires = self._retrieve_prev_ops_same_wire(op)

                for prev_op in previous_ops_on_wires:
                    super().interpret_operation(prev_op)

                return super().interpret_operation(op)

            try:
                cumulative_angles = math.stack(op.single_qubit_rot_angles())
            except (NotImplementedError, AttributeError):
                return self._handle_non_fusible_op(op)

            return self._handle_fusible_op(op, cumulative_angles)

        def interpret_all_previous_ops(self) -> None:
            """Interpret all previous operations stored in the instance."""

            for op in self.previous_ops.values():
                super().interpret_operation(op)

            self.previous_ops.clear()

        def eval(self, jaxpr: "jax.extend.core.Jaxpr", consts: list, *args) -> list:
            """Evaluate a jaxpr.

            Args:
                jaxpr (jax.extend.core.Jaxpr): the jaxpr to evaluate
                consts (list[TensorLike]): the constant variables for the jaxpr
                *args (tuple[TensorLike]): The arguments for the jaxpr.

            Returns:
                list[TensorLike]: the results of the execution.
            """

            self.setup()

            for arg, invar in zip(args, jaxpr.invars, strict=True):
                self._env[invar] = arg
            for const, constvar in zip(consts, jaxpr.constvars, strict=True):
                self._env[constvar] = const

            for eqn in jaxpr.eqns:

                prim_type = getattr(eqn.primitive, "prim_type", "")

                custom_handler = self._primitive_registrations.get(eqn.primitive, None)
                if custom_handler:
                    self.interpret_all_previous_ops()
                    invals = [self.read(invar) for invar in eqn.invars]
                    outvals = custom_handler(self, *invals, **eqn.params)
                elif prim_type == "operator":
                    outvals = self.interpret_operation_eqn(eqn)
                elif prim_type == "measurement":
                    self.interpret_all_previous_ops()
                    outvals = self.interpret_measurement_eqn(eqn)
                else:
                    invals = [self.read(invar) for invar in eqn.invars]
                    subfuns, params = eqn.primitive.get_bind_params(eqn.params)
                    outvals = eqn.primitive.bind(*subfuns, *invals, **params)

                if not eqn.primitive.multiple_results:
                    outvals = [outvals]
                for outvar, outval in zip(eqn.outvars, outvals, strict=True):
                    self._env[outvar] = outval

            self.interpret_all_previous_ops()

            outvals = []
            for var in jaxpr.outvars:
                outval = self.read(var)
                if isinstance(outval, Operator):
                    outvals.append(super().interpret_operation(outval))
                else:
                    outvals.append(outval)

            self.cleanup()
            return outvals

    @SingleQubitFusionInterpreter.register_primitive(measure_prim)
    def _(_, *invals, **params):
        subfuns, params = measure_prim.get_bind_params(params)
        return measure_prim.bind(*subfuns, *invals, **params)

    def single_qubit_fusion_plxpr_to_plxpr(jaxpr, consts, targs, tkwargs, *args):
        """Function for applying the ``single_qubit_fusion`` transform on plxpr."""

        interpreter = SingleQubitFusionInterpreter(*targs, **tkwargs)

        def wrapper(*inner_args):
            return interpreter.eval(jaxpr, consts, *inner_args)

        return make_jaxpr(wrapper)(*args)

    return SingleQubitFusionInterpreter, single_qubit_fusion_plxpr_to_plxpr


SingleQubitFusionInterpreter, single_qubit_plxpr_to_plxpr = _get_plxpr_single_qubit_fusion()


@partial(transform, plxpr_transform=single_qubit_plxpr_to_plxpr)
def single_qubit_fusion(  # pylint: disable=too-many-branches
    tape: QuantumScript, atol: float | None = 1e-8, exclude_gates: list[str] | None = None
) -> tuple[QuantumScriptBatch, PostprocessingFn]:
    r"""Quantum function transform to fuse together groups of single-qubit
    operations into a general single-qubit unitary operation (:class:`~.Rot`).

    Fusion is performed only between gates that implement the property
    ``single_qubit_rot_angles``. Any sequence of two or more single-qubit gates
    (on the same qubit) with that property defined will be fused into one ``Rot``.

    Args:
        tape (QNode or QuantumTape or Callable): A quantum circuit.
        atol (float): An absolute tolerance for which to apply a rotation after
            fusion. After fusion of gates, if the fused angles :math:`\theta` are such that
            :math:`|\theta|\leq \text{atol}`, no rotation gate will be applied.
        exclude_gates (None or list[str]): A list of gates that should be excluded
            from full fusion. If set to ``None``, all single-qubit gates that can
            be fused will be fused.

    Returns:
        qnode (QNode) or quantum function (Callable) or tuple[List[QuantumTape], Callable]:
        The transformed circuit as described in :func:`qml.transform <pennylane.transform>`.

    **Example**

    >>> dev = qml.device('default.qubit', wires=1)

    You can apply the transform directly on :class:`QNode`:

    .. code-block:: python

        @qml.transforms.single_qubit_fusion
        @qml.qnode(device=dev)
        def qfunc(r1, r2):
            qml.Hadamard(wires=0)
            qml.Rot(*r1, wires=0)
            qml.Rot(*r2, wires=0)
            qml.RZ(r1[0], wires=0)
            qml.RZ(r2[0], wires=0)
            return qml.expval(qml.X(0))

    The single qubit gates are fused before execution.

    .. note::

        The fused angles between two sets of rotation angles are not always defined uniquely
        because Euler angles are not unique for some rotations. ``single_qubit_fusion``
        makes a particular choice in this case.

    .. note::

        The order of the gates resulting from the fusion may be different depending
        on whether program capture is enabled or not. This only impacts the order of
        operations that do not share any wires, so the correctness of the circuit is not affected.

    .. warning::

        This function is not differentiable everywhere. It has singularities for specific
        input rotation angles, where the derivative will be NaN.

    .. warning::

        This function is numerically unstable at its singular points. It is recommended to use
        it with 64-bit floating point precision.

    .. details::
        :title: Usage Details

        Consider the following quantum function.

        .. code-block:: python

            def qfunc(r1, r2):
                qml.Hadamard(wires=0)
                qml.Rot(*r1, wires=0)
                qml.Rot(*r2, wires=0)
                qml.RZ(r1[0], wires=0)
                qml.RZ(r2[0], wires=0)
                return qml.expval(qml.X(0))

        The circuit before optimization:

        >>> qnode = qml.QNode(qfunc, dev)
        >>> print(qml.draw(qnode)([0.1, 0.2, 0.3], [0.4, 0.5, 0.6]))
        0: ──H──Rot(0.10,0.20,0.30)──Rot(0.40,0.50,0.60)──RZ(0.10)──RZ(0.40)─┤  <X>

        Full single-qubit gate fusion allows us to collapse this entire sequence into a
        single ``qml.Rot`` rotation gate.

        >>> optimized_qfunc = qml.transforms.single_qubit_fusion(qfunc)
        >>> optimized_qnode = qml.QNode(optimized_qfunc, dev)
        >>> print(qml.draw(optimized_qnode)([0.1, 0.2, 0.3], [0.4, 0.5, 0.6]))
        0: ──Rot(3.57,2.09,2.05)──GlobalPhase(-1.57)─┤  <X>

    .. details::
        :title: Derivation
        :href: derivation

        The matrix for an individual rotation is given by

        .. math::

            R(\phi_j,\theta_j,\omega_j)
            &= \begin{bmatrix}
            e^{-i(\phi_j+\omega_j)/2}\cos(\theta_j/2) & -e^{i(\phi_j-\omega_j)/2}\sin(\theta_j/2)\\
            e^{-i(\phi_j-\omega_j)/2}\sin(\theta_j/2) & e^{i(\phi_j+\omega_j)/2}\cos(\theta_j/2)
            \end{bmatrix}\\
            &= \begin{bmatrix}
            e^{-i\alpha_j}c_j & -e^{i\beta_j}s_j \\
            e^{-i\beta_j}s_j & e^{i\alpha_j}c_j
            \end{bmatrix},

        where we introduced abbreviations :math:`\alpha_j,\beta_j=\frac{\phi_j\pm\omega_j}{2}`,
        :math:`c_j=\cos(\theta_j / 2)` and :math:`s_j=\sin(\theta_j / 2)` for notational brevity.
        The upper left entry of the matrix product
        :math:`R(\phi_2,\theta_2,\omega_2)R(\phi_1,\theta_1,\omega_1)` reads

        .. math::

            x = e^{-i(\alpha_2+\alpha_1)} c_2 c_1 - e^{i(\beta_2-\beta_1)} s_2 s_1

        and should equal :math:`e^{-i\alpha_f}c_f` for the fused rotation angles.
        This means that we can obtain :math:`\theta_f` from the magnitude of the matrix product
        entry above, choosing :math:`c_f=\cos(\theta_f / 2)` to be non-negative:

        .. math::

            c_f = |x| &=
            \left|
            e^{-i(\alpha_2+\alpha_1)} c_2 c_1
            -e^{i(\beta_2-\beta_1)} s_2 s_1
            \right| \\
            &= \sqrt{c_1^2 c_2^2 + s_1^2 s_2^2 - 2 c_1 c_2 s_1 s_2 \cos(\omega_1 + \phi_2)}.

        Now we again make a choice and pick :math:`\theta_f` to be non-negative:

        .. math::

            \theta_f = 2\arccos(|x|).

        We can also extract the angle combination :math:`\alpha_f` from :math:`x` via
        :math:`\operatorname{arg}(x)`, which can be readily computed with :math:`\arctan`:

        .. math::

            \alpha_f = -\arctan\left(
            \frac{-c_1c_2\sin(\alpha_1+\alpha_2)-s_1s_2\sin(\beta_2-\beta_1)}
            {c_1c_2\cos(\alpha_1+\alpha_2)-s_1s_2\cos(\beta_2-\beta_1)}
            \right).

        We can use the standard numerical function ``arctan2``, which
        computes :math:`\arctan(x_1/x_2)` from :math:`x_1` and :math:`x_2` while handling
        special points suitably, to obtain the argument of the underlying complex number
        :math:`x_2 + x_1 i`.

        Finally, to obtain :math:`\beta_f`, we need a second element of the matrix product from
        above. We compute the lower-left entry to be

        .. math::

            y = e^{-i(\beta_2+\alpha_1)} s_2 c_1 + e^{i(\alpha_2-\beta_1)} c_2 s_1,

        which should equal :math:`e^{-i \beta_f}s_f`. From this, we can compute

        .. math::

            \beta_f = -\arctan\left(
            \frac{-c_1s_2\sin(\alpha_1+\beta_2)+s_1c_2\sin(\alpha_2-\beta_1)}
            {c_1s_2\cos(\alpha_1+\beta_2)+s_1c_2\cos(\alpha_2-\beta_1)}
            \right).

        From this, we may extract

        .. math::

            \phi_f = \alpha_f + \beta_f\qquad
            \omega_f = \alpha_f - \beta_f

        and are done.

        **Special cases:**

        There are a number of special cases for which we can skip the computation above and
        can combine rotation angles directly.

        1. If :math:`\omega_1=\phi_2=0`, we can simply merge the ``RY`` rotation angles
           :math:`\theta_j` and obtain :math:`(\phi_1, \theta_1+\theta_2, \omega_2)`.

        2. If :math:`\theta_j=0`, we can merge the two ``RZ`` rotations of the same ``Rot``
           and obtain :math:`(\phi_1+\omega_1+\phi_2, \theta_2, \omega_2)` or
           :math:`(\phi_1, \theta_1, \omega_1+\phi_2+\omega_2)`. If both ``RY`` angles vanish
           we get :math:`(\phi_1+\omega_1+\phi_2+\omega_2, 0, 0)`.

        Note that this optimization is not performed for differentiable input parameters,
        in order to maintain differentiability.

        **Mathematical properties:**

        All functions above are well-defined on the domain we are using them on,
        if we handle :math:`\arctan` via standard numerical implementations such as
        ``np.arctan2``.
        Based on the choices we made in the derivation above, the fused angles will lie in
        the intervals

        .. math::

            \phi_f, \omega_f \in [-\pi, \pi],\quad \theta_f \in [0, \pi].

        Close to the boundaries of these intervals, ``single_qubit_fusion`` exhibits
        discontinuities, depending on the combination of input angles.
        These discontinuities also lead to singular (non-differentiable) points as discussed below.

        **Differentiability:**

        The function derived above is differentiable almost everywhere.
        In particular, there are two problematic scenarios at which the derivative is not defined.
        First, the square root is not differentiable at :math:`0`, making all input angles with
        :math:`|x|=0` singular. Second, :math:`\arccos` is not differentiable at :math:`1`, making
        all input angles with :math:`|x|=1` singular.

    """
    # Make a working copy of the list to traverse
    list_copy = tape.operations.copy()
    new_operations = []
    global_phase = 0
    while len(list_copy) > 0:
        current_gate = list_copy[0]

        # If the gate should be excluded, queue it and move on regardless
        # of fusion potential
        if exclude_gates is not None:
            if current_gate.name in exclude_gates:
                new_operations.append(current_gate)
                list_copy.pop(0)
                continue

        # Look for single_qubit_rot_angles; if not available, queue and move on.
        # If available, grab the angles and try to fuse.
        try:
            cumulative_angles = math.stack(current_gate.single_qubit_rot_angles())
            _, phase = math.convert_to_su2(current_gate.matrix(), return_global_phase=True)
            global_phase += phase
        except (NotImplementedError, AttributeError):
            new_operations.append(current_gate)
            list_copy.pop(0)
            continue

        # Find the next gate that acts on at least one of the same wires
        next_gate_idx = find_next_gate(current_gate.wires, list_copy[1:])

        if next_gate_idx is None:
            new_operations.append(current_gate)
            list_copy.pop(0)
            continue

        # Before entering the loop, we check to make sure the next gate is not in the
        # exclusion list. If it is, we should apply the original gate as-is, and not the
        # Rot version (example in test test_single_qubit_fusion_exclude_gates).
        if exclude_gates is not None:
            next_gate = list_copy[next_gate_idx + 1]
            if next_gate.name in exclude_gates:
                new_operations.append(current_gate)
                list_copy.pop(0)
                continue

        # Loop as long as a valid next gate exists
        while next_gate_idx is not None:
            next_gate = list_copy[next_gate_idx + 1]

            # Check first if the next gate is in the exclusion list
            if exclude_gates is not None:
                if next_gate.name in exclude_gates:
                    break

            # Try to extract the angles; since the Rot angles are implemented
            # solely for single-qubit gates, and we used find_next_gate to obtain
            # the gate in question, only valid single-qubit gates on the same
            # wire as the current gate will be fused.
            try:
                next_gate_angles = math.stack(next_gate.single_qubit_rot_angles())
                _, phase = math.convert_to_su2(next_gate.matrix(), return_global_phase=True)
                global_phase += phase
            except (NotImplementedError, AttributeError):
                break
            cumulative_angles = fuse_rot_angles(cumulative_angles, next_gate_angles)

            list_copy.pop(next_gate_idx + 1)
            next_gate_idx = find_next_gate(current_gate.wires, list_copy[1:])

        # If we are tracing/jitting or differentiating, don't perform any conditional checks and
        # apply the rotation regardless of the angles.
        # If not tracing or differentiating, check whether total rotation is trivial by checking
        # if the RY angle and the sum of the RZ angles are close to 0
        if (
            math.is_abstract(cumulative_angles)
            or math.requires_grad(cumulative_angles)
            or not math.allclose(
                math.stack([cumulative_angles[0] + cumulative_angles[2], cumulative_angles[1]]),
                0.0,
                atol=atol,
                rtol=0,
            )
        ):
            with QueuingManager.stop_recording():
                new_operations.append(Rot(*cumulative_angles, wires=current_gate.wires))

        # Remove the starting gate from the list
        list_copy.pop(0)

    if math.is_abstract(global_phase) or not math.allclose(global_phase, 0):
        new_operations.append(qml.GlobalPhase(-global_phase))
    new_tape = tape.copy(operations=new_operations)

    def null_postprocessing(results):
        """A postprocesing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        """
        return results[0]

    return [new_tape], null_postprocessing
