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
# pylint: disable=too-many-branches

from functools import lru_cache
from typing import Optional

import pennylane as qml
from pennylane.ops.qubit import Rot
from pennylane.queuing import QueuingManager
from pennylane.tape import QuantumScript, QuantumScriptBatch
from pennylane.transforms import transform
from pennylane.typing import PostprocessingFn

from .optimization_utils import find_next_gate, fuse_rot_angles


@lru_cache
def _get_plxpr_single_qubit_fusion():  # pylint: disable=missing-function-docstring,too-many-statements
    try:
        # pylint: disable=import-outside-toplevel
        from jax import make_jaxpr

        from pennylane.capture import PlxprInterpreter
        from pennylane.operation import Operator
    except ImportError:  # pragma: no cover
        return None, None

    # pylint: disable=redefined-outer-name

    class SingleQubitFusionInterpreter(PlxprInterpreter):
        """Plxpr Interpreter for applying the ``cancel_inverses`` transform to callables or jaxpr
        when program capture is enabled.

        .. note::

            In the process of transforming plxpr, this interpreter may reorder operations that do
            not share any wires. This will not impact the correctness of the circuit.
        """

        def __init__(self, exclude_gates: Optional[list[str]] = None):
            """Initialize the interpreter."""
            self.exclude_gates = exclude_gates
            self.previous_ops = {}
            super().__init__()

        def setup(self) -> None:
            """Initialize the instance before interpreting equations."""
            self.previous_ops = {}

        def cleanup(self) -> None:
            """Clean up the instance after interpreting equations."""
            self.previous_ops = {}

        def interpret_operation(self, op: Operator):
            """Interpret a PennyLane operation instance."""

            # For operators like Identity()
            if len(op.wires) == 0:
                return super().interpret_operation(op)

            # If the gate should be excluded, we interpret it as-is
            # regardless of fusion potential
            if self.exclude_gates is not None:
                if op.name in self.exclude_gates:
                    return super().interpret_operation(op)

            try:
                # Check if the operation has the single_qubit_rot_angles method
                cumulative_angles = qml.math.stack(op.single_qubit_rot_angles())
            except (NotImplementedError, AttributeError):

                # The order might not be deterministic if wires (the keys) are abstract.
                # However, this only impacts operators without any shared wires,
                # which does not affect the correctness of the result.
                previous_ops_on_wires = list(
                    dict.fromkeys(
                        self.previous_ops.get(w)
                        for w in op.wires
                        if self.previous_ops.get(w) is not None
                    )
                )

                # We convert to Rot and interpret the previous operation(s)
                # on the same wire(s) before interpreting the current operation.
                res = []
                for prev_op in previous_ops_on_wires:

                    o_angles = qml.math.stack(prev_op.single_qubit_rot_angles())
                    # pylint: disable=protected-access
                    o_rot = qml.Rot._primitive.impl(*o_angles, wires=prev_op.wires)
                    res.append(super().interpret_operation(o_rot))

                res.append(super().interpret_operation(op))

                # Finally, we remove the previous operations on the wires
                for w in op.wires:
                    if w in self.previous_ops:
                        self.previous_ops.pop(w)

                return res

            # Retrieve previous operation on the same wire
            prev_op = self.previous_ops.get(op.wires[0], None)
            if prev_op is None:
                for w in op.wires:
                    self.previous_ops[w] = op
                return []

            # Fuse previous and current rotation angles
            prev_op_angles = qml.math.stack(prev_op.single_qubit_rot_angles())
            cumulative_angles = fuse_rot_angles(prev_op_angles, cumulative_angles)

            # Store the new fused rotation
            # pylint: disable=protected-access
            new_rot = qml.Rot._primitive.impl(*cumulative_angles, wires=op.wires)
            for w in op.wires:
                self.previous_ops[w] = new_rot

            return []

        def interpret_all_previous_ops(self) -> None:
            """Interpret all previous operations stored in the instance."""

            # As above, the order might not be deterministic if wires (the keys) are abstract.
            # However, this only impacts operators without any shared wires,
            # which does not affect the correctness of the result.
            ops_remaining = list(dict.fromkeys(self.previous_ops.values()))

            for op in ops_remaining:
                super().interpret_operation(op)

            all_wires = tuple(self.previous_ops.keys())
            for w in all_wires:
                self.previous_ops.pop(w)

        def eval(self, jaxpr: "jax.core.Jaxpr", consts: list, *args) -> list:
            """Evaluate a jaxpr.

            Args:
                jaxpr (jax.core.Jaxpr): the jaxpr to evaluate
                consts (list[TensorLike]): the constant variables for the jaxpr
                *args (tuple[TensorLike]): The arguments for the jaxpr.

            Returns:
                list[TensorLike]: the results of the execution.

            """
            # pylint: disable=too-many-branches,attribute-defined-outside-init
            self._env = {}
            self.setup()

            for arg, invar in zip(args, jaxpr.invars, strict=True):
                self._env[invar] = arg
            for const, constvar in zip(consts, jaxpr.constvars, strict=True):
                self._env[constvar] = const

            for eqn in jaxpr.eqns:

                custom_handler = self._primitive_registrations.get(eqn.primitive, None)
                if custom_handler:
                    self.interpret_all_previous_ops()
                    invals = [self.read(invar) for invar in eqn.invars]
                    outvals = custom_handler(self, *invals, **eqn.params)
                elif getattr(eqn.primitive, "prim_type", "") == "operator":
                    outvals = self.interpret_operation_eqn(eqn)
                elif getattr(eqn.primitive, "prim_type", "") == "measurement":
                    self.interpret_all_previous_ops()
                    outvals = self.interpret_measurement_eqn(eqn)
                else:
                    if getattr(eqn.primitive, "prim_type", "") == "transform":
                        self.interpret_all_previous_ops()
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
            self._env = {}
            return outvals

    def single_qubit_fusion_plxpr_to_plxpr(
        jaxpr, consts, targs, tkwargs, *args
    ):  # pylint: disable=unused-argument
        interpreter = SingleQubitFusionInterpreter()

        def wrapper(*inner_args):
            return interpreter.eval(jaxpr, consts, *inner_args)

        return make_jaxpr(wrapper)(*args)

    return SingleQubitFusionInterpreter, single_qubit_fusion_plxpr_to_plxpr


SingleQubitFusionInterpreter, single_qubit_plxpr_to_plxpr = _get_plxpr_single_qubit_fusion()


@transform
def single_qubit_fusion(
    tape: QuantumScript, atol: Optional[float] = 1e-8, exclude_gates: Optional[list[str]] = None
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
        on wether program capture is enabled or not. This only impacts the order of
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
        0: ──H──Rot(0.1, 0.2, 0.3)──Rot(0.4, 0.5, 0.6)──RZ(0.1)──RZ(0.4)──┤ ⟨X⟩

        Full single-qubit gate fusion allows us to collapse this entire sequence into a
        single ``qml.Rot`` rotation gate.

        >>> optimized_qfunc = qml.transforms.single_qubit_fusion(qfunc)
        >>> optimized_qnode = qml.QNode(optimized_qfunc, dev)
        >>> print(qml.draw(optimized_qnode)([0.1, 0.2, 0.3], [0.4, 0.5, 0.6]))
        0: ──Rot(3.57, 2.09, 2.05)──┤ ⟨X⟩

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
            cumulative_angles = qml.math.stack(current_gate.single_qubit_rot_angles())
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
                next_gate_angles = qml.math.stack(next_gate.single_qubit_rot_angles())
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
            qml.math.is_abstract(cumulative_angles)
            or qml.math.requires_grad(cumulative_angles)
            or not qml.math.allclose(
                qml.math.stack([cumulative_angles[0] + cumulative_angles[2], cumulative_angles[1]]),
                0.0,
                atol=atol,
                rtol=0,
            )
        ):
            with QueuingManager.stop_recording():
                new_operations.append(Rot(*cumulative_angles, wires=current_gate.wires))

        # Remove the starting gate from the list
        list_copy.pop(0)

    new_tape = tape.copy(operations=new_operations)

    def null_postprocessing(results):
        """A postprocesing function returned by a transform that only converts the batch of results
        into a result for a single ``QuantumTape``.
        """
        return results[0]

    return [new_tape], null_postprocessing
