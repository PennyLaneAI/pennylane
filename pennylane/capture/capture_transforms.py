# Copyright 2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Trace implementations for program capture"""
from typing import Any, Optional

import jax
import jax.numpy as jnp
from jax.core import MainTrace, Primitive, ShapedArray, Trace, Tracer

import pennylane as qml

from .base_interpreter import PlxprInterpreter
from .primitives import adjoint_transform_prim, ctrl_transform_prim

_mp_return_types = tuple(repr(value) for value in qml.measurements.ObservableReturnTypes)


class TransformTrace(Trace):
    """Trace for processing primitives for PennyLane transforms.

    Args:
        main (jax.core.MainTrace): Active interpreter to which this ``Trace`` belongs.
        sublevel (int): Sublevel for this ``Trace``.
        transform_program (pennylane.transforms.core.TransformProgram): ``TransformProgram``
            containing the transforms to apply to the primitives being interpreted.
        state (Optional[dict]): Dictionary containing environment information for transforms
            to use.
    """

    def __init__(
        self,
        main: MainTrace,
        sublevel: int,
        transform_program: "qml.transforms.core.TransformProgram",
        state: Optional[dict] = None,
    ):
        super().__init__(main, sublevel)
        self._transform_program = transform_program
        self._state = state or {}

    def pure(self, val: Any):
        """Create a new ``TransformTracer``."""
        return TransformTracer(self, val, 0)

    lift = sublift = pure

    @property
    def state(self):
        """Dictionary containing environment information for transforms to use."""
        return self._state

    def process_primitive(self, primitive: Primitive, tracers: tuple[Tracer], params: dict):
        """Interpret a given primitive.

        Args:
            primitive (jax.core.Primitive): Primitive to interpret.
            tracers (Sequence[jax.core.Tracer]): Input tracers to the primitive.
            params (dict): Keyword arguments/metadata for the primitive.

        Returns:
            Any: The result of the interpretation.
        """
        idx = max(t.idx for t in tracers if isinstance(t, TransformTracer))
        is_qml_primitive = (
            primitive.__class__.__module__.split(".")[0] == "pennylane"
            or primitive.name.split("_")[0] in _mp_return_types
        )

        if idx >= len(self._transform_program) or not is_qml_primitive:
            # Either all transforms have been applied or the primitive is not an operator or measurement
            tracers = [t.val if isinstance(t, TransformTracer) else t for t in tracers]
            return primitive.bind(*tracers, **params)

        transform: "qml.transforms.core.TransformContainer" = self._transform_program[idx]
        bind_fn = transform.plxpr_transform
        targs, tkwargs = transform.args, transform.kwargs

        return bind_fn(primitive, tracers, params, targs, tkwargs, state=self.state)


class TransformTracer(Tracer):
    """Tracer for tracing PennyLane transforms.

    Args:
        trace (pennylane.capture.TransformTrace): TransformTrace that boxed up inputs into this
            ``TransformTracer``.
        val (Any): Input value to box inside this ``TransformTracer``.
        idx (int): Index into the transform program to track which transform to apply.
    """

    def __init__(self, trace: TransformTrace, val: Any, idx: int):
        super().__init__(trace=trace)
        self.val = val
        self.idx = idx

        # Eagerly setting the abstract eval in __init__ to avoid recursion errors later. If all
        # TransformTracers set abstract eval in __init__, then recursion will never be needed.
        self._aval = self.get_aval(val)

    @staticmethod
    def get_aval(val: Any) -> jax.core.AbstractValue:
        """Get abstract value."""
        # pylint: disable=protected-access
        if isinstance(val, Tracer):
            aval = val.aval
        elif isinstance(val, jax.core.AbstractValue):
            aval = val
        elif isinstance(val, qml.operation.Operator):
            aval = qml.capture.AbstractOperator()
        elif isinstance(val, qml.measurements.MeasurementProcess):
            if val.obs is not None:
                kwargs = {"n_wires": None}
            elif val.mv is not None:
                kwargs = {"n_wires": len(val.mv) if isinstance(val.mv, list) else 1}
            else:
                kwargs = {
                    "n_wires": len(val.wires),
                    "has_eigvals": val._eigvals is not None,
                }
            aval = qml.capture.AbstractMeasurement(val._abstract_eval, **kwargs)
        else:
            if isinstance(val, (list, tuple, int, float, complex, bool)):
                val = jnp.array(val)
            aval = ShapedArray(qml.math.shape(val), val.dtype)

        return aval

    def __repr__(self):
        return f"TransformTracer({self._trace}, val={self.val}, idx={self.idx})"

    @property
    def aval(self):
        """Abstract value of this ``TransformTracer``."""
        return self._aval

    def full_lower(self):  # pylint: disable=missing-function-docstring
        return self


class TransformInterpreter(PlxprInterpreter):
    r"""Interpreter for transforming PLxPR.

    This interpreter can be used to apply transforms to PLxPR natively without having to create
    ``QuantumTape``\ s as an intermediate step.

    Args:
        transform_program (pennylane.transforms.core.TransformProgram): Transform program containing
            all transforms to be applied to the input function.

    **Example**

    Let's say a user has defined a transform that converts all ``qml.RX`` gates into ``qml.RY``
    gates. First, a PLxPR compatible version of the transform needs to be implemented. This can
    be done as shown below:

    .. code-block:: python

        from functools import partial

        def _convert_rx_to_ry_plxpr_transform(primitive, tracers, params, targs, tkwargs, state):
            from pennylane.capture import TransformTracer

            # Step 1: Transform primitive
            primitive = qml.RY._primitive if primitive.name == "RX" else primitive
            # Step 2: Update tracers
            tracers = [
                TransformTracer(t._trace, t.val, t.idx + 1) if isinstance(t, TransformTracer) else t
                for t in tracers
            ]
            # Step 3: Return the result of the transformation
            return primitive.bind(*tracers, **params)

        @partial(qml.transforms.core.transform, plxpr_transform=_convert_rx_to_ry_plxpr_transform)
        def convert_rx_to_ry(tape):
            new_ops = [
                qml.RY(op.data[0], op.wires) if isinstance(op, qml.RX) else op for op in tape.operations
            ]
            new_tape = qml.tape.QuantumScript(
                new_ops, tape.measurements, shots=tape.shots, trainable_params=tape.trainable_params
            )
            return [new_tape], lambda results: results[0]

    We can now use this to transform user functions:

    .. code-block:: python

        def func(x, y):
            qml.RX(x, 0)
            qml.CNOT([0, 1])
            qml.RY(y, 1)
            return qml.expval(qml.Z(1))

    >>> print(qml.capture.make_plxpr(func)(1.2, 3.4))
    { lambda ; a:f32[] b:f32[]. let
        _:AbstractOperator() = RX[n_wires=1] a 0
        _:AbstractOperator() = CNOT[n_wires=2] 0 1
        _:AbstractOperator() = RY[n_wires=1] b 1
        c:AbstractOperator() = PauliZ[n_wires=1] 1
        d:AbstractMeasurement(n_wires=None) = expval_obs c
      in (d,) }

    .. code-block:: python

        program = qml.transforms.core.TransformProgram()
        program.add_transform(convert_rx_to_ry)
        transform_interpreter = TransformInterpreter(program)
        transformed_func = transform_interpreter(func)

    >>> print(qml.capture.make_plxpr(transformed_func)(1.2, 3.4))
    { lambda ; a:f32[] b:f32[]. let
        _:AbstractOperator() = RY[n_wires=1] a 0
        _:AbstractOperator() = CNOT[n_wires=2] 0 1
        _:AbstractOperator() = RY[n_wires=1] b 1
        c:AbstractOperator() = PauliZ[n_wires=1] 1
        d:AbstractMeasurement(n_wires=None) = expval_obs c
      in (d,) }

    """

    _trace: TransformTrace
    _transform_program: "qml.transforms.core.TransformProgram"
    _state: dict

    def __init__(self, transform_program: "qml.transforms.core.TransformProgram"):
        self._trace = None
        self._state = {}
        self._transform_program = transform_program

        super().__init__()

    def cleanup(self) -> None:
        """Perform any final steps after iterating through all equations.

        Blank by default, this method can clean up instance variables. Particularily,
        this method can be used to deallocate qubits and registers when converting to
        a Catalyst variant jaxpr.
        """
        self._trace = None
        self._state = {}

    def read_with_trace(self, var):
        """Extract the value corresponding to a variable and box it in a ``TransformTracer``
        if not already boxed."""
        if getattr(var, "_trace", None) is self._trace:
            return var

        return self._trace.full_raise(var)

    def interpret_operation_eqn(self, eqn: "jax.core.JaxprEqn"):
        """Interpret an equation corresponding to an operator.

        Args:
            eqn (jax.core.JaxprEqn): a jax equation for an operator.

        See also: :meth:`~.interpret_operation`.

        """
        invals = [self.read(invar) for invar in eqn.invars]

        # We only wrap inputs in tracers if interpreting PennyLane primitive,
        # and, thus, only transform those primitives.
        if isinstance(eqn.outvars[0], jax.core.DropVar):
            invals = [self.read_with_trace(inval) for inval in invals]
        return eqn.primitive.bind(*invals, **eqn.params)

    def interpret_measurement_eqn(self, eqn: "jax.core.JaxprEqn"):
        """Interpret an equation corresponding to a measurement process.

        Args:
            eqn (jax.core.JaxprEqn)

        See also :meth:`~.interpret_measurement`.

        """
        invals = [self.read(invar) for invar in eqn.invars]
        invals = [self.read_with_trace(inval) for inval in invals]
        return eqn.primitive.bind(*invals, **eqn.params)

    def eval(self, jaxpr: "jax.core.Jaxpr", consts: list, *args) -> list:
        """Evaluate a jaxpr.

        Args:
            jaxpr (jax.core.Jaxpr): the jaxpr to evaluate
            consts (list[TensorLike]): the constant variables for the jaxpr
            *args (tuple[TensorLike]): The arguments for the jaxpr.

        Returns:
            list[TensorLike]: the results of the execution.

        """
        with jax.core.new_main(
            TransformTrace, transform_program=self._transform_program, state=self._state
        ) as main:
            self._trace = main.with_cur_sublevel()
            tracers_out = super().eval(jaxpr, consts, *args)

        return [r.val if isinstance(r, TransformTracer) else r for r in tracers_out]


@TransformInterpreter.register_primitive(adjoint_transform_prim)
def handle_adjoint_transform(self, *invals, jaxpr, lazy, n_consts):
    """Interpret an adjoint transform primitive."""
    raise NotImplementedError


# pylint: disable=too-many-arguments
@TransformInterpreter.register_primitive(ctrl_transform_prim)
def handle_ctrl_transform(self, *invals, n_control, jaxpr, control_values, work_wires, n_consts):
    """Interpret a ctrl transform primitive."""
    raise NotImplementedError
