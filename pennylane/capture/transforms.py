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
from copy import copy
from typing import Any

import jax
import jax.numpy as jnp
from jax.core import MainTrace, Primitive, ShapedArray, Trace, Tracer

import pennylane as qml

from .base_interpreter import PlxprInterpreter, jaxpr_to_jaxpr
from .primitives import qnode_prim

# pylint: disable=missing-function-docstring


class TransformTrace(Trace):
    """Trace for processing primitives for PennyLane transforms"""

    def __init__(
        self,
        main: MainTrace,
        sublevel: int,
        transform_program: "qml.transforms.core.TransformProgram",
    ):
        super().__init__(main, sublevel)
        self._transform_program = transform_program
        self._state = None

    def pure(self, val: Any):
        return TransformTracer(self, val, 0)

    lift = sublift = pure

    @property
    def state(self):
        """Dictionary containing environment information for transforms to use."""
        return self._state

    @state.setter
    def state(self, state):
        self._state = state

    def process_primitive(self, primitive: Primitive, tracers: tuple[Tracer], params: dict):
        idx = max(t.idx for t in tracers if isinstance(t, TransformTracer))
        is_qml_primitive = primitive.__class__.__module__.split(".")[0] == "pennylane"

        if idx >= len(self._transform_program) or not is_qml_primitive:
            # Either all transforms have been applied or the primitive is not an operator or measurement
            tracers = [t.val if isinstance(t, TransformTracer) else t for t in tracers]
            return primitive.bind(*tracers, **params)

        transform: "qml.transforms.core.TransformContainer" = self._transform_program[idx]
        bind_fn = transform.plxpr_transform
        targs, tkwargs = transform.args, transform.kwargs

        return bind_fn(primitive, tracers, params, targs, tkwargs, state=self.state)


class TransformTracer(Tracer):
    """Tracer for tracing PennyLane transforms"""

    def __init__(self, trace: TransformTrace, val: Any, idx: int):
        super().__init__(trace=trace)
        self.val = val
        self.idx = idx

        # Eagerly setting the abstract eval in __init__ to avoid recursion errors later. If all
        # TransformTracers set abstract eval in __init__, then recursion will never be needed.
        if isinstance(val, Tracer):
            self._aval = val.aval
        elif isinstance(val, jax.core.AbstractValue):
            self._aval = val
        else:
            if isinstance(val, (list, tuple, int, float, complex, bool)):
                val = jnp.array(val)
            self._aval = ShapedArray(qml.math.shape(val), val.dtype)

    def __repr__(self):
        return f"TransformTracer({self._trace}, val={self.val}, idx={self.idx})"

    @property
    def aval(self):
        return self._aval

    def full_lower(self):  # pylint: disable=missing-function-docstring
        return self


class TransformInterpreter(PlxprInterpreter):
    """Interpreter for transforming PLxPR."""

    _trace: TransformTrace
    _transform_program: "qml.transforms.core.TransformProgram"

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
        """Extract the value corresponding to a variable."""
        if getattr(var, "_trace", None) is self._trace:
            return var

        return self._trace.pure(var)

    def interpret_operation_eqn(self, eqn: "jax.core.JaxprEqn"):
        """Interpret an equation corresponding to an operator.

        Args:
            eqn (jax.core.JaxprEqn): a jax equation for an operator.

        See also: :meth:`~.interpret_operation`.

        """
        invals = [self.read(invar) for invar in eqn.invars]

        # We only wrap inputs in tracers if interpreting PennyLane primitive,
        # and only transform primitives which are being applied in circuit
        if isinstance(eqn.outvars[0], jax.core.DropVar):
            invals = [self.read_with_trace(inval) for inval in invals]
            return eqn.primitive.bind(*invals, **eqn.params)

        # Other operators are created normally and saved to the environment
        # to be used for later
        op = eqn.primitive.impl(*invals, **eqn.params)

        return op

    def interpret_measure_eqn(self, eqn: "jax.core.JaxprEqn"):
        """Interpret an equation corresponding to a measurement process.

        Args:
            eqn (jax.core.JaxprEqn)

        See also :meth:`~.interpret_measurement`.

        """
        invals = [self.read(invar) for invar in eqn.invars]
        invals = [self.read_with_trace(self.read(inval)) for inval in invals]
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
        with jax.core.new_main(TransformTrace, transform_program=self._transform_program) as main:
            self._trace = main.with_cur_sublevel()
            tracers_out = super().eval(jaxpr, consts, *args)

        return [r.val if isinstance(r, TransformTracer) else r for r in tracers_out]


# pylint: disable=unused-argument, too-many-arguments, protected-access
@TransformInterpreter.register_primitive(qnode_prim)
def handle_qnode(self, *invals, shots, qnode, device, qnode_kwargs, qfunc_jaxpr, n_consts):
    """Handle a qnode primitive."""
    consts = invals[:n_consts]

    self._state["shots"] = qml.measurements.Shots(shots)  # pylint: disable=protected-access
    new_qfunc_jaxpr = jaxpr_to_jaxpr(copy(self), qfunc_jaxpr, consts, *invals[n_consts:])

    return qnode_prim.bind(
        *invals,
        shots=shots,
        qnode=qnode,
        device=device,
        qnode_kwargs=qnode_kwargs,
        qfunc_jaxpr=new_qfunc_jaxpr,
        n_consts=n_consts,
    )
