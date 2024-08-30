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
"""
This submodule defines a strategy structure for defining custom plxpr interpreters
"""

import copy
from functools import partial, wraps
from typing import Callable

# note: the module has a jax dependency and cannot exist in the standard import path for now.
import jax

import pennylane as qml
from pennylane import cond
from pennylane.compiler.qjit_api import (
    _get_for_loop_qfunc_prim,
    _get_while_loop_qfunc_prim,
    for_loop,
    while_loop,
)
from pennylane.ops.op_math.adjoint import _get_adjoint_qfunc_prim, adjoint
from pennylane.ops.op_math.condition import _get_cond_qfunc_prim
from pennylane.ops.op_math.controlled import _get_ctrl_qfunc_prim, ctrl

from .capture_qnode import _get_qnode_prim
from .primitives import _get_abstract_measurement, _get_abstract_operator

for_prim = _get_for_loop_qfunc_prim()
while_prim = _get_while_loop_qfunc_prim()
cond_prim = _get_cond_qfunc_prim()
qnode_prim = _get_qnode_prim()
adjoint_transform_prim = _get_adjoint_qfunc_prim()
ctrl_transform_prim = _get_ctrl_qfunc_prim()

AbstractOperator = _get_abstract_operator()
AbstractMeasurement = _get_abstract_measurement()


class PlxprInterpreter:
    """A template base class for defining plxpr interpreters

    Args:
        state (Any): any kind of information that may need to get carried around between different interpreters.

    **State property:**

    Higher order primitives can often be handled by a separate interpreter, but need to reference or modify the same values.
    For example, a device interpreter may need to modify a statevector, or conversion to a tape may need to modify operations
    and measurement lists. By maintaining this information in the optional ``state`` property, this information can automatically
    by passed to new sub-interpreters.


    **Examples:**

    .. code-block:: python

        class SimplifyInterpreter(PlxprInterpreter):

        def interpret_operation(self, op):
            new_op = op.simplify()
            if new_op is op:
                # if new op isn't queued, need to requeue op.
                data, struct = jax.tree_util.tree_flatten(new_op)
                new_op = jax.tree_util.tree_unflatten(struct, data)
            return new_op

    """

    _env: dict
    _primitive_registrations: dict["jax.core.Primitive", Callable] = {}
    _op_math_cache: dict

    def __init_subclass__(cls) -> None:
        cls._primitive_registrations = copy.copy(cls._primitive_registrations)

    def __init__(self, state=None):
        self._env = {}
        self._op_math_cache = {}
        self.state = state

    @classmethod
    def register_primitive(cls, primitive: "jax.core.Primitive") -> Callable[[Callable], Callable]:
        """Registers a custom method for handling a primitive

        Args:
            primitive (jax.core.Primitive): the primitive we want  custom behavior for

        Returns:
            Callable: a decorator for adding a function to the custom registrations map

        Side Effect:
            Calling the returned decorator with a function will place the function into the
            primitive registrations map.

        ```
        my_primitive = jax.core.Primitive("my_primitve")

        @Interpreter_Type.register(my_primitive)
        def handle_my_primitive(self: Interpreter_Type, *invals, **params)
            return invals[0] + invals[1] # some sort of custom handling
        ```

        """

        def decorator(f: Callable) -> Callable:
            cls._primitive_registrations[primitive] = f
            return f

        return decorator

    # pylint: disable=unidiomatic-typecheck
    def read(self, var):
        """Extract the value corresponding to a variable."""
        if self._env is None:
            raise ValueError("_env not yet initialized.")
        if type(var) is jax.core.Literal:
            return var.val
        return var.val if type(var) is jax.core.Literal else self._env[var]

    def setup(self):
        """Initialize the instance before interpretting equations.

        Blank by default, this method can initialize any additional instance variables
        needed by an interpreter

        """

    def cleanup(self):
        """Perform any final steps after iterating through all equations.

        Blank by default, this method can clean up instance variables, or perform
        equations that have been deffered till later.

        """

    def interpret_operation(self, op: "pennylane.operation.Operator"):
        """Interpret a PennyLane operation instance.

        Args:
            op (Operator): a pennylane operator instance

        Returns:
            Any

        This method is only called when the operator's output is a dropped variable,
        so the output will not effect later equations in the circuit.

        See also: :meth:`~.interpret_operation_eqn`.

        """
        return op

    def interpret_operation_eqn(self, eqn: "jax.core.JaxprEqn"):
        """Interpret an equation corresponding to an operator.

        Args:
            primitive (jax.core.Primitive): a jax primitive corresponding to an operation
            outvar
            *invals (Any): the positional input variables for the equation

        Keyword Args:
            **params: The equations parameters dictionary

        See also: :meth:`~.interpret_operation`.

        """

        invals = [
            (
                invar.val
                if type(invar) is jax.core.Literal
                else self._op_math_cache.get(invar, self.read(invar))
            )
            for invar in eqn.invars
        ]
        op = eqn.primitive.impl(*invals, **eqn.params)
        if isinstance(eqn.outvars[0], jax.core.DropVar):
            return self.interpret_operation(op)

        self._op_math_cache[eqn.outvars[0]] = op
        return op

    def interpret_measurement_eqn(self, primitive, *invals, **params):
        """Interpret an equation corresponding to a measurement process.

        Args:
            primitive (jax.core.Primitive): a jax primitive corresponding to a measurement.
            *invals (Any): the positional input variables for the equation

        Keyword Args:
            **params: The equations parameters dictionary

        """
        return primitive.bind(*invals, **params)

    def eval(self, jaxpr: "jax.core.Jaxpr", consts: list, *args) -> list:
        """Evaluate a jaxpr.

        Args:
            jaxpr (jax.core.Jaxpr): the jaxpr to evaluate
            consts (list[TensorLike]): the constant variables for the jaxpr
            *args (tuple[TensorLike]): The arguments for the jaxpr.

        Returns:
            list[TensorLike]: the results of the execution.

        """
        self._env = {}
        self._op_math_cache = {}
        self.setup()

        for arg, invar in zip(args, jaxpr.invars):
            self._env[invar] = arg
        for const, constvar in zip(consts, jaxpr.constvars):
            self._env[constvar] = const

        for eqn in jaxpr.eqns:

            custom_handler = self._primitive_registrations.get(eqn.primitive, None)
            if custom_handler:
                invals = [self.read(invar) for invar in eqn.invars]
                outvals = custom_handler(self, *invals, **eqn.params)
            elif isinstance(eqn.outvars[0].aval, AbstractOperator):
                outvals = self.interpret_operation_eqn(eqn)
            elif isinstance(eqn.outvars[0].aval, AbstractMeasurement):
                invals = [self.read(invar) for invar in eqn.invars]
                outvals = self.interpret_measurement_eqn(eqn.primitive, *invals, **eqn.params)
            else:
                invals = [self.read(invar) for invar in eqn.invars]
                outvals = eqn.primitive.bind(*invals, **eqn.params)

            if not eqn.primitive.multiple_results:
                outvals = [outvals]
            for outvar, outval in zip(eqn.outvars, outvals):
                self._env[outvar] = outval

        self.cleanup()
        # Read the final result of the Jaxpr from the environment
        outvals = []
        for var in jaxpr.outvars:
            if var in self._op_math_cache:
                outvals.append(self.interpret_operation(self._op_math_cache[var]))
            else:
                outvals.append(self.read(var))
        self._op_math_cache = {}
        self._env = {}
        return outvals

    def __call__(self, f: Callable) -> Callable:
        @wraps(f)
        def wrapper(*args, **kwargs):
            jaxpr = jax.make_jaxpr(partial(f, **kwargs))(*args)
            return self.eval(jaxpr.jaxpr, jaxpr.consts, *args)

        return wrapper


@PlxprInterpreter.register_primitive(adjoint_transform_prim)
def handle_adjoint_transform(self, *invals, jaxpr, lazy, n_consts):
    """Interpret an adjoint transform primitive."""
    consts = invals[:n_consts]
    args = invals[n_consts:]

    def new_qfunc(*inner_args):
        return type(self)(state=self.state).eval(jaxpr, consts, *inner_args)

    jaxpr = jax.make_jaxpr(new_qfunc)(*args)

    return adjoint_transform_prim.bind(*invals, jaxpr=jaxpr.jaxpr, lazy=lazy, n_consts=n_consts)


# pylint: disable=too-many-arguments
@PlxprInterpreter.register_primitive(ctrl_transform_prim)
def handle_ctrl_transform(self, *invals, n_control, jaxpr, control_values, work_wires, n_consts):
    """Interpret a ctrl transform primitive."""
    consts = invals[:n_consts]
    control_wires = invals[-n_control:]
    args = invals[n_consts:-n_control]

    def new_qfunc(*inner_args):
        return type(self)(state=self.state).eval(jaxpr, consts, *inner_args)

    jaxpr = jax.make_jaxpr(new_qfunc)(*args)

    return ctrl_transform_prim.bind(
        *invals,
        n_control=n_control,
        jaxpr=jaxpr.jaxpr,
        control_values=control_values,
        work_wires=work_wires,
        n_consts=n_consts,
    )


@PlxprInterpreter.register_primitive(for_prim)
def handle_for_loop(self, *invals, jaxpr_body_fn, n_consts):
    """Handle a for loop primitive."""
    start, stop, step = invals[0], invals[1], invals[2]
    consts = invals[3 : 3 + n_consts]

    @for_loop(start, stop, step)
    def g(i, *init_state):
        return type(self)(state=self.state).eval(jaxpr_body_fn.jaxpr, consts, i, *init_state)

    return g(*invals[3 + n_consts :])


@PlxprInterpreter.register_primitive(cond_prim)
def handle_cond(self, *invals, jaxpr_branches, n_consts_per_branch, n_args):
    """Handle a cond primitive."""
    n_branches = len(jaxpr_branches)
    conditions = invals[:n_branches]
    consts_flat = invals[n_branches + n_args :]

    def true_fn(*args):
        return type(self)(state=self.state).eval(
            jaxpr_branches[0].jaxpr, consts_flat[: n_consts_per_branch[0]], *args
        )

    def false_fn(*args):
        return type(self)(state=self.state).eval(
            jaxpr_branches[-1].jaxpr, consts_flat[n_consts_per_branch[0] :], *args
        )

    cond_fn = cond(conditions[0], true_fn, false_fn=false_fn)
    return cond_fn(*invals[n_branches : n_branches + n_args])


@PlxprInterpreter.register_primitive(while_prim)
def handle_while_loop(self, *invals, jaxpr_body_fn, jaxpr_cond_fn, n_consts_body, n_consts_cond):
    """Handle a while loop primitive."""
    consts_body = invals[:n_consts_body]
    consts_cond = invals[n_consts_body : n_consts_body + n_consts_cond]
    init_state = invals[n_consts_body + n_consts_cond :]

    def cond_fn(*args):
        return jax.core.eval_jaxpr(jaxpr_cond_fn.jaxpr, consts_cond, *args)

    @while_loop(cond_fn)
    def loop(*args):
        return type(self)(state=self.state).eval(jaxpr_body_fn.jaxpr, consts_body, *args)

    return loop(*init_state)


# pylint: disable=unused-argument, too-many-arguments
@PlxprInterpreter.register_primitive(qnode_prim)
def handle_qnode(self, *invals, shots, qnode, device, qnode_kwargs, qfunc_jaxpr, n_consts):
    """Handle a qnode primitive."""
    consts = invals[:n_consts]

    @qml.qnode(device, **qnode_kwargs)
    def new_qnode(*args):
        return type(self)(state=self.state).eval(qfunc_jaxpr, consts, *args)

    return new_qnode(*invals[n_consts:], shots=shots)
