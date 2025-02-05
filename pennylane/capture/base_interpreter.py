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
# pylint: disable=no-self-use
from copy import copy
from functools import partial, wraps
from typing import Callable, Optional, Sequence

import jax

import pennylane as qml

from .flatfn import FlatFn
from .primitives import (
    adjoint_transform_prim,
    cond_prim,
    ctrl_transform_prim,
    for_loop_prim,
    grad_prim,
    jacobian_prim,
    qnode_prim,
    while_loop_prim,
)

FlattenedHigherOrderPrimitives: dict["jax.core.Primitive", Callable] = {}
"""
A dictionary containing flattened style cond, while, and for loop higher order primitives.
.. code-block::
    MyInterpreter._primitive_registrations.update(FlattenedHigherOrderPrimitives)
"""


def _fill_in_shape_with_dyn_shape(dyn_shape: tuple["jax.core.Tracer"], shape: tuple[Optional[int]]):
    """
    A helper for broadcast_in_dim and iota to combine static dimensions and dynamic dimensions.

    For example, with `shape=(None, 4, None)` and `dyn_shape=(a, b)`, then the processed shape is
    `(a, 4, b)`.

    When capturing `broadcast_in_dim_p` with a dynamic shape, we might end up with:
    ```
    >>> import jax
    >>> qml.capture.enable()
    >>> jax.config.update("jax_dynamic_shapes", True)
    >>> def f(n):
    ...     return jax.numpy.ones((n, 4, n))
    >>> jax.make_jaxpr(f)(4)
    { lambda ; a:i32[]. let
        b:f32[a,4,a] = broadcast_in_dim[
        broadcast_dimensions=()
        shape=(None, 4, None)
        ] 1.0 a a
    in (b,) }
    ```

    `1.0` is the value we want to fill. `a, a` are the two dynamic shapes.
    The static part of the shape is `(None, 4, None)`. We need to replace the two `None`
    values with `a` and `a`.

    `broadcast_in_dim` also can't handle shapes where an integer is a concrete jax arrays,
    so we need to convert any concrete jax arrays to normal integers.

    """
    dyn_shape_iter = iter(dyn_shape)
    new_shape = []
    for s in shape:
        if s is not None:
            new_shape.append(s)
        else:
            # pull from iterable of dynamic shapes
            next_s = next(dyn_shape_iter)
            if not qml.math.is_abstract(next_s):
                # may need to cast to a built-in integer if possible
                next_s = int(next_s)
            new_shape.append(next_s)

    return tuple(new_shape)


def jaxpr_to_jaxpr(
    interpreter: "PlxprInterpreter", jaxpr: "jax.core.Jaxpr", consts, *args
) -> "jax.core.Jaxpr":
    """A convenience utility for converting jaxpr to a new jaxpr via an interpreter."""

    f = partial(interpreter.eval, jaxpr, consts)

    return jax.make_jaxpr(f)(*args)


class PlxprInterpreter:
    """A base class for defining plxpr interpreters.

    **Examples:**

    .. code-block:: python

        import jax
        from pennylane.capture import PlxprInterpreter

        class SimplifyInterpreter(PlxprInterpreter):

            def interpret_operation(self, op):
                new_op = qml.simplify(op)
                if new_op is op:
                    # simplify didnt create a new operator, so it didnt get captured
                    data, struct = jax.tree_util.tree_flatten(new_op)
                    new_op = jax.tree_util.tree_unflatten(struct, data)
                return new_op

            def interpret_measurement(self, measurement):
                new_mp = measurement.simplify()
                if new_mp is measurement:
                    new_mp = new_mp._unflatten(*measurement._flatten())
                    # if new op isn't queued, need to requeue op.
                return new_mp

    Now the interpreter can be used to transform functions and jaxpr:

    >>> qml.capture.enable()
    >>> interpreter = SimplifyInterpreter()
    >>> def f(x):
    ...     qml.RX(x, 0)**2
    ...     qml.adjoint(qml.Z(0))
    ...     return qml.expval(qml.X(0) + qml.X(0))
    >>> simplified_f = interpreter(f)
    >>> print(qml.draw(simplified_f)(0.5))
    0: ──RX(1.00)──Z─┤  <2.00*X>
    >>> jaxpr = jax.make_jaxpr(f)(0.5)
    >>> interpreter.eval(jaxpr.jaxpr, [], 0.5)
    [expval(2.0 * X(0))]

    **Handling higher order primitives:**

    Two main strategies exist for handling higher order primitives (primitives with jaxpr as metadata).
    The first one is structure preserving (tracing the execution preserves the higher order primitive),
    and the second one is structure flattening (tracing the execution eliminates the higher order primitive).

    Compilation transforms, like the above ``SimplifyInterpreter``, may prefer to handle higher order primitives
    via a structure-preserving method. After transforming the jaxpr, the `for_loop` still exists. This maintains
    the compact structure of the jaxpr and reduces the size of the program. This behavior is the default.

    >>> def g(x):
    ...     @qml.for_loop(3)
    ...     def loop(i, x):
    ...         qml.RX(x, 0) ** i
    ...         return x
    ...     loop(1.0)
    ...     return qml.expval(qml.Z(0) + 3*qml.Z(0))
    >>> jax.make_jaxpr(interpreter(g))(0.5)
    { lambda ; a:f32[]. let
        _:f32[] = for_loop[
          args_slice=slice(0, None, None)
          consts_slice=slice(0, 0, None)
          jaxpr_body_fn={ lambda ; b:i32[] c:f32[]. let
            d:f32[] = convert_element_type[new_dtype=float32 weak_type=True] b
            e:f32[] = mul c d
            _:AbstractOperator() = RX[n_wires=1] e 0
          in (c,) }
        ] 0 3 1 1.0
        f:AbstractOperator() = PauliZ[n_wires=1] 0
        g:AbstractOperator() = SProd[_pauli_rep=4.0 * Z(0)] 4.0 f
        h:AbstractMeasurement(n_wires=None) = expval_obs g
      in (h,) }

    Accumulation transforms, like device execution or conversion to tapes, may need to flatten out
    the higher order primitive to execute it.

    .. code-block:: python

        import copy

        class AccumulateOps(PlxprInterpreter):

            def __init__(self, ops=None):
                self.ops = ops

            def setup(self):
                if self.ops is None:
                    self.ops = []

            def interpret_operation(self, op):
                self.ops.append(op)

        @AccumulateOps.register_primitive(qml.capture.primitives.for_loop_prim)
        def _(self, start, stop, step, *invals, jaxpr_body_fn, consts_slice, args_slice):
            consts = invals[consts_slice]
            state = invals[args_slice]

            for i in range(start, stop, step):
                state = copy.copy(self).eval(jaxpr_body_fn, consts, i, *state)
            return state

    >>> @qml.for_loop(3)
    ... def loop(i, x):
    ...     qml.RX(x, i)
    ...     return x
    >>> accumulator = AccumulateOps()
    >>> accumulator(loop)(0.5)
    >>> accumulator.ops
    [RX(0.5, wires=[0]), RX(0.5, wires=[1]), RX(0.5, wires=[2])]

    In this case, we need to actually evaluate the jaxpr 3 times using our interpreter. If jax's
    evaluation interpreter ran it three times, we wouldn't actually manage to accumulate the operations.
    """

    _env: dict
    _primitive_registrations: dict["jax.core.Primitive", Callable] = {}

    def __init_subclass__(cls) -> None:
        cls._primitive_registrations = copy(cls._primitive_registrations)

    def __init__(self):
        self._env = {}

    @classmethod
    def register_primitive(cls, primitive: "jax.core.Primitive") -> Callable[[Callable], Callable]:
        """Registers a custom method for handling a primitive

        Args:
            primitive (jax.core.Primitive): the primitive we want custom behavior for

        Returns:
            Callable: a decorator for adding a function to the custom registrations map

        Side Effect:
            Calling the returned decorator with a function will place the function into the
            primitive registrations map.

        .. code-block:: python

            my_primitive = jax.core.Primitive("my_primitve")

            @Interpreter_Type.register(my_primitive)
            def handle_my_primitive(self: Interpreter_Type, *invals, **params)
                return invals[0] + invals[1] # some sort of custom handling

        """

        def decorator(f: Callable) -> Callable:
            cls._primitive_registrations[primitive] = f
            return f

        return decorator

    def read(self, var):
        """Extract the value corresponding to a variable."""
        return var.val if isinstance(var, jax.core.Literal) else self._env[var]

    def setup(self) -> None:
        """Initialize the instance before interpreting equations.

        Blank by default, this method can initialize any additional instance variables
        needed by an interpreter. For example, a device interpreter could initialize a statevector,
        or a compilation interpreter could initialize a staging area for the latest operation on each wire.
        """

    def cleanup(self) -> None:
        """Perform any final steps after iterating through all equations.

        Blank by default, this method can clean up instance variables. Particularly,
        this method can be used to deallocate qubits and registers when converting to
        a Catalyst variant jaxpr.
        """

    def interpret_operation(self, op: "pennylane.operation.Operator"):
        """Interpret a PennyLane operation instance.

        Args:
            op (Operator): a pennylane operator instance

        Returns:
            Any

        This method is only called when the operator's output is a dropped variable,
        so the output will not affect later equations in the circuit.

        See also: :meth:`~.interpret_operation_eqn`.

        """
        data, struct = jax.tree_util.tree_flatten(op)
        return jax.tree_util.tree_unflatten(struct, data)

    def interpret_operation_eqn(self, eqn: "jax.core.JaxprEqn"):
        """Interpret an equation corresponding to an operator.

        Args:
            eqn (jax.core.JaxprEqn): a jax equation for an operator.

        See also: :meth:`~.interpret_operation`.

        """
        invals = (self.read(invar) for invar in eqn.invars)
        with qml.QueuingManager.stop_recording():
            op = eqn.primitive.impl(*invals, **eqn.params)
        if isinstance(eqn.outvars[0], jax.core.DropVar):
            return self.interpret_operation(op)
        return op

    def interpret_measurement_eqn(self, eqn: "jax.core.JaxprEqn"):
        """Interpret an equation corresponding to a measurement process.

        Args:
            eqn (jax.core.JaxprEqn)

        See also :meth:`~.interpret_measurement`.

        """
        invals = (self.read(invar) for invar in eqn.invars)
        with qml.QueuingManager.stop_recording():
            mp = eqn.primitive.impl(*invals, **eqn.params)
        return self.interpret_measurement(mp)

    def interpret_measurement(self, measurement: "qml.measurement.MeasurementProcess"):
        """Interpret a measurement process instance.

        Args:
            measurement (MeasurementProcess): a measurement instance.

        See also :meth:`~.interpret_measurement_eqn`.

        """
        data, struct = jax.tree_util.tree_flatten(measurement)
        return jax.tree_util.tree_unflatten(struct, data)

    def eval(self, jaxpr: "jax.core.Jaxpr", consts: Sequence, *args) -> list:
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
                invals = [self.read(invar) for invar in eqn.invars]
                outvals = custom_handler(self, *invals, **eqn.params)
            elif getattr(primitive, "prim_type", "") == "operator":
                outvals = self.interpret_operation_eqn(eqn)
            elif getattr(primitive, "prim_type", "") == "measurement":
                outvals = self.interpret_measurement_eqn(eqn)
            else:
                invals = [self.read(invar) for invar in eqn.invars]
                subfuns, params = primitive.get_bind_params(eqn.params)
                outvals = primitive.bind(*subfuns, *invals, **params)

            if not primitive.multiple_results:
                outvals = [outvals]
            for outvar, outval in zip(eqn.outvars, outvals, strict=True):
                self._env[outvar] = outval

        # Read the final result of the Jaxpr from the environment
        outvals = []
        for var in jaxpr.outvars:
            outval = self.read(var)
            if isinstance(outval, qml.operation.Operator):
                outvals.append(self.interpret_operation(outval))
            else:
                outvals.append(outval)
        self.cleanup()
        self._env = {}
        return outvals

    def __call__(self, f: Callable) -> Callable:

        flat_f = FlatFn(f)

        @wraps(f)
        def wrapper(*args, **kwargs):
            with qml.QueuingManager.stop_recording():
                jaxpr = jax.make_jaxpr(partial(flat_f, **kwargs))(*args)
            results = self.eval(jaxpr.jaxpr, jaxpr.consts, *args)
            assert flat_f.out_tree
            # slice out any dynamic shape variables
            results = results[-flat_f.out_tree.num_leaves :]
            return jax.tree_util.tree_unflatten(flat_f.out_tree, results)

        return wrapper


# pylint: disable=unused-argument
@PlxprInterpreter.register_primitive(jax.lax.broadcast_in_dim_p)
def _(self, x, *dyn_shape, shape, broadcast_dimensions):
    """Handle the broadcast_in_dim primitive created by jnp.ones, jnp.zeros, jnp.full

    >>> import jax
    >>> qml.capture.enable()
    >>> jax.config.update("jax_dynamic_shapes", True)
    >>> def f(n):
    ...     return jax.numpy.ones((n, 4, n))
    >>> jax.make_jaxpr(f)(4)
    { lambda ; a:i32[]. let
        b:f32[a,4,a] = broadcast_in_dim[
        broadcast_dimensions=()
        shape=(None, 4, None)
        ] 1.0 a a
    in (b,) }

    """
    # needs custom primitive as jax.core.eval_jaxpr will error out with this
    new_shape = _fill_in_shape_with_dyn_shape(dyn_shape, shape)

    return jax.lax.broadcast_in_dim(x, new_shape, broadcast_dimensions=broadcast_dimensions)


# pylint: disable=unused-argument
@PlxprInterpreter.register_primitive(jax.lax.iota_p)
def _(self, *dyn_shape, dimension, dtype, shape):
    """Handle the iota primitive created by jnp.arange

    >>> import jax
    >>> qml.capture.enable()
    >>> jax.config.update("jax_dynamic_shapes", True)
    >>> def f(n):
    ...     return jax.numpy.arange(n)
    >>> jax.make_jaxpr(f)(4)
    { lambda ; a:i32[]. let
    b:i32[a] = iota[dimension=0 dtype=int32 shape=(None,)] a
    in (b,) }
    """
    # iota is primitive created by jnp.arange
    new_shape = _fill_in_shape_with_dyn_shape(dyn_shape, shape)
    return jax.lax.broadcasted_iota(dtype, new_shape, dimension)


@PlxprInterpreter.register_primitive(adjoint_transform_prim)
def handle_adjoint_transform(self, *invals, jaxpr, lazy, n_consts):
    """Interpret an adjoint transform primitive."""
    consts = invals[:n_consts]
    args = invals[n_consts:]
    jaxpr = jaxpr_to_jaxpr(copy(self), jaxpr, consts, *args)

    return adjoint_transform_prim.bind(
        *jaxpr.consts, *args, jaxpr=jaxpr.jaxpr, lazy=lazy, n_consts=len(jaxpr.consts)
    )


# pylint: disable=too-many-arguments
@PlxprInterpreter.register_primitive(ctrl_transform_prim)
def handle_ctrl_transform(self, *invals, n_control, jaxpr, control_values, work_wires, n_consts):
    """Interpret a ctrl transform primitive."""
    consts = invals[:n_consts]
    args = invals[n_consts:-n_control]
    jaxpr = jaxpr_to_jaxpr(copy(self), jaxpr, consts, *args)

    return ctrl_transform_prim.bind(
        *jaxpr.consts,
        *args,
        *invals[-n_control:],
        n_control=n_control,
        jaxpr=jaxpr.jaxpr,
        control_values=control_values,
        work_wires=work_wires,
        n_consts=len(jaxpr.consts),
    )


@PlxprInterpreter.register_primitive(for_loop_prim)
def handle_for_loop(
    self, start, stop, step, *args, jaxpr_body_fn, consts_slice, args_slice, abstract_shapes_slice
):
    """Handle a for loop primitive."""
    consts = args[consts_slice]
    init_state = args[args_slice]
    abstract_shapes = args[abstract_shapes_slice]
    new_jaxpr_body_fn = jaxpr_to_jaxpr(
        copy(self), jaxpr_body_fn, consts, *abstract_shapes, start, *init_state
    )

    consts_slice = slice(0, len(new_jaxpr_body_fn.consts))
    abstract_shapes_slice = slice(consts_slice.stop, consts_slice.stop + len(abstract_shapes))
    args_slice = slice(abstract_shapes_slice.stop, None)
    return for_loop_prim.bind(
        start,
        stop,
        step,
        *new_jaxpr_body_fn.consts,
        *abstract_shapes,
        *init_state,
        jaxpr_body_fn=new_jaxpr_body_fn.jaxpr,
        consts_slice=consts_slice,
        args_slice=args_slice,
        abstract_shapes_slice=abstract_shapes_slice,
    )


@PlxprInterpreter.register_primitive(cond_prim)
def handle_cond(self, *invals, jaxpr_branches, consts_slices, args_slice):
    """Handle a cond primitive."""
    args = invals[args_slice]

    new_jaxprs = []
    new_consts = []
    new_consts_slices = []
    end_const_ind = len(jaxpr_branches)

    for const_slice, jaxpr in zip(consts_slices, jaxpr_branches):
        consts = invals[const_slice]
        if jaxpr is None:
            new_jaxprs.append(None)
            new_consts_slices.append(slice(0, 0))
        else:
            new_jaxpr = jaxpr_to_jaxpr(copy(self), jaxpr, consts, *args)
            new_jaxprs.append(new_jaxpr.jaxpr)
            new_consts.extend(new_jaxpr.consts)
            new_consts_slices.append(slice(end_const_ind, end_const_ind + len(new_jaxpr.consts)))
            end_const_ind += len(new_jaxpr.consts)

    new_args_slice = slice(end_const_ind, None)
    return cond_prim.bind(
        *invals[: len(jaxpr_branches)],
        *new_consts,
        *args,
        jaxpr_branches=new_jaxprs,
        consts_slices=new_consts_slices,
        args_slice=new_args_slice,
    )


@PlxprInterpreter.register_primitive(while_loop_prim)
def handle_while_loop(
    self,
    *invals,
    jaxpr_body_fn,
    jaxpr_cond_fn,
    body_slice,
    cond_slice,
    args_slice,
    abstract_shapes_slice,
):
    """Handle a while loop primitive."""
    consts_body = invals[body_slice]
    consts_cond = invals[cond_slice]
    init_state = invals[args_slice]
    abstract_shapes = invals[abstract_shapes_slice]

    new_jaxpr_body_fn = jaxpr_to_jaxpr(
        copy(self), jaxpr_body_fn, consts_body, *abstract_shapes, *init_state
    )
    new_jaxpr_cond_fn = jaxpr_to_jaxpr(
        copy(self), jaxpr_cond_fn, consts_cond, *abstract_shapes, *init_state
    )

    body_consts = slice(0, len(new_jaxpr_body_fn.consts))
    cond_consts = slice(body_consts.stop, body_consts.stop + len(new_jaxpr_cond_fn.consts))
    abstract_shapes_slice = slice(cond_consts.stop, cond_consts.stop + len(abstract_shapes))
    args_slice = slice(abstract_shapes_slice.stop, None)

    return while_loop_prim.bind(
        *new_jaxpr_body_fn.consts,
        *new_jaxpr_cond_fn.consts,
        *abstract_shapes,
        *init_state,
        jaxpr_body_fn=new_jaxpr_body_fn.jaxpr,
        jaxpr_cond_fn=new_jaxpr_cond_fn.jaxpr,
        body_slice=body_consts,
        cond_slice=cond_consts,
        args_slice=args_slice,
        abstract_shapes_slice=abstract_shapes_slice,
    )


# pylint: disable=unused-argument, too-many-arguments
@PlxprInterpreter.register_primitive(qnode_prim)
def handle_qnode(self, *invals, shots, qnode, device, qnode_kwargs, qfunc_jaxpr, n_consts):
    """Handle a qnode primitive."""
    consts = invals[:n_consts]
    args = invals[n_consts:]

    new_qfunc_jaxpr = jaxpr_to_jaxpr(copy(self), qfunc_jaxpr, consts, *args)

    return qnode_prim.bind(
        *new_qfunc_jaxpr.consts,
        *args,
        shots=shots,
        qnode=qnode,
        device=device,
        qnode_kwargs=qnode_kwargs,
        qfunc_jaxpr=new_qfunc_jaxpr.jaxpr,
        n_consts=len(new_qfunc_jaxpr.consts),
    )


@PlxprInterpreter.register_primitive(grad_prim)
def handle_grad(self, *invals, jaxpr, n_consts, **params):
    """Handle the grad primitive."""
    consts = invals[:n_consts]
    args = invals[n_consts:]
    new_jaxpr = jaxpr_to_jaxpr(copy(self), jaxpr, consts, *args)
    return grad_prim.bind(
        *new_jaxpr.consts, *args, jaxpr=new_jaxpr.jaxpr, n_consts=len(new_jaxpr.consts), **params
    )


@PlxprInterpreter.register_primitive(jacobian_prim)
def handle_jacobian(self, *invals, jaxpr, n_consts, **params):
    """Handle the jacobian primitive."""
    consts = invals[:n_consts]
    args = invals[n_consts:]
    new_jaxpr = jaxpr_to_jaxpr(copy(self), jaxpr, consts, *args)
    return jacobian_prim.bind(
        *new_jaxpr.consts, *args, jaxpr=new_jaxpr.jaxpr, n_consts=len(new_jaxpr.consts), **params
    )


def flatten_while_loop(
    self,
    *invals,
    jaxpr_body_fn,
    jaxpr_cond_fn,
    body_slice,
    cond_slice,
    args_slice,
    abstract_shapes_slice,
):
    """Handle the while loop by a flattened python strategy."""
    consts_body = invals[body_slice]
    consts_cond = invals[cond_slice]
    init_state = invals[args_slice]
    abstract_shapes = invals[abstract_shapes_slice]

    fn_res = init_state
    while copy(self).eval(jaxpr_cond_fn, consts_cond, *abstract_shapes, *fn_res)[0]:
        fn_res = copy(self).eval(jaxpr_body_fn, consts_body, *abstract_shapes, *fn_res)

    return fn_res


FlattenedHigherOrderPrimitives[while_loop_prim] = flatten_while_loop


def flattened_cond(self, *invals, jaxpr_branches, consts_slices, args_slice):
    """Handle the cond primitive by a flattened python strategy."""
    n_branches = len(jaxpr_branches)
    conditions = invals[:n_branches]
    args = invals[args_slice]

    for pred, jaxpr, const_slice in zip(conditions, jaxpr_branches, consts_slices):
        consts = invals[const_slice]
        if pred and jaxpr is not None:
            return copy(self).eval(jaxpr, consts, *args)
    return ()


FlattenedHigherOrderPrimitives[cond_prim] = flattened_cond


def flattened_for(
    self, start, stop, step, *invals, jaxpr_body_fn, consts_slice, args_slice, abstract_shapes_slice
):
    """Handle the for loop by a flattened python strategy."""
    consts = invals[consts_slice]
    init_state = invals[args_slice]
    abstract_shapes = invals[abstract_shapes_slice]

    res = init_state
    for i in range(start, stop, step):
        res = copy(self).eval(jaxpr_body_fn, consts, *abstract_shapes, i, *res)

    return res


FlattenedHigherOrderPrimitives[for_loop_prim] = flattened_for
