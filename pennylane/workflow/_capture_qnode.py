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
This submodule defines a capture compatible call to QNodes.

Workflow Development Status
---------------------------

The non-exhaustive list of unsupported features are:

**Breaking ``vmap``/parameter broadcasting into a non-broadcasted state**. The current workflow assumes
that the device execution can natively handle broadcasted parameters. ``vmap`` and parameter broadcasting
will not work with devices other than default qubit.

>>> @qml.qnode(qml.device('lightning.qubit', wires=1))
... def circuit(x):
...     qml.RX(x, 0)
...     return qml.expval(qml.Z(0))
>>> jax.vmap(circuit)(jax.numpy.array([1.0, 2.0, 3.0]))
TypeError: RX(): incompatible function arguments. The following argument types are supported:
    1. (self: pennylane_lightning.lightning_qubit_ops.StateVectorC128, arg0: list[int], arg1: bool, arg2: list[float]) -> None
    2. (self: pennylane_lightning.lightning_qubit_ops.StateVectorC128, arg0: list[int], arg1: list[bool], arg2: list[int], arg3: bool, arg4: list[float]) -> None

**Grouping commuting measurements and/or splitting up non-commuting measurements.** Currently, each
measurement is fully independent and generated from different raw samples than every other measurement.
To generate multiple measurement from the same samples, we need a way of denoting which measurements
should be taken together. A "Combination measurement process" higher order primitive, or something like it.
We will also need to figure out how to implement splitting up a circuit with non-commuting measurements into
multiple circuits.

>>> @partial(qml.set_shots, shots=5)
... @qml.qnode(qml.device('default.qubit', wires=1))
... def circuit():
...     qml.H(0)
...     return qml.sample(wires=0), qml.sample(wires=0)
>>> circuit()
(Array([1, 0, 1, 0, 0], dtype=int64), Array([0, 0, 1, 0, 0], dtype=int64))

**Figuring out what types of data can be sent to the device.** Is the device always
responsible for converting jax arrays to numpy arrays? Is the device responsible for having a
pure-callback boundary if the execution is not jittable? We do have an opportunity here
to have GPU end-to-end simulation on ``lightning.gpu`` and ``lightning.kokkos``.

**Jitting workflows involving qnodes**. While the execution of jaxpr on ``default.qubit`` is
currently jittable, we will need to register a lowering for the qnode primitive.  We will also
need to figure out where to apply a ``jax.pure_callback`` for devices like ``lightning.qubit`` that are
not jittable.

**Result caching**. The new workflow is not capable of caching the results of executions, and we have
not even started thinking about how it might be possible to do so.

**Unknown other features**. The workflow currently has limited testing, so this list of unsupported
features is non-exhaustive.

"""
import logging
from collections.abc import Sequence
from functools import partial
from numbers import Number
from warnings import warn

import jax
from jax.interpreters import ad, batching, mlir
from jax.interpreters import partial_eval as pe

import pennylane as qml
from pennylane.capture import FlatFn, QmlPrimitive
from pennylane.exceptions import CaptureError
from pennylane.logging import debug_logger
from pennylane.measurements import Shots
from pennylane.typing import TensorLike

from .construct_execution_config import construct_execution_config

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _is_scalar_tensor(arg) -> bool:
    """Check if an argument is a scalar tensor-like object or a numeric scalar."""

    if isinstance(arg, Number):
        return True

    if isinstance(arg, TensorLike):

        if arg.size == 0:
            raise ValueError("Empty tensors are not supported with jax.vmap.")

        if arg.shape == ():
            return True

    return False


def _get_batch_shape(non_const_args, non_const_batch_dims):
    """Calculate the batch shape for the given arguments and batch dimensions."""

    input_shapes = [
        (arg.shape[batch_dim],)
        for arg, batch_dim in zip(non_const_args, non_const_batch_dims, strict=True)
        if batch_dim is not None
    ]

    return jax.lax.broadcast_shapes(*input_shapes)


def _get_shapes_for(*measurements, shots=None, num_device_wires=0, batch_shape=()):
    """Calculate the abstract output shapes for the given measurements."""

    if jax.config.jax_enable_x64:
        dtype_map = {
            float: jax.numpy.float64,
            int: jax.numpy.int64,
            complex: jax.numpy.complex128,
        }
    else:
        dtype_map = {
            float: jax.numpy.float32,
            int: jax.numpy.int32,
            complex: jax.numpy.complex64,
        }

    shapes = []
    if not shots:
        shots = [None]

    for s in shots:
        for m in measurements:
            s = s.val if isinstance(s, jax.extend.core.Literal) else s
            shape, dtype = m.aval.abstract_eval(shots=s, num_device_wires=num_device_wires)
            if all(isinstance(si, int) for si in shape):
                aval_type = jax.core.ShapedArray
            else:
                aval_type = jax.core.DShapedArray
                if not jax.config.jax_dynamic_shapes:
                    raise ValueError(
                        "Returning arrays with a dynamic shape requires setting jax.config.update('jax_dynamic_shapes', True)"
                    )
            dtype = jax.numpy.dtype(dtype_map.get(dtype, dtype))
            shapes.append(aval_type(batch_shape + shape, dtype))
    return shapes


qnode_prim = QmlPrimitive("qnode")
qnode_prim.multiple_results = True
qnode_prim.prim_type = "higher_order"


# pylint: disable=too-many-arguments
@debug_logger
@qnode_prim.def_impl
def _(*args, qnode, device, execution_config, qfunc_jaxpr, n_consts, shots_len, batch_dims=None):

    warn(
        "Executing PennyLane programs with capture enabled should be done inside ``qml.qjit``. Native execution of captured programs is an unmaintained experimental feature.",
        UserWarning,
    )

    execution_config = device.setup_execution_config(execution_config)

    if shots_len == 0:
        shots = None
        non_shots_args = args
    else:
        shots, non_shots_args = args[:shots_len], args[shots_len:]

    consts = non_shots_args[:n_consts]
    non_const_args = non_shots_args[n_consts:]

    device_program = device.preprocess_transforms(execution_config)
    if batch_dims is not None:
        temp_all_args = []
        for a, d in zip(args, batch_dims, strict=True):
            if d is not None:
                slices = [slice(None)] * qml.math.ndim(a)
                slices[d] = 0
                temp_all_args.append(a[tuple(slices)])
            else:
                temp_all_args.append(a)
        temp_consts = temp_all_args[shots_len : (n_consts + shots_len)]
        temp_args = temp_all_args[(n_consts + shots_len) :]
    else:
        temp_consts = consts
        temp_args = non_const_args

    # Expand user transforms applied to the qfunc
    if getattr(qfunc_jaxpr.eqns[0].primitive, "prim_type", "") == "transform":
        transformed_func = qml.capture.expand_plxpr_transforms(
            partial(qml.capture.eval_jaxpr, qfunc_jaxpr, temp_consts)
        )

        qfunc_jaxpr = jax.make_jaxpr(transformed_func)(*temp_args)
        temp_consts = qfunc_jaxpr.consts
        qfunc_jaxpr = qfunc_jaxpr.jaxpr

    # Expand user transforms applied to the qnode
    qfunc_jaxpr = qnode.transform_program(qfunc_jaxpr, temp_consts, *temp_args)
    temp_consts = qfunc_jaxpr.consts
    qfunc_jaxpr = qfunc_jaxpr.jaxpr

    # Apply device preprocessing transforms
    graph_enabled = qml.decomposition.enabled_graph()
    try:
        qml.decomposition.disable_graph()
        qfunc_jaxpr = device_program(qfunc_jaxpr, temp_consts, *temp_args)
    finally:
        if graph_enabled:
            qml.decomposition.enable_graph()
    consts = qfunc_jaxpr.consts
    qfunc_jaxpr = qfunc_jaxpr.jaxpr

    partial_eval = partial(
        device.eval_jaxpr,
        qfunc_jaxpr,
        consts,
        execution_config=execution_config,
        shots=Shots(shots),
    )
    if batch_dims is None:
        return partial_eval(*non_const_args)
    return jax.vmap(partial_eval, batch_dims[(n_consts + shots_len) :])(*non_const_args)


def custom_staging_rule(
    jaxpr_trace: pe.DynamicJaxprTrace, source_info, *tracers: pe.DynamicJaxprTracer, **params
) -> Sequence[pe.DynamicJaxprTracer] | pe.DynamicJaxprTracer:
    """
    Add new jaxpr equation to the jaxpr_trace and return new tracers.

    See capture/intro_to_dynamic_shapes.py for more context and capture.register_custom_staging_rule
    for the implementation used on other higher order primitives.
    """
    shots_len, jaxpr = params["shots_len"], params["qfunc_jaxpr"]
    device = params["device"]
    invars = [jaxpr_trace.getvar(x) for x in tracers]
    shots_vars = invars[:shots_len]

    batch_dims = params.get("batch_dims")
    split = params["n_consts"] + params["shots_len"]
    batch_shape = (
        _get_batch_shape(tracers[split:], batch_dims[split:]) if batch_dims is not None else ()
    )

    new_shapes = _get_shapes_for(
        *jaxpr.outvars,
        shots=shots_vars,
        num_device_wires=len(device.wires),
        batch_shape=batch_shape,
    )
    out_tracers = [pe.DynamicJaxprTracer(jaxpr_trace, o) for o in new_shapes]

    eqn = jax.core.new_jaxpr_eqn(
        invars,
        [jaxpr_trace.makevar(o) for o in out_tracers],
        qnode_prim,
        params,
        jax.core.no_effects,
        source_info=source_info,
    )

    jaxpr_trace.frame.add_eqn(eqn)
    return out_tracers


pe.custom_staging_rules[qnode_prim] = custom_staging_rule


# pylint: disable=too-many-arguments
def _qnode_batching_rule(
    batched_args,
    batch_dims,
    *,
    qnode,
    device,
    execution_config,
    qfunc_jaxpr,
    shots_len,
    n_consts,
):
    """
    Batching rule for the ``qnode`` primitive.

    This rule exploits the parameter broadcasting feature of the QNode to vectorize the circuit execution.
    """

    for idx, (arg, batch_dim) in enumerate(zip(batched_args, batch_dims, strict=True)):

        if _is_scalar_tensor(arg):
            continue

        # Regardless of their shape, jax.vmap automatically inserts `None` as the batch dimension for constants.
        # However, if the constant is not a standard JAX type, the batch dimension is not inserted at all.
        # How to handle this case is still an open question. For now, we raise a warning and give the user full flexibility.
        if idx < (n_consts + shots_len):
            warn(
                f"Constant argument at index {idx} is not scalar. "
                "This may lead to unintended behavior or wrong results if the argument is provided "
                "using parameter broadcasting to a quantum operation that supports batching.",
                UserWarning,
            )

        # To resolve this ambiguity, we might add more properties to the AbstractOperator
        # class to indicate which operators support batching and check them here.
        # As above, at this stage we raise a warning and give the user full flexibility.
        elif arg.size > 1 and batch_dim is None:
            warn(
                f"Argument at index {idx} has size > 1 but its batch dimension is None. "
                "This may lead to unintended behavior or wrong results if the argument is provided "
                "using parameter broadcasting to a quantum operation that supports batching.",
                UserWarning,
            )

    result = qnode_prim.bind(
        *batched_args,
        shots_len=shots_len,
        qnode=qnode,
        device=device,
        execution_config=execution_config,
        qfunc_jaxpr=qfunc_jaxpr,
        n_consts=n_consts,
        batch_dims=batch_dims,
    )

    # The batch dimension is at the front (axis 0) for all elements in the result.
    # JAX doesn't expose `out_axes` in the batching rule.
    return result, (0,) * len(result)


### JVP CALCULATION #########################################################
# This structure will change as we add more diff methods


@debug_logger
def _finite_diff(args, tangents, **impl_kwargs):
    if not jax.config.jax_enable_x64:
        warn(
            "Detected 32 bits precision with finite differences. This can lead to incorrect results."
            " Recommend enabling jax.config.update('jax_enable_x64', True).",
            UserWarning,
        )
    f = partial(qnode_prim.bind, **impl_kwargs)
    return qml.gradients.finite_diff_jvp(
        f, args, tangents, **impl_kwargs["execution_config"].gradient_keyword_arguments
    )


diff_method_map = {"finite-diff": _finite_diff}


@debug_logger
def _qnode_jvp(args, tangents, *, execution_config, device, qfunc_jaxpr, **impl_kwargs):
    execution_config = device.setup_execution_config(execution_config)
    if execution_config.use_device_gradient:
        return device.jaxpr_jvp(qfunc_jaxpr, args, tangents, execution_config=execution_config)

    if execution_config.gradient_method not in diff_method_map:
        raise NotImplementedError(
            f"diff_method {execution_config.gradient_method} not yet implemented."
        )

    return diff_method_map[execution_config.gradient_method](
        args,
        tangents,
        execution_config=execution_config,
        device=device,
        qfunc_jaxpr=qfunc_jaxpr,
        **impl_kwargs,
    )


### END JVP CALCULATION #######################################################

ad.primitive_jvps[qnode_prim] = _qnode_jvp

batching.primitive_batchers[qnode_prim] = _qnode_batching_rule

mlir.register_lowering(qnode_prim, mlir.lower_fun(qnode_prim.impl, multiple_results=True))


def _split_static_args(args, static_argnums):
    """Helper function to split a ``QNode``'s positional arguments into sequences
    of dynamic and static arguments respectively.

    Args:
        args (tuple): positional arguments of ``QNode``
        static_argnums (tuple[int]): indices for static arguments of the ``QNode``

    Returns:
        tuple, tuple: tuples containing the dynamic and static arguments, respectively.
    """
    if len(static_argnums) == 0:
        return args, ()

    dynamic_args, static_args = [], []
    static_argnums_iter = iter(static_argnums)
    static_argnum = next(static_argnums_iter)
    i = 0

    for i, arg in enumerate(args):
        if i == static_argnum:
            static_args.append(arg)
            static_argnum = next(static_argnums_iter, None)
        else:
            dynamic_args.append(arg)

        if static_argnum is None:
            break

    if i < len(args) - 1:
        dynamic_args.extend(args[i + 1 :])

    return tuple(dynamic_args), tuple(static_args)


def _get_jaxpr_cache_key(dynamic_args, static_args, kwargs, abstracted_axes):
    """Create a hash using the arguments and keyword arguments of a QNode.

    The hash is dependent on the abstract evaluation of ``dynamic_args``. For any indices
    in ``static_args``, the concrete value of the argument will be used to create
    the hash. If any arguments have dynamic shapes, their abstract axes will be replaced
    by the respective letter provided in ``abstracted_axes``.

    For keyword arguments, the string representation of the keyword argument
    dictionary will be used to create the hash.

    Args:
        dynamic_args (tuple): dynamic positional arguments of the cached qfunc
        static_args (tuple): static positional arguments of the cached qfunc
        kwargs (dict): keyword arguments of the cached qfunc
        abstract_axes (Optional[tuple[dict[int, str]]]): corresponding abstract axes
            of positional arguments

    Returns:
        int: hash to be used as the jaxpr cache's key
    """
    serialized = "args="

    for i, arg in enumerate(dynamic_args):
        if abstracted_axes:
            serialized_shape = tuple(
                abstracted_axes[i].get(j, s) for j, s in enumerate(qml.math.shape(arg))
            )
        else:
            serialized_shape = qml.math.shape(arg)
        serialized += f"{serialized_shape},{qml.math.get_dtype_name(arg)};"

    for arg in static_args:
        serialized += f"{arg};"

    serialized += f";;{kwargs=}"
    return hash(serialized)


def _extract_qfunc_jaxpr(qnode, abstracted_axes, *args, **kwargs):
    """Process the quantum function of a QNode to create a Jaxpr."""

    qfunc = partial(qnode.func, **kwargs) if kwargs else qnode.func
    flat_fn = FlatFn(qfunc)

    try:
        qfunc_jaxpr = jax.make_jaxpr(
            flat_fn, abstracted_axes=abstracted_axes, static_argnums=qnode.static_argnums
        )(*args)
    except (
        jax.errors.TracerArrayConversionError,
        jax.errors.TracerIntegerConversionError,
        jax.errors.TracerBoolConversionError,
    ) as exc:
        raise CaptureError(
            "Autograph must be used when Python control flow is dependent on a dynamic "
            "variable (a function input). Please ensure that autograph is being correctly enabled with "
            "`qml.capture.run_autograph` or disabled with `qml.capture.disable_autograph` or consider using PennyLane native control "
            "flow functions like `qml.for_loop`, `qml.while_loop`, or `qml.cond`."
        ) from exc

    assert flat_fn.out_tree is not None, "out_tree should be set by call to flat_fn"
    return qfunc_jaxpr, flat_fn.out_tree


def capture_qnode(qnode: "qml.QNode", *args, **kwargs) -> "qml.typing.Result":
    """A capture compatible call to a QNode. This function is internally used by ``QNode.__call__``.

    Args:
        qnode (QNode): a QNode
        args: the arguments the QNode is called with

    Keyword Args:
        kwargs (Any): Any keyword arguments accepted by the quantum function

    Returns:
        qml.typing.Result: the result of a qnode execution

    **Example:**

    .. code-block:: python

        qml.capture.enable()

        @qml.qnode(qml.device('lightning.qubit', wires=1))
        def circuit(x):
            qml.RX(x, wires=0)
            return qml.expval(qml.Z(0)), qml.probs()

        def f(x):
            expval_z, probs = circuit(np.pi * x, shots=50000)
            return 2 * expval_z + probs

        jaxpr = jax.make_jaxpr(f)(0.1)
        print("jaxpr:")
        print(jaxpr)

        res = jax.core.eval_jaxpr(jaxpr.jaxpr, jaxpr.consts, 0.7)
        print()
        print("result:")
        print(res)


    .. code-block:: none

        jaxpr:
        { lambda ; a:f32[]. let
            b:f32[] = mul 3.141592653589793 a
            c:f32[] d:f32[2] = qnode[
              device=<lightning.qubit device (wires=1) at 0x10557a070>
              qfunc_jaxpr={ lambda ; e:f32[]. let
                  _:AbstractOperator() = RX[n_wires=1] e 0
                  f:AbstractOperator() = PauliZ[n_wires=1] 0
                  g:AbstractMeasurement(n_wires=None) = expval_obs f
                  h:AbstractMeasurement(n_wires=0) = probs_wires
                in (g, h) }
              qnode=<QNode: device='<lightning.qubit device (wires=1) at 0x10557a070>', interface='auto', diff_method='best'>
              qnode_kwargs={'diff_method': 'best', 'grad_on_execution': 'best', 'cache': False, 'cachesize': 10000, 'max_diff': 1, 'device_vjp': False, 'mcm_method': None, 'postselect_mode': None}
              shots=Shots(total=50000)
            ] b
            i:f32[] = mul 2.0 c
            j:f32[2] = add i d
          in (j,) }

        result:
        [Array([-0.96939224, -0.38207346], dtype=float32)]


    """
    # apply transform to a callable so will be captured when called
    qnode_func = partial(_bind_qnode, qnode)
    for t in qnode.transform_program:
        qnode_func = t(qnode_func)

    return qnode_func(*args, **kwargs)


def _bind_qnode(qnode, *args, **kwargs):
    if qnode.device.wires is None:
        raise NotImplementedError(
            "devices must specify wires for integration with program capture."
        )

    # We compute ``abstracted_axes`` using the flattened arguments because trying to flatten
    # pytree ``abstracted_axes`` causes the abstract axis dictionaries to get flattened, which
    # we don't want to correctly compute the ``cache_key``.
    dynamic_args, static_args = _split_static_args(args, qnode.static_argnums)
    flat_dynamic_args, dynamic_args_struct = jax.tree_util.tree_flatten(dynamic_args)
    flat_static_args = jax.tree_util.tree_leaves(static_args)
    abstracted_axes, abstract_shapes = qml.capture.determine_abstracted_axes(flat_dynamic_args)
    cache_key = _get_jaxpr_cache_key(flat_dynamic_args, flat_static_args, kwargs, abstracted_axes)

    if cached_value := qnode.capture_cache.get(cache_key, None):
        qfunc_jaxpr, config, out_tree = cached_value
    else:
        config = construct_execution_config(
            qnode, resolve=False
        )()  # no need for args and kwargs as not resolving

        if abstracted_axes:
            # We unflatten the ``abstracted_axes`` here to be have the same pytree structure
            # as the original dynamic arguments
            abstracted_axes = jax.tree_util.tree_unflatten(dynamic_args_struct, abstracted_axes)

        qfunc_jaxpr, out_tree = _extract_qfunc_jaxpr(qnode, abstracted_axes, *args, **kwargs)

        qnode.capture_cache[cache_key] = (qfunc_jaxpr, config, out_tree)

    flat_shots = tuple(qnode._shots) if qnode._shots else ()  # pylint: disable=protected-access

    res = qnode_prim.bind(
        *flat_shots,
        *qfunc_jaxpr.consts,
        *abstract_shapes,
        *flat_dynamic_args,
        shots_len=len(flat_shots),
        qnode=qnode,
        device=qnode.device,
        execution_config=config,
        qfunc_jaxpr=qfunc_jaxpr.jaxpr,
        n_consts=len(qfunc_jaxpr.consts),
    )

    if len(flat_shots) > 1:
        shots_struct = jax.tree_util.tree_structure(flat_shots)
        out_tree = shots_struct.compose(out_tree)
    return jax.tree_util.tree_unflatten(out_tree, res)
