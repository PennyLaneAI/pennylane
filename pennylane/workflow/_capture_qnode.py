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

**Overridden shots:** Device execution currently pulls the shot information from the device. In order
to support dynamic shots, we need to develop an additional protocol for communicating the shot information
associated with a circuit. Dynamically mutating objects is not compatible with jaxpr and jitting.

**Shot vectors**.  Shot vectors are not yet supported. We need to figure out how to stack
and reshape the outputs from measurements on the device when multiple measurements are present.

**Gradients other than default qubit backprop**. We managed to get backprop of default qubit for
free, but no other gradients methods have support yet.

**MCM methods other than single branch statistics**. Mid circuit measurements
are only handled via a "single branch statistics" algorithm, which will lead to unexpected
results. Even on analytic devices, one branch will be randomly chosen on each execution.
Returning measurements based on mid circuit measurements, ``qml.sample(m0)``,
is also not yet supported on default qubit or lightning.

>>> @qml.qnode(qml.device('default.qubit', wires=1))
>>> def circuit(x):
...     qml.H(0)
...     m0 = qml.measure(0)
...     qml.cond(m0, qml.RX, qml.RZ)(x,0)
...     return qml.expval(qml.Z(0))
>>> circuit(0.5), circuit(0.5), circuit(0.5)
(Array(-0.87758256, dtype=float64),
Array(1., dtype=float64),
Array(-0.87758256, dtype=float64))
>>> qml.capture.disable()
>>> circuit(0.5)
np.float64(0.06120871905481362)
>>> qml.capture.enable()

**Device preprocessing and validation**. No device preprocessing and validation will occur. The captured
jaxpr is directly sent to the device, whether or not the device can handle it.

>>> @qml.qnode(qml.device('default.qubit', wires=3))
... def circuit():
...     qml.Permute(jax.numpy.array((0,1,2)), wires=(2,1,0))
...     return qml.state()
>>> circuit()
MatrixUndefinedError:

**Transforms are still under development**. No transforms will currently be applied as part of the workflow.

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
measurment is fully independent and generated from different raw samples than every other measurement.
To generate multiple measurments from the same samples, we need a way of denoting which measurements
should be taken together. A "Combination measurement process" higher order primitive, or something like it.
We will also need to figure out how to implement splitting up a circuit with non-commuting measurements into
multiple circuits.

>>> @qml.qnode(qml.device('default.qubit', wires=1, shots=5))
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
from copy import copy
from dataclasses import asdict
from functools import lru_cache, partial
from numbers import Number
from warnings import warn

import pennylane as qml
from pennylane.capture import FlatFn
from pennylane.typing import TensorLike

has_jax = True
try:
    import jax
    from jax.interpreters import ad, batching, mlir
except ImportError:
    has_jax = False


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
        for arg, batch_dim in zip(non_const_args, non_const_batch_dims)
        if batch_dim is not None
    ]

    return jax.lax.broadcast_shapes(*input_shapes)


def _get_shapes_for(*measurements, shots=None, num_device_wires=0, batch_shape=()):
    """Calculate the abstract output shapes for the given measurements."""

    if jax.config.jax_enable_x64:  # pylint: disable=no-member
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
            shape, dtype = m.aval.abstract_eval(shots=s, num_device_wires=num_device_wires)
            shapes.append(jax.core.ShapedArray(batch_shape + shape, dtype_map.get(dtype, dtype)))

    return shapes


@lru_cache()
def _get_qnode_prim():
    if not has_jax:
        return None
    qnode_prim = jax.core.Primitive("qnode")
    qnode_prim.multiple_results = True

    # pylint: disable=too-many-arguments, unused-argument
    @qnode_prim.def_impl
    def qnode_impl(
        *args, qnode, shots, device, qnode_kwargs, qfunc_jaxpr, n_consts, batch_dims=None
    ):
        if shots != device.shots:
            raise NotImplementedError(
                "Overriding shots is not yet supported with the program capture execution."
            )
        if qnode_kwargs["diff_method"] not in {"backprop", "best"}:
            raise NotImplementedError(
                "Only backpropagation derivatives are supported at this time."
            )

        consts = args[:n_consts]
        non_const_args = args[n_consts:]

        if batch_dims is None:
            return device.eval_jaxpr(qfunc_jaxpr, consts, *non_const_args)
        return jax.vmap(partial(device.eval_jaxpr, qfunc_jaxpr, consts), batch_dims[n_consts:])(
            *non_const_args
        )

    # pylint: disable=unused-argument
    @qnode_prim.def_abstract_eval
    def _(*args, qnode, shots, device, qnode_kwargs, qfunc_jaxpr, n_consts, batch_dims=None):

        mps = qfunc_jaxpr.outvars

        batch_shape = (
            _get_batch_shape(args[n_consts:], batch_dims[n_consts:])
            if batch_dims is not None
            else ()
        )

        return _get_shapes_for(
            *mps, shots=shots, num_device_wires=len(device.wires), batch_shape=batch_shape
        )

    # pylint: disable=too-many-arguments
    def _qnode_batching_rule(
        batched_args,
        batch_dims,
        *,
        qnode,
        shots,
        device,
        qnode_kwargs,
        qfunc_jaxpr,
        n_consts,
    ):
        """
        Batching rule for the ``qnode`` primitive.

        This rule exploits the parameter broadcasting feature of the QNode to vectorize the circuit execution.
        """

        for idx, (arg, batch_dim) in enumerate(zip(batched_args, batch_dims)):

            if _is_scalar_tensor(arg):
                continue

            # Regardless of their shape, jax.vmap automatically inserts `None` as the batch dimension for constants.
            # However, if the constant is not a standard JAX type, the batch dimension is not inserted at all.
            # How to handle this case is still an open question. For now, we raise a warning and give the user full flexibility.
            if idx < n_consts:
                warn(
                    f"Constant argument at index {idx} is not scalar. "
                    "This may lead to unintended behavior or wrong results if the argument is provided "
                    "using parameter broadcasting to a quantum operation that supports batching.",
                    UserWarning,
                )

            else:

                # To resolve this ambiguity, we might add more properties to the AbstractOperator
                # class to indicate which operators support batching and check them here.
                # As above, at this stage we raise a warning and give the user full flexibility.
                if arg.size > 1 and batch_dim is None:
                    warn(
                        f"Argument at index {idx} has size > 1 but its batch dimension is None. "
                        "This may lead to unintended behavior or wrong results if the argument is provided "
                        "using parameter broadcasting to a quantum operation that supports batching.",
                        UserWarning,
                    )

        result = qnode_prim.bind(
            *batched_args,
            shots=shots,
            qnode=qnode,
            device=device,
            qnode_kwargs=qnode_kwargs,
            qfunc_jaxpr=qfunc_jaxpr,
            n_consts=n_consts,
            batch_dims=batch_dims,
        )

        # The batch dimension is at the front (axis 0) for all elements in the result.
        # JAX doesn't expose `out_axes` in the batching rule.
        return result, (0,) * len(result)

    def make_zero(tan, arg):
        return jax.lax.zeros_like_array(arg) if isinstance(tan, ad.Zero) else tan

    def _qnode_jvp(args, tangents, **impl_kwargs):
        tangents = tuple(map(make_zero, tangents, args))
        return jax.jvp(partial(qnode_prim.impl, **impl_kwargs), args, tangents)

    ad.primitive_jvps[qnode_prim] = _qnode_jvp

    batching.primitive_batchers[qnode_prim] = _qnode_batching_rule

    mlir.register_lowering(qnode_prim, mlir.lower_fun(qnode_impl, multiple_results=True))

    return qnode_prim


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

    if "shots" in kwargs:
        shots = qml.measurements.Shots(kwargs.pop("shots"))
    else:
        shots = qnode.device.shots

    if shots.has_partitioned_shots:
        # Questions over the pytrees and the nested result object shape
        raise NotImplementedError("shot vectors are not yet supported with plxpr capture.")

    if not qnode.device.wires:
        raise NotImplementedError("devices must specify wires for integration with plxpr capture.")

    qfunc = partial(qnode.func, **kwargs) if kwargs else qnode.func
    flat_fn = FlatFn(qfunc)
    qfunc_jaxpr = jax.make_jaxpr(flat_fn)(*args)

    execute_kwargs = copy(qnode.execute_kwargs)
    mcm_config = asdict(execute_kwargs.pop("mcm_config"))
    qnode_kwargs = {"diff_method": qnode.diff_method, **execute_kwargs, **mcm_config}
    qnode_prim = _get_qnode_prim()

    flat_args = jax.tree_util.tree_leaves(args)

    res = qnode_prim.bind(
        *qfunc_jaxpr.consts,
        *flat_args,
        shots=shots,
        qnode=qnode,
        device=qnode.device,
        qnode_kwargs=qnode_kwargs,
        qfunc_jaxpr=qfunc_jaxpr.jaxpr,
        n_consts=len(qfunc_jaxpr.consts),
    )
    assert flat_fn.out_tree is not None, "out_tree should be set by call to flat_fn"
    return jax.tree_util.tree_unflatten(flat_fn.out_tree, res)
