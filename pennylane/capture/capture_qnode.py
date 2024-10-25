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
"""
import warnings
from copy import copy
from dataclasses import asdict
from functools import lru_cache, partial

import pennylane as qml
from pennylane.math.utils import is_non_scalar_tensor

from .flatfn import FlatFn

has_jax = True
try:
    import jax
    from jax.interpreters import ad, batching

except ImportError:
    has_jax = False


def _get_shapes_for(*measurements, shots=None, num_device_wires=0, batch_shape=()):
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
    def _(
        *args, qnode, shots, device, qnode_kwargs, qfunc_jaxpr, n_consts, in_axes=(), batch_shape=()
    ):

        print(f"qnode_def_impl")

        print(f"args: {args}")

        consts = args[:n_consts]
        args = args[n_consts:]

        def qfunc(*inner_args):
            return jax.core.eval_jaxpr(qfunc_jaxpr, consts, *inner_args)

        # Create a QNode with the given function, device, and additional kwargs
        qnode = qml.QNode(qfunc, device, **qnode_kwargs)

        if batch_shape != ():
            return jax.vmap(qnode._impl_call, in_axes=in_axes, out_axes=0)(
                *args
            )  # pylint: disable=protected-access

        return qnode._impl_call(*args, shots=shots)  # pylint: disable=protected-access

    # pylint: disable=unused-argument
    @qnode_prim.def_abstract_eval
    def _(*args, qnode, shots, device, qnode_kwargs, qfunc_jaxpr, n_consts, in_axes=(), batch_shape=()):

        print(f"qnode_abstract_eval")

        mps = qfunc_jaxpr.outvars

        return _get_shapes_for(
            *mps,
            shots=shots,
            num_device_wires=len(device.wires),
            batch_shape=batch_shape,
        )

    # pylint: disable=too-many-arguments, too-many-positional-arguments
    def _qnode_batching_rule(
        batched_args,
        batch_dims,
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

        print("batching rule called")

        print(f"batched_args received by batching rule: {batched_args}")
        print(f"batch_dims received by batching rule: {batch_dims}")

        assert len(batched_args) == len(batch_dims), "Mismatch in batched arguments and dimensions."
        assert all(
            batch_dim is None or isinstance(batch_dim, int) for batch_dim in batch_dims
        ), "Invalid batch dimensions found."

        args = batched_args[n_consts:]

        for i, (arg, batch_dim) in enumerate(zip(batched_args, batch_dims)):

            if not is_non_scalar_tensor(arg):
                continue

            if i < n_consts:
                raise ValueError("Batched constant cannot currently be captured with jax.vmap.")

            if arg.size == 0:
                raise ValueError("Empty tensors are not supported with jax.vmap.")

            # TODO: to fix this, we need to add more properties to the AbstractOperator
            # class to indicate which operators support batching and check them here
            if arg.size > 1 and batch_dim is None:
                warnings.warn(
                    f"Argument at index {i} has more than 1 element but is not batched. "
                    "This may lead to unintended behavior or wrong results if the argument is provided "
                    "using parameter broadcasting to a quantum operation that supports batching.",
                    UserWarning,
                )

        # TODO: this must be extended to the multidimensional case

        input_shapes = [arg.shape for arg in args if is_non_scalar_tensor(arg)]
        batch_shape = jax.lax.broadcast_shapes(*input_shapes)

        result = qnode_prim.bind(
            *batched_args,
            shots=shots,
            qnode=qnode,
            device=device,
            qnode_kwargs=qnode_kwargs,
            qfunc_jaxpr=qfunc_jaxpr,
            n_consts=n_consts,
            in_axes=batch_dims[n_consts:],
            batch_shape=batch_shape,
        )

        # The batch dimension is at the front (axis 0) for all elements in the result.
        return result, [0] * len(result)

    def make_zero(tan, arg):
        return jax.lax.zeros_like_array(arg) if isinstance(tan, ad.Zero) else tan

    def _qnode_jvp(args, tangents, **impl_kwargs):
        tangents = tuple(map(make_zero, tangents, args))
        return jax.jvp(partial(qnode_prim.impl, **impl_kwargs), args, tangents)

    ad.primitive_jvps[qnode_prim] = _qnode_jvp

    batching.primitive_batchers[qnode_prim] = _qnode_batching_rule

    return qnode_prim


def qnode_call(qnode: "qml.QNode", *args, **kwargs) -> "qml.typing.Result":

    print(f"qnode_call")

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
