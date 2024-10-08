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

from copy import copy
from dataclasses import asdict
from functools import lru_cache, partial

import pennylane as qml

from .flatfn import FlatFn

has_jax = True
try:
    import jax
    from jax.interpreters import ad

except ImportError:
    has_jax = False


def _get_shapes_for(*measurements, shots=None, num_device_wires=0):
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
            shapes.append(jax.core.ShapedArray(shape, dtype_map.get(dtype, dtype)))
    return shapes


@lru_cache()
def _get_qnode_prim():
    if not has_jax:
        return None
    qnode_prim = jax.core.Primitive("qnode")
    qnode_prim.multiple_results = True

    # pylint: disable=too-many-arguments
    @qnode_prim.def_impl
    def _(*args, qnode, shots, device, qnode_kwargs, qfunc_jaxpr, n_consts):
        consts = args[:n_consts]
        args = args[n_consts:]

        def qfunc(*inner_args):
            return jax.core.eval_jaxpr(qfunc_jaxpr, consts, *inner_args)

        qnode = qml.QNode(qfunc, device, **qnode_kwargs)
        return qnode._impl_call(*args, shots=shots)  # pylint: disable=protected-access

    # pylint: disable=unused-argument
    @qnode_prim.def_abstract_eval
    def _(*args, qnode, shots, device, qnode_kwargs, qfunc_jaxpr, n_consts):
        mps = qfunc_jaxpr.outvars
        return _get_shapes_for(*mps, shots=shots, num_device_wires=len(device.wires))

    def make_zero(tan, arg):
        return jax.lax.zeros_like_array(arg) if isinstance(tan, ad.Zero) else tan

    def _qnode_jvp(args, tangents, **impl_kwargs):
        tangents = tuple(map(make_zero, tangents, args))
        return jax.jvp(partial(qnode_prim.impl, **impl_kwargs), args, tangents)

    ad.primitive_jvps[qnode_prim] = _qnode_jvp

    return qnode_prim


def qnode_call(qnode: "qml.QNode", *args, **kwargs) -> "qml.typing.Result":
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
