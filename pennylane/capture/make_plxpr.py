# Copyright 2018-2024 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The make_plxpr function and helper methods"""

from collections.abc import Callable, Sequence

import pennylane as qml

from .autograph import run_autograph

has_jax = True
try:
    import jax
except ImportError:  # pragma: no cover
    has_jax = False


def make_plxpr(func: Callable, static_argnums: int | Sequence[int] = (), autograph=True, **kwargs):
    r"""Takes a function and returns a ``Callable`` that, when called, produces a PLxPR representing
    the function with the given args.

    This function relies on ``jax.make_jaxpr`` as part of creating the representation. Any
    keyword arguments passed to ``make_plxpr`` that are not directly used in the function will
    be passed to ``make_jaxpr``.

    Args:
        func (Callable): the ``Callable`` to be captured

    Keyword Args:
        static_argnums (Union(int, Sequence[int])): optional, an ``int`` or collection of ``int``\ s
            that specify which positional arguments to treat as static (trace- and compile-time constant).
        autograph (bool): whether to use AutoGraph to convert Python control flow to native PennyLane
            control flow. Defaults to True.

    Returns:
        Callable: function that, when called, returns the PLxPR representation of ``func`` for the specified inputs.

    .. note::

        More details on using AutoGraph are provided under Usage Details.

        There are some limitations and sharp bits regarding AutoGraph; to better understand
        supported behaviour and limitations, see https://docs.pennylane.ai/en/stable/development/autograph.html

    **Example**

    .. code-block:: python

        qml.capture.enable()

        dev = qml.device("default.qubit", wires=1)

        @qml.qnode(dev)
        def circ(x):
            qml.RX(x, 0)
            qml.Hadamard(0)
            return qml.expval(qml.X(0))

        plxpr = qml.capture.make_plxpr(circ)(1.2)


    >>> print(plxpr)
    { lambda ; a:f32[]. let
        b:f32[] = qnode[
            device=<default.qubit device (wires=1) at 0x152a6f010>
            n_consts=0
            qfunc_jaxpr={ lambda ; c:f32[]. let
                _:AbstractOperator() = RX[n_wires=1] c 0
                _:AbstractOperator() = Hadamard[n_wires=1] 0
                d:AbstractOperator() = PauliX[n_wires=1] 0
                e:AbstractMeasurement(n_wires=None) = expval_obs d
              in (e,) }
            qnode=<QNode: device='<default.qubit device (wires=1) at 0x152a6f010>', interface='auto', diff_method='best'>
            qnode_kwargs={'diff_method': 'best', 'grad_on_execution': 'best', 'cache': False, 'cachesize': 10000, 'max_diff': 1, 'device_vjp': False, 'mcm_method': None, 'postselect_mode': None}
            shots=Shots(total=None)
        ] a
      in (b,) }

    .. details ::
        :title: Usage Details

        The ``autograph`` argument is ``True`` by default, converting Pythonic control flow to PennyLane
        supported control flow. This requires the ``diastatic-malt`` package, a standalone fork of the AutoGraph
        module in TensorFlow (`official documentation <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/autograph/g3doc/reference/index.md>`_
        ).

        .. note::

            There are some limitations and sharp bits regarding AutoGraph; to better understand
            supported behaviour and limitations, see https://docs.pennylane.ai/en/stable/development/autograph.html

        On its own, capture of standard Python control flow is not supported:

        .. code-block:: python

            def fn(x):
                if x > 5:
                    return x+1
                return x+2

        For this function, capture doesn't work without autograph:

        >>> plxpr_fn = qml.capture.make_plxpr(fn, autograph=False)
        >>> plxpr = plxpr_fn(3)
        TracerBoolConversionError: Attempted boolean conversion of traced array with shape bool[].

        With AutoGraph, the control flow is automatically converted to the native PennyLane control
        flow implementation, and succeeds:

        >>> plxpr_fn = qml.capture.make_plxpr(fn)
        >>> plxpr = plxpr_fn(3)
        >>> plxpr
        { lambda ; a:i64[]. let
            b:bool[] = gt a 5
            _:bool[] c:i64[] = cond[
              args_slice=slice(4, None, None)
              consts_slices=[slice(2, 3, None), slice(3, 4, None)]
              jaxpr_branches=[{ lambda a:i64[]; . let  in (True, a) }, { lambda a:i64[]; . let b:i64[] = add a 2 in (True, b) }]
            ] b True a a
          in (c,) }

        We can evaluate this to get the results:

        >>> jax.core.eval_jaxpr(plxpr.jaxpr, plxpr.consts, 2)
        [Array(4, dtype=int64, weak_type=True)]

        >>> jax.core.eval_jaxpr(plxpr.jaxpr, plxpr.consts, 7)
        [Array(8, dtype=int64, weak_type=True)]
    """
    if not has_jax:  # pragma: no cover
        raise ImportError(
            "Module jax is required for the ``make_plxpr`` function. "
            "You can install jax via: pip install jax~=0.6.0"
        )

    if not qml.capture.enabled():
        raise RuntimeError(
            "Capturing PLxPR with ``make_plxpr`` requires PennyLane capture to be enabled. "
            "You can enable capture with ``qml.capture.enable()``"
        )

    if autograph:
        func = run_autograph(func)

    return jax.make_jaxpr(func, static_argnums=static_argnums, **kwargs)
