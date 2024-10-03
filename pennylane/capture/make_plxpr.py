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

from typing import Sequence, Union

has_jax = True
try:
    import jax
except ImportError:
    has_jax = False


def make_plxpr(
    circuit: "qml.QNode", static_argnums: Union[int, Sequence[int], None] = None, **kwargs
):
    """Takes a QNode and returns a function that, when called, produces a PLxPR representing
    the QNode with the given args.

    This function relies on `jax.make_jaxpr` as part of creating the representation. Any
    keyword arguments passed to `make_plxpr` that are not directly used in the function will
    be passed to make_jaxpr.

    Args:
        circuit (QNode):  the QNode to be captured
        static_argnums (Union(int, Sequence[int], None)): optional, an int or collection of ints
            that specify which positional arguments to treat as static (trace- and compile-time constant).

    Kwargs:
        **kwargs: any additional arguements specifically for jax.make_jaxpr

    Returns:
        Callable: function that, when called, returns the PLxPR representation of `circuit` for the specified inputs.


    ** Example **

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

    """
    return jax.make_jaxpr(circuit, static_argnums, **kwargs)
