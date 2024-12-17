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
Helper function for expanding transforms with program capture
"""
from functools import wraps
from typing import Callable

from .base_interpreter import PlxprInterpreter


class ExpandTransformsInterpreter(PlxprInterpreter):
    """Interpreter for expanding transform primitives that are applied to plxpr.

    This interpreter does not do anything special by itself. Instead, it is used
    by the PennyLane transforms to expand transform primitives in plxpr by
    applying the respective transform to the inner plxpr. When a transform is created
    using :func:`~pennylane.transform`, a custom primitive interpretation rule for
    that transform is automatically registered for ``ExpandTransformsInterpreter``.
    """


def expand_plxpr_transforms(f: Callable) -> Callable:
    """Function for expanding plxpr transforms.

    This function wraps the input callable where transforms may be present, but not yet applied.
    The returned wrapper expands all transforms that may be present on the input callable.

    **Example**

    .. code-block:: python

        from functools import partial
        import jax

        qml.capture.enable()
        wire_map = {0: 3, 1: 6, 2: 9}

        @partial(qml.map_wires, wire_map=wire_map)
        def circuit(x, y):
            qml.RX(x, 0)
            qml.CNOT([0, 1])
            qml.CRY(y, [1, 2])
            return qml.expval(qml.Z(2))

    >>> jax.make_jaxpr(circuit)(1.2, 3.4)
    { lambda ; a:f32[] b:f32[]. let
        c:AbstractMeasurement(n_wires=None) = _map_wires_transform_transform[
        args_slice=slice(0, 2, None)
        consts_slice=slice(2, 2, None)
        inner_jaxpr={ lambda ; d:f32[] e:f32[]. let
            _:AbstractOperator() = RX[n_wires=1] d 0
            _:AbstractOperator() = CNOT[n_wires=2] 0 1
            _:AbstractOperator() = CRY[n_wires=2] e 1 2
            f:AbstractOperator() = PauliZ[n_wires=1] 2
            g:AbstractMeasurement(n_wires=None) = expval_obs f
            in (g,) }
        targs_slice=slice(2, None, None)
        tkwargs={'wire_map': {0: 3, 1: 6, 2: 9}, 'queue': False}
        ] a b
    in (c,) }

    >>> transformed_circuit = qml.capture.expand_plxpr_transforms(circuit)
    >>> jax.make_jaxpr(transformed_circuit)(1.2, 3.4)
    { lambda ; a:f32[] b:f32[]. let
        _:AbstractOperator() = RX[n_wires=1] a 3
        _:AbstractOperator() = CNOT[n_wires=2] 3 6
        _:AbstractOperator() = CRY[n_wires=2] b 6 9
        c:AbstractOperator() = PauliZ[n_wires=1] 9
        d:AbstractMeasurement(n_wires=None) = expval_obs c
    in (d,) }

    Args:
        f (Callable): The callable for which we want to expand transforms.

    Returns:
        Callable: Callable with expanded transforms
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        transformed_f = ExpandTransformsInterpreter()(f)
        return transformed_f(*args, **kwargs)

    return wrapper
