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
from collections.abc import Callable
from functools import wraps

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
    """Function for applying transforms to plxpr.

    Currently, when program capture is enabled, transforms are used as higher-order primitives.
    These primitives are present in the program, but their respective transform is not applied
    when a transformed function is called. ``expand_plxpr_transforms`` further "transforms" the
    input function to apply any transform primitives that are present in the program being run.

    **Example**

    In the below example, we can see that the ``qml.transforms.cancel_inverses`` transform has been
    applied to a function. However, the resulting program representation leaves the
    ``cancel_inverses`` transform as a primitive without actually transforming the program.

    .. code-block:: python

        qml.capture.enable()

        @qml.transforms.cancel_inverses
        def circuit():
            qml.X(0)
            qml.S(1)
            qml.X(0)
            qml.adjoint(qml.S(1))
            return qml.expval(qml.Z(1))

    >>> qml.capture.make_plxpr(circuit)()
    { lambda ; . let
        a:AbstractMeasurement(n_wires=None) = cancel_inverses_transform[
        args_slice=slice(0, 0, None)
        consts_slice=slice(0, 0, None)
        inner_jaxpr={ lambda ; . let
            _:AbstractOperator() = PauliX[n_wires=1] 0
            _:AbstractOperator() = S[n_wires=1] 1
            _:AbstractOperator() = PauliX[n_wires=1] 0
            b:AbstractOperator() = S[n_wires=1] 1
            _:AbstractOperator() = Adjoint b
            c:AbstractOperator() = PauliZ[n_wires=1] 1
            d:AbstractMeasurement(n_wires=None) = expval_obs c
          in (d,) }
        targs_slice=slice(0, None, None)
        tkwargs={}
        ]
      in (a,) }

    To apply the transform, we can use ``expand_plxpr_transforms`` as follows:

    >>> transformed_circuit = qml.capture.expand_plxpr_transforms(circuit)
    >>> qml.capture.make_plxpr(transformed_circuit)()
    { lambda ; . let
        a:AbstractOperator() = PauliZ[n_wires=1] 1
        b:AbstractMeasurement(n_wires=None) = expval_obs a
      in (b,) }

    As seen, the transform primitive is no longer present, but it has been applied
    to the original program, indicated by the inverse operators being cancelled.

    Args:
        f (Callable): The callable to which any present transforms should be applied.

    Returns:
        Callable: Callable with transforms applied.
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        transformed_f = ExpandTransformsInterpreter()(f)
        return transformed_f(*args, **kwargs)

    return wrapper
