# Copyright 2018 Xanadu Quantum Technologies Inc.

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
This module contains the :func:`qnode` decorator, which turns a quantum function into
a :class:`Qnode` object.
"""

# pylint: disable=redefined-outer-name
from functools import wraps, lru_cache

from .qnode import QNode


def qnode(device, interface='numpy', cache=False):
    """QNode decorator.

    Args:
        device (~pennylane._device.Device): a PennyLane-compatible device
        interface (str): the interface that will be used for automatic
            differentiation and classical processing. This affects
            the types of objects that can be passed to/returned from the QNode:

            * ``interface='numpy'``: The QNode accepts default Python types
              (floats, ints, lists) as well as NumPy array arguments,
              and returns NumPy arrays.

            * ``interface='torch'``: The QNode accepts and returns Torch tensors.

            * ``interface='tfe'``: The QNode accepts and returns eager execution
              TensorFlow ``tfe.Variable`` objects.

        cache (bool): If ``True``, the quantum function used to generate the QNode will
            only be called to construct the quantum circuit once, on first execution,
            and this circuit structure (i.e., the placement of templates, gates, measurements, etc.) will be cached for all further executions. The circuit parameters can still change with every call. Only activate this
            feature if your quantum circuit structure will never change.
    """
    @lru_cache()
    def qfunc_decorator(func):
        """The actual decorator"""

        qnode = QNode(func, device, cache=cache)

        if interface == 'torch':
            return qnode.to_torch()

        if interface == 'tf':
            return qnode.to_tf()

        @wraps(func)
        def wrapper(*args, **kwargs):
            """Wrapper function"""
            return qnode(*args, **kwargs)

        # bind the jacobian method to the wrapped function
        wrapper.jacobian = qnode.jacobian
        wrapper.metric_tensor = qnode.metric_tensor

        # bind the qnode attributes to the wrapped function
        wrapper.__dict__.update(qnode.__dict__)

        return wrapper
    return qfunc_decorator
