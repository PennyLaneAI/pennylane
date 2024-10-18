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
"""Contains a function for getting the best differentiation method for a given QNode.

"""

from functools import wraps

import pennylane as qml
from pennylane.workflow.qnode import _make_execution_config


# pylint: disable=unused-argument
def get_best_diff_method(qnode):
    """Returns a function that computes the 'best' differentiation method
    for a particular QNode.

    This method attempts to determine support for differentiation
    methods using the following order:

    * ``"device"``
    * ``"backprop"``
    * ``"parameter-shift"``
    * ``"finite-diff"``

    The first differentiation method that is supported (going from
    top to bottom) will be returned. Note that the SPSA-based and Hadamard-based gradients
    are not included here.

    Args:
        qnode (.QNode): the qnode to get the 'best' differentiation method for.
        return_as_str (bool): return the 'best' differentiation method in human-readable format.

    Returns:
        str or .TransformDispatcher (a.k.a the ``gradient_fn``)
    """

    def handle_return(transform, return_as_str):
        """Helper function to manage the return type"""
        if return_as_str:
            if transform is qml.gradients.finite_diff:
                return "finite-diff"
            if transform in (qml.gradients.param_shift, qml.gradients.param_shift_cv):
                return "parameter-shift"
        return transform

    @wraps(qnode)
    def wrapper(*args, return_as_str=False, **kwargs):
        device = qnode.device
        (tape,), _ = qml.workflow.construct_batch(qnode)(*args, **kwargs)

        config = _make_execution_config(None, "best")

        if device.supports_derivatives(config, circuit=tape):
            new_config = device.preprocess(config)[1]
            transform = new_config.gradient_method
            return handle_return(transform, return_as_str)

        if tape and any(isinstance(o, qml.operation.CV) for o in tape):
            transform = qml.gradients.param_shift_cv
            return handle_return(transform, return_as_str)

        transform = qml.gradients.param_shift
        return handle_return(transform, return_as_str)

    return wrapper
