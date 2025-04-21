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
"""Contains a function for getting the best differentiation method for a given QNode."""

from functools import wraps

import pennylane as qml
from pennylane.workflow.qnode import QNode, _make_execution_config


def get_best_diff_method(qnode: QNode):
    """Returns a function that computes the 'best' differentiation method
    for a particular QNode.

    This method prioritizes differentiation methods in the following order (SPSA-based and Hadamard-based gradients
    are not included here):

    * ``"device"``
    * ``"backprop"``
    * ``"parameter-shift"``

    .. note::

        The first differentiation method that is supported (from top to bottom)
        will be returned. The order is designed to maximize efficiency, generality,
        and stability.

    .. seealso::

        For a detailed comparison of the backpropagation and parameter-shift methods,
        refer to the :doc:`quantum gradients with backpropagation example <demo:demos/tutorial_backprop>`.

    Args:
        qnode (.QNode): the qnode to get the 'best' differentiation method for.

    Returns:
        str: the gradient transform.
    """

    def handle_return(transform):
        """Helper function to manage the return"""
        if transform in (qml.gradients.param_shift, qml.gradients.param_shift_cv):
            return "parameter-shift"
        return transform

    @wraps(qnode)
    def wrapper(*args, **kwargs):
        device = qnode.device
        tape = qml.workflow.construct_tape(qnode)(*args, **kwargs)

        config = _make_execution_config(None, "best")

        if device.supports_derivatives(config, circuit=tape):
            new_config = device.setup_execution_config(config)
            transform = new_config.gradient_method
            return handle_return(transform)

        if tape and any(isinstance(o, qml.operation.CV) for o in tape):
            transform = qml.gradients.param_shift_cv
            return handle_return(transform)

        transform = qml.gradients.param_shift
        return handle_return(transform)

    return wrapper
