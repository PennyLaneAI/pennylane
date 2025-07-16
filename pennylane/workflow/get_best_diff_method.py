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

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING

import pennylane as qml
from pennylane.workflow.qnode import _make_execution_config
from pennylane.workflow.resolution import _resolve_execution_config

if TYPE_CHECKING:
    from pennylane.workflow.qnode import QNode


def get_best_diff_method(qnode: QNode):
    """Returns a function that computes the 'best' differentiation method
    for a particular QNode.

    This method follows the same resolution logic as the modern PennyLane execution workflow,
    prioritizing differentiation methods in the following order:

    1. **Device-provided methods**: If the device supports derivatives natively (e.g., ``"adjoint"``, ``"backprop"``)
    2. **Parameter-shift**: For hardware-compatible circuits and CV operations
    3. **Fallback transforms**: As determined by the resolution system

    The method uses the same resolution logic as :func:`~pennylane.workflow.execution.execute` 
    and :func:`~pennylane.workflow.construct_batch.construct_batch` to ensure consistency
    across the workflow.

    .. seealso::

        For a detailed comparison of the backpropagation and parameter-shift methods,
        refer to the `quantum gradients with backpropagation example <demo:demos/tutorial_backprop>`__.

    Args:
        qnode (.QNode): the qnode to get the 'best' differentiation method for.

    Returns:
        str: the gradient method name or transform.
    """

    def handle_return(transform):
        """Helper function to manage the return and normalize transform names"""
        if transform in (qml.gradients.param_shift, qml.gradients.param_shift_cv):
            return "parameter-shift"
        if hasattr(transform, '__name__'):
            # For transform functions, return their name
            return transform.__name__
        return transform

    @wraps(qnode)
    def wrapper(*args, **kwargs):
        device = qnode.device
        
        # Construct the tape using the same method as the execution workflow
        tape = qml.workflow.construct_tape(qnode)(*args, **kwargs)
        
        # Create execution config with "best" method - this matches the workflow behavior
        mcm_config = qml.devices.MCMConfig(
            postselect_mode=qnode.execute_kwargs.get("postselect_mode"),
            mcm_method=qnode.execute_kwargs.get("mcm_method"),
        )
        config = _make_execution_config(qnode, "best", mcm_config)
        
        # Use the same resolution logic as execute() and construct_batch()
        resolved_config = _resolve_execution_config(config, device, [tape])
        
        return handle_return(resolved_config.gradient_method)

    return wrapper
