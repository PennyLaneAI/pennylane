# Copyright 2018-2021 Xanadu Quantum Technologies Inc.

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
This subpackage defines functions for interfacing devices with batch execution
capabilities with different machine learning libraries.
"""
# pylint: disable=import-outside-toplevel)
import functools

from .unwrap import UnwrapTape
from .autograd import batch_execute as batch_execute_autograd


@functools.wraps(batch_execute_autograd)
def batch_execute(tapes, device, gradient_fn=None, interface="autograd", **kwargs):
    """Execute a batch of tapes with NumPy parameters on a device.
    This function is a wrapper that dispatches to the correct interface."""
    import pennylane as qml

    unsupported_op = lambda op: op.grad_recipe is None
    supported_op = lambda op: op.grad_recipe is not None
    trainable_op = lambda op: any(qml.math.requires_grad(p) for p in op.parameters)

    for idx, t in enumerate(tapes):
        if any(unsupported_op(op) and trainable_op(op) for op in t.operations):
            tapes[idx] = t.expand(
                depth=10,
                stop_at=lambda obj: not isinstance(obj, qml.measure.MeasurementProcess)
                and ((supported_op(obj) and trainable_op(obj)) or not trainable_op(obj)),
            )

    else:
        c_jac = None

    if interface == "autograd":
        return batch_execute_autograd(tapes, device, gradient_fn=gradient_fn, **kwargs)

    if interface in ["tf", "tensorflow"]:
        from .tf import batch_execute as batch_execute_tf

        return batch_execute_tf(tapes, device, gradient_fn=gradient_fn, **kwargs)

    if interface in ["torch"]:
        from .torch import batch_execute as batch_execute_torch

        return batch_execute_torch(tapes, device, gradient_fn=gradient_fn, **kwargs)

    raise ValueError(f"Unknown interface {interface}")
