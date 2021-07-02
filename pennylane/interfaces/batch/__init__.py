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
def batch_execute(*args, interface="autograd", **kwargs):
    """Execute a batch of tapes with NumPy parameters on a device.
    This function is a wrapper that dispatches to the correct interface."""

    if interface == "autograd":
        return batch_execute_autograd(*args, **kwargs)

    if interface in ["tf", "tensorflow"]:
        from .tf import batch_execute as batch_execute_tf

        return batch_execute_tf(*args, **kwargs)

    if interface in ["torch"]:
        from .torch import batch_execute as batch_execute_torch

        return batch_execute_torch(*args, **kwargs)

    raise ValueError(f"Unknown interface {interface}")
