# Copyright 2026 Xanadu Quantum Technologies Inc.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The device whose quantum function is currently being traced."""

import contextvars
from contextlib import contextmanager

_TRACING_DEVICE = contextvars.ContextVar("tracing_device", default=None)


@contextmanager
def tracing_device(device):
    """Publish ``device`` as the one whose quantum function is being traced, for the duration.

    Args:
        device: The device the QNode being traced was constructed with.
    """
    token = _TRACING_DEVICE.set(device)
    try:
        yield device
    finally:
        _TRACING_DEVICE.reset(token)


def get_tracing_device():
    """The device whose quantum function is being traced.

    Returns:
        The device, or ``None`` outside a quantum function trace.
    """
    return _TRACING_DEVICE.get()
