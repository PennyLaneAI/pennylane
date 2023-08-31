# Copyright 2023 Xanadu Quantum Technologies Inc.

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
A wrapper module for the PennyLane's native compilation mode via Catalyst.
"""

import functools

try:
    import catalyst

    pl_qjit_available = True
except ImportError:
    pl_qjit_available = False


def qjit(*args, **kwargs):
    if not pl_qjit_available:
        raise ImportError(
            "Catalyst is required for the QJIT-compilation mode, "
            "you can install it with `pip install pennylane-catalyst`"
        )

    return catalyst.qjit(*args, **kwargs)


def cond(*args, **kwargs):
    if not pl_qjit_available:
        raise ImportError(
            "Catalyst is required for the QJIT-compilation mode, "
            "you can install it with `pip install pennylane-catalyst`"
        )

    return catalyst.cond(*args, **kwargs)


def for_loop(*args, **kwargs):
    if not pl_qjit_available:
        raise ImportError(
            "Catalyst is required for the QJIT-compilation mode, "
            "you can install it with `pip install pennylane-catalyst`"
        )

    return catalyst.for_loop(*args, **kwargs)


def while_loop(*args, **kwargs):
    if not pl_qjit_available:
        raise ImportError(
            "Catalyst is required for the QJIT-compilation mode, "
            "you can install it with `pip install pennylane-catalyst`"
        )

    return catalyst.while_loop(*args, **kwargs)


if pl_qjit_available:
    qjit.__doc__ = catalyst.qjit.__doc__
    cond.__doc__ = catalyst.cond.__doc__
    for_loop.__doc__ = catalyst.for_loop.__doc__
    while_loop.__doc__ = catalyst.while_loop.__doc__
