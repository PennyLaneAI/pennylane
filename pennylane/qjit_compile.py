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

try:
    import catalyst

    pl_qjit_available = True
except ImportError:
    pl_qjit_available = False


# Catalyst QJIT decorator
def qjit(*args, **kwargs):
    """The ``catalyst.qjit`` wrapper method"""

    if not pl_qjit_available:
        raise ImportError(
            "Catalyst is required for the QJIT-compilation mode, "
            "you can install it with `pip install pennylane-catalyst`"
        )

    return catalyst.qjit(*args, **kwargs)


# Catalyst control-flow statements
def if_cond(*args, **kwargs):
    """The ``catalyst.cond`` wrapper method"""

    if not pl_qjit_available:
        raise ImportError(
            "Catalyst is required for the QJIT-compilation mode, "
            "you can install it with `pip install pennylane-catalyst`"
        )

    return catalyst.cond(*args, **kwargs)


def for_loop(*args, **kwargs):
    """The ``catalyst.for_loop`` wrapper method"""

    if not pl_qjit_available:
        raise ImportError(
            "Catalyst is required for the QJIT-compilation mode, "
            "you can install it with `pip install pennylane-catalyst`"
        )

    return catalyst.for_loop(*args, **kwargs)


def while_loop(*args, **kwargs):
    """The ``catalyst.while`` wrapper method"""

    if not pl_qjit_available:
        raise ImportError(
            "Catalyst is required for the QJIT-compilation mode, "
            "you can install it with `pip install pennylane-catalyst`"
        )

    return catalyst.while_loop(*args, **kwargs)


# Catalyst wrapper methods used in dispatchers
def catalyst_is_tracing():
    """The ``catalyst.while`` wrapper method"""

    return catalyst.utils.tracing.TracingContext.is_tracing()


def catalyst_measure(wires):
    """The ``catalyst.measure`` wrapper method"""

    return catalyst.measure(wires)


def catalyst_adjoint(fn):
    """The ``catalyst.adjoint`` wrapper method"""

    return catalyst.adjoint(fn)


def catalyst_grad(fun, method=None, h=None, argnum=None):
    """The ``catalyst.grad`` wrapper method"""

    return catalyst.grad(fun, method=method, h=h, argnum=argnum)


def catalyst_jacobian(fun, method=None, h=None, argnum=None):
    """The ``catalyst.jacobian`` wrapper method"""

    return catalyst.jacobian(fun, method=method, h=h, argnum=argnum)


if pl_qjit_available:
    qjit.__doc__ = catalyst.qjit.__doc__
    if_cond.__doc__ = catalyst.cond.__doc__
    for_loop.__doc__ = catalyst.for_loop.__doc__
    while_loop.__doc__ = catalyst.while_loop.__doc__
    catalyst_measure.__doc__ = catalyst.measure.__doc__
    catalyst_adjoint.__doc__ = catalyst.adjoint.__doc__
    catalyst_grad.__doc__ = catalyst.grad.__doc__
    catalyst_jacobian.__doc__ = catalyst.jacobian.__doc__
