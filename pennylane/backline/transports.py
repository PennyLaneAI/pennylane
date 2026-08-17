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

"""The named transport registry for backline placement.

A Transport specifies how data move between executors, and are chosen by name. The data transport
implementation lives in the compiled runtime.
"""

from dataclasses import dataclass

_TRANSPORTS = {}


@dataclass(frozen=True)
class Transport:
    """A named data transport.

    Passed to :class:`~pennylane.Backline` as the ``transport`` argument to select how messages move
    between the :class:`~.Controller` and its :class:`coprocessors <.Coprocessor>`.

    .. warning::

        Backline is experimental. Its API may change without notice, and it is only usable through
        the Catalyst compiler.

    Args:
        name (str): The registry name of the transport, e.g. ``"rdma"``.

    .. seealso:: :func:`~.get_transport`, :func:`~.register_transport`

    **Example**

    Transports are normally selected by name, and :class:`~pennylane.Backline` resolves the string:

    >>> dev = qp.Backline(controller=qp.Controller(), transport="rdma")
    >>> dev.transport
    Transport(name='rdma')

    Passing a :class:`~.Transport` instance directly is equivalent.
    """

    name: str
    """The registry name of the transport, e.g. ``"rdma"``."""


def register_transport(name):
    """Register a transport factory under ``name``.

    Args:
        name (str): The name the transport is selected by, as passed to
            :class:`~pennylane.Backline`.

    Returns:
        Callable: A decorator that registers the factory.

    .. seealso:: :func:`~.get_transport`, :class:`~.Transport`
    """

    def decorator(factory):
        _TRANSPORTS[name] = factory
        return factory

    return decorator


def get_transport(name):
    """Return the :class:`Transport` registered under ``name``.

    Args:
        name (str): The transport name, as registered by :func:`~.register_transport`.

    Returns:
        Transport: The :class:`~.Transport` produced by the registered factory.

    Raises:
        ValueError: If ``name`` is not registered.

    .. seealso:: :func:`~.register_transport`, :class:`~.Transport`
    """
    try:
        factory = _TRANSPORTS[name]
    except KeyError:
        raise ValueError(
            f"unknown transport {name!r}; registered transports: {sorted(_TRANSPORTS)}"
        ) from None
    return factory()


@register_transport("rdma")
def _rdma():
    return Transport("rdma")
