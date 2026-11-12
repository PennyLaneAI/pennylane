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

    A :class:`~pennylane.Backline` uses a transport to select how messages move between the
    :class:`~.Controller` and its :class:`coprocessors <.Coprocessor>`, and exposes it as
    :attr:`~pennylane.Backline.transport`. The compiler combines the transport with each node's
    hardware to select a concrete runtime backend.

    The built-in transports are:

    * ``"rdma"``: messages travel over RDMA, so every coprocessor needs an :class:`~.Endpoint`.
    * ``"memcpy"``: messages are copied in memory, with no network endpoint.
    * ``"none"``: no messages move. The placement is a :class:`~.Controller` alone, and passing
      any coprocessor raises an error.

    Any other transport a backend in Catalyst's transport backend registry implements is accepted
    as well, so a backend package adds a transport without a PennyLane change.

    .. warning::

        :mod:`Backline <.backline>` is experimental and only usable through the Catalyst
        compiler.

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
        ValueError: If ``name`` is neither registered here nor implemented by a backend in
            Catalyst's transport backend registry. Without Catalyst installed, any name is
            accepted and checked when the program is compiled.

    .. seealso:: :func:`~.register_transport`, :class:`~.Transport`
    """
    if name in _TRANSPORTS:
        return _TRANSPORTS[name]()
    known = _catalyst_transports()
    if known is None or name in known:
        return Transport(name)
    raise ValueError(
        f"unknown transport {name!r}; registered transports: {sorted({*_TRANSPORTS, *known})}"
    )


def _catalyst_transports():
    """The transports Catalyst's backend registry implements, or ``None`` without Catalyst."""
    try:
        from catalyst.backline import (  # pylint: disable=import-outside-toplevel
            backend_transports,
        )
    except ImportError:
        return None
    return backend_transports()


@register_transport("rdma")
def _rdma():
    return Transport("rdma")


@register_transport("memcpy")
def _memcpy():
    return Transport("memcpy")


@register_transport("none")
def _none():
    return Transport("none")
