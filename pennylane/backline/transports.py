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

    See the Attributes section to learn more about the available options.
    """

    name: str
    """The registry name of the transport, e.g. ``"rdma"``."""


def register_transport(name):
    """Register a transport factory under ``name``.

    Args:
        name (str): The name the transport is selected by.

    Returns:
        Callable: A decorator that registers the factory.
    """

    def decorator(factory):
        _TRANSPORTS[name] = factory
        return factory

    return decorator


def get_transport(name):
    """Return the :class:`Transport` registered under ``name``.

    Args:
        name (str): The transport name.

    Returns:
        Transport: The transport produced by the registered factory.

    Raises:
        ValueError: If ``name`` is not registered.
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
