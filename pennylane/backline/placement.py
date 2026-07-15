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

"""Placement types for backline heterogeneous execution."""

from dataclasses import dataclass

from .transports import Transport, get_transport


@dataclass(frozen=True)
class Endpoint:
    """A single backline endpoint: a host and the role it plays in the fabric.

    Args:
        host (str): Hostname or IP, optionally ``host:port``.
        role (str): Short label for the node's role, e.g. ``"fpga-qpu"`` or ``"gpu-decoder"``.
        name (str | None): Optional unique name, used to disambiguate when several endpoints share a
            role. Defaults to ``None``.
        local (bool): Whether the endpoint is attached to the host running the program (``True``) or a
            separate machine reached over the network (``False``). Local endpoints run their code
            directly; remote endpoints are cross-compiled and dispatched. Defaults to
            ``True`` (local).
        attrs (dict | None): Optional backend-specific hints (e.g. a cross-compilation target
            triple). Defaults to ``None``.
        decoder (str | object | None): The decoder this endpoint runs (coprocessors only). A
            ``str`` names a decoder implementation (selector); an object is a builder the
            compiler consumes. Meaningful only for endpoints placed in
            :attr:`Backline.coprocessors`. Defaults to ``None``.
    """

    host: str
    role: str
    name: str | None = None
    local: bool = True
    attrs: dict | None = None
    decoder: str | object | None = None


@dataclass(frozen=True)
class Backline:
    """Declarative placement for heterogeneous execution.

    Groups the controller node, any coprocessor nodes, and the transport that carries data between
    them.

    Args:
        controller (Endpoint): The node controlling the QPU.
        transport (str | Transport): How bytes move between endpoints, by registry name (e.g.
            ``"roce"``) or a :class:`~.Transport`.
        coprocessors (tuple[Endpoint, ...]): Accelerators for coprocessing. Defaults to ``()``.
    """

    controller: Endpoint
    transport: str | Transport
    coprocessors: tuple = ()

    def __post_init__(self):
        if isinstance(self.transport, str):
            get_transport(self.transport)


def backline(controller, transport, coprocessors=()):
    """Construct a :class:`Backline` placement."""
    return Backline(controller=controller, transport=transport, coprocessors=coprocessors)
