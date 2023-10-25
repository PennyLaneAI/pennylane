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
"""Compiler developer functions"""

from typing import List
from importlib import reload
from collections import defaultdict
import dataclasses
import pkg_resources


class CompileError(Exception):
    """Error encountered in the compilation phase."""


@dataclasses.dataclass
class AvailableCompilers:
    """This contains data of installed PennyLane compiler packages."""

    entrypoints_interface = ("qjit", "context", "ops")
    names_entrypoints = {}


def _refresh_compilers():
    """Scan installed PennyLane compiler packages to refresh the compilers
    names and entry points.
    """

    # Refresh the list of compilers
    AvailableCompilers.names_entrypoints = defaultdict(dict)

    # Iterator packages entry-points with the 'pennylane.compilers' group name
    for entry in pkg_resources.iter_entry_points("pennylane.compilers"):
        # Only need name of the parent module
        module_name = entry.module_name.split(".")[0]
        AvailableCompilers.names_entrypoints[module_name][entry.name] = entry

    # Check whether available compilers follow the entry_point interface
    # by validating that all entry points (qjit, context, and ops) are defined.
    for _, eps_dict in AvailableCompilers.names_entrypoints.items():
        ep_interface = AvailableCompilers.entrypoints_interface
        if any(ep not in eps_dict.keys() for ep in ep_interface):
            raise KeyError(f"expected {ep_interface}, but recieved {eps_dict}")


# Scan installed compiler packages
# and update AvailableCompilers
_refresh_compilers()


def _reload_compilers():
    """Reload and scan installed PennyLane compiler packages to refresh the
    compilers names and entry points.
    """

    reload(pkg_resources)
    _refresh_compilers()


def available_compilers() -> List[str]:
    """Loads and returns a list of available compilers that are
    installed and compatible with the :func:`~.qjit` decorator.

    **Example**

    This method returns the name of installed compiler packages supported in
    PennyLane. For example, after installing the
    `Catalyst <https://github.com/pennylaneai/catalyst>`__
    compiler, this will now appear as an available compiler:

    >>> qml.compiler.available_compilers()
    ['catalyst']
    """

    # Reload installed packages and updates
    # the class variable names_entrypoints
    _reload_compilers()

    return list(AvailableCompilers.names_entrypoints.keys())


def available(compiler="catalyst") -> bool:
    """Check the availability of the given compiler package.

    Args:
        compiler (str): name of the compiler package (default value is ``catalyst``)

    Return:
        bool: ``True`` if the compiler package is installed on the system

    **Example**

    Before installing the ``pennylane-catalyst`` package:

    >>> qml.compiler.available("catalyst")
    False

    After installing the ``pennylane-catalyst`` package:

    >>> qml.compiler.available("catalyst")
    True
    """

    # It only refreshes the compilers names and entry points if the name
    # is not already stored. This reduces the number of re-importing
    # ``pkg_resources`` as it can be a very slow operation on systems
    # with a large number of installed packages.

    if compiler not in AvailableCompilers.names_entrypoints:
        # Reload installed packages and updates
        # the class variable names_entrypoints
        _reload_compilers()

    return compiler in AvailableCompilers.names_entrypoints


def active_compiler() -> str:
    """Check which compiler is activated inside a :func:`~.qjit` evaluation context.

    This helper function may be used during implementation
    to allow differing logic for circuits or operations that are
    just-in-time compiled versus those that are not.

    Return:
        str or None: Name of the active compiler inside a :func:`~.qjit` evaluation context.

    **Example**

    For example, you can use this method in your hybrid program to execute it
    conditionally whether is called inside :func:`~.qjit` with ``"catalyst"``
    as the activate compiler or not.

    .. code-block:: python

        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(phi, theta):
            if qml.compiler.active() == "catalyst":
                qml.RX(phi, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PhaseShift(theta, wires=0)
            return qml.expval(qml.PauliZ(0))

    >>> circuit(np.pi, np.pi / 2)
    1.0
    >>> qml.qjit(circuit)(np.pi, np.pi / 2)
    -1.0

    """

    for name, eps in AvailableCompilers.names_entrypoints.items():
        tracer_loader = eps["context"].load()
        if tracer_loader.is_tracing():
            return name

    return None


def active() -> bool:
    """Check whether the caller is inside a :func:`~.qjit` evaluation context.

    This helper function may be used during implementation
    to allow differing logic for circuits or operations that are
    just-in-time compiled versus those that are not.

    Return:
        bool: True if the caller is inside a QJIT evaluation context

    **Example**

    For example, you can use this method in your hybrid program to execute it
    conditionally whether is called inside :func:`~.qjit` or not.

    .. code-block:: python

        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(phi, theta):
            if qml.compiler.active():
                qml.RX(phi, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PhaseShift(theta, wires=0)
            return qml.expval(qml.PauliZ(0))

    >>> circuit(np.pi, np.pi / 2)
    1.0
    >>> qml.qjit(circuit)(np.pi, np.pi / 2)
    -1.0
    """

    return active_compiler() is not None
