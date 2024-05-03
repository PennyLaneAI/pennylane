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

from typing import List, Optional
from sys import version_info
from importlib import reload, metadata
from collections import defaultdict
import dataclasses
import re

from semantic_version import Version

PL_CATALYST_MIN_VERSION = Version("0.6.0")


class CompileError(Exception):
    """Error encountered in the compilation phase."""


@dataclasses.dataclass
class AvailableCompilers:
    """This contains data of installed PennyLane compiler packages."""

    # The collection of entry points that compiler packages must export.
    # Note that this is still an experimental interface and is subject to change.
    # This variable is used for validity checks of installed packages entry points.
    # For any compiler packages seeking to be registered, it is imperative
    # that they expose the ``entry_points`` metadata under the designated
    # group name ``pennylane.compilers``, with the following entry points:
    # - ``context``: Path to the compilation evaluation context manager.
    # - ``ops``: Path to the compiler operations module.
    # - ``qjit``: Path to the JIT decorator provided by the compiler.
    entrypoints_interface = ("context", "qjit", "ops")

    # The dictionary of installed compiler packages
    # and their entry point loaders.
    names_entrypoints = defaultdict(dict)


def _check_compiler_version(name):
    """Check if the installed version of the given compiler is greater than
    or equal to the required minimum version.
    """
    if name == "catalyst":
        installed_catalyst_version = metadata.version("pennylane-catalyst")
        if Version(re.sub(r"\.dev\d+", "", installed_catalyst_version)) < PL_CATALYST_MIN_VERSION:
            raise CompileError(
                f"PennyLane-Catalyst {PL_CATALYST_MIN_VERSION} or greater is required, but installed {installed_catalyst_version}"
            )


def _refresh_compilers():
    """Scan installed PennyLane compiler packages to refresh the compilers
    names and entry points.
    """

    # Refresh the list of compilers
    AvailableCompilers.names_entrypoints = defaultdict(dict)

    # Iterator packages entry-points with the 'pennylane.compilers' group name
    entries = (
        defaultdict(dict, metadata.entry_points())["pennylane.compilers"]
        if version_info[:2] == (3, 9)
        # pylint:disable=unexpected-keyword-arg
        else metadata.entry_points(group="pennylane.compilers")
    )

    for entry in entries:
        try:
            # First element of split is the compiler name
            # New convention for entry point.
            compiler_name, e_name = entry.name.split(".")
            AvailableCompilers.names_entrypoints[compiler_name][e_name] = entry  # pragma: no cover
        except ValueError:
            # Keep old behaviour.
            # TODO: Deprecate in 0.35 release
            compiler_name = entry.module.split(".")[0]
            AvailableCompilers.names_entrypoints[compiler_name][entry.name] = entry

    # Check whether available compilers follow the entry_point interface
    # by validating that all entry points (qjit, context, and ops) are defined.
    for _, eps_dict in AvailableCompilers.names_entrypoints.items():
        ep_interface = AvailableCompilers.entrypoints_interface
        if any(ep not in eps_dict.keys() for ep in ep_interface):
            raise KeyError(f"expected {ep_interface}, but recieved {eps_dict}")  # pragma: no cover


# Scan installed compiler packages
# and update AvailableCompilers
_refresh_compilers()


def _reload_compilers():
    """Reload and scan installed PennyLane compiler packages to refresh the
    compilers names and entry points.
    """

    # Note re-importing ``importlib.metadata`` can be a very slow operation
    # on systems with a large number of installed packages.

    reload(metadata)
    _refresh_compilers()


def available_compilers() -> List[str]:
    """Load and return a list of available compilers that are
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
        compiler (str): Name of the compiler package (default value is ``catalyst``)

    Returns:
        bool: ``True`` if the compiler package is installed on the system

    **Example**

    Before installing the ``pennylane-catalyst`` package:

    >>> qml.compiler.available("catalyst")
    False

    After installing the ``pennylane-catalyst`` package:

    >>> qml.compiler.available("catalyst")
    True
    """

    # It only refreshes the compilers names and entry points if
    # the name is not already stored.

    if compiler not in AvailableCompilers.names_entrypoints:
        # Reload installed packages and updates
        # the class variable names_entrypoints
        _reload_compilers()

    return compiler in AvailableCompilers.names_entrypoints


def active_compiler() -> Optional[str]:
    """Check which compiler is activated inside a :func:`~.qjit` evaluation context.

    This helper function may be used during implementation
    to allow differing logic for transformations or operations that are
    just-in-time compiled, versus those that are not.

    Returns:
        Optional[str]: Name of the active compiler inside a :func:`~.qjit` evaluation
        context. If there is no active compiler, ``None`` will be returned.

    **Example**

    This method can be used to execute logical
    branches that are conditioned on whether hybrid compilation with a specific
    compiler is occurring.

    .. code-block:: python

        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(phi, theta):
            if qml.compiler.active_compiler() == "catalyst":
                qml.RX(phi, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PhaseShift(theta, wires=0)
            return qml.expval(qml.Z(0))

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

    Returns:
        bool: ``True`` if the caller is inside a QJIT evaluation context

    **Example**

    For example, you can use this method in your hybrid program to execute it
    conditionally whether called inside :func:`~.qjit` or not.

    .. code-block:: python

        dev = qml.device("lightning.qubit", wires=2)

        @qml.qnode(dev)
        def circuit(phi, theta):
            if qml.compiler.active():
                qml.RX(phi, wires=0)
            qml.CNOT(wires=[0, 1])
            qml.PhaseShift(theta, wires=0)
            return qml.expval(qml.Z(0))

    >>> circuit(np.pi, np.pi / 2)
    1.0
    >>> qml.qjit(circuit)(np.pi, np.pi / 2)
    -1.0
    """

    return active_compiler() is not None
