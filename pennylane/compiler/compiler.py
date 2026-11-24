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
"""Catalyst availability and compilation-context helpers."""

import re
import sys
from importlib import metadata

from packaging.version import Version

from pennylane.exceptions import CompileError

PL_CATALYST_MIN_VERSION = Version("0.15.0")


class AvailableCompilers:
    """Entry-point loaders exported by the installed Catalyst distribution."""

    entrypoints_interface = ("context", "qjit", "ops")
    names_entrypoints = {}


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
    """Refresh entry points from the installed Catalyst distribution."""

    AvailableCompilers.names_entrypoints = {}
    try:
        distribution = metadata.distribution("pennylane-catalyst")
    except metadata.PackageNotFoundError:
        return

    ep_interface = AvailableCompilers.entrypoints_interface
    eps_dict = {
        name: entry
        for entry in distribution.entry_points
        for name in ep_interface
        if entry.group == "pennylane.compilers" and entry.name == f"catalyst.{name}"
    }
    if any(name not in eps_dict for name in ep_interface):
        raise KeyError(f"expected Catalyst entry points {ep_interface}, but received {eps_dict}")
    AvailableCompilers.names_entrypoints = {"catalyst": eps_dict}


_refresh_compilers()


def _reload_compilers():
    """Refresh Catalyst's installed package metadata."""

    _refresh_compilers()


def available_compilers() -> list[str]:
    """Return ``["catalyst"]`` if Catalyst is installed, or an empty list otherwise.

    **Example**

    For example, after installing the
    `Catalyst <https://github.com/pennylaneai/catalyst>`__
    compiler, this will now appear as an available compiler:

    >>> qp.compiler.available_compilers()
    ['catalyst']
    """

    _reload_compilers()

    return list(AvailableCompilers.names_entrypoints.keys())


def available(compiler="catalyst") -> bool:
    """Check whether Catalyst is installed.

    Args:
        compiler (str): Only ``"catalyst"`` is supported. Other names return ``False``.

    Returns:
        bool: ``True`` if Catalyst is requested and installed on the system

    **Example**

    Before installing the ``pennylane-catalyst`` package:

    >>> qp.compiler.available("catalyst")
    False

    After installing the ``pennylane-catalyst`` package:

    >>> qp.compiler.available("catalyst")
    True
    """

    if compiler != "catalyst":
        return False

    if compiler not in AvailableCompilers.names_entrypoints:
        _reload_compilers()

    return compiler in AvailableCompilers.names_entrypoints


def active_compiler() -> str | None:
    """Check whether Catalyst is activated inside a :func:`~.qjit` evaluation context.

    This helper function may be used during implementation
    to allow differing logic for transformations or operations that are
    just-in-time compiled, versus those that are not.

    Returns:
        Optional[str]: ``"catalyst"`` inside a Catalyst tracing context, or ``None`` otherwise.

    **Example**

    This method can be used to execute logical
    branches that are conditioned on whether Catalyst compilation is occurring.

    .. code-block:: python

        dev = qp.device("lightning.qubit", wires=2)

        @qp.qnode(dev)
        def circuit(phi, theta):
            if qp.compiler.active_compiler() == "catalyst":
                qp.RX(phi, wires=0)
            qp.CNOT(wires=[0, 1])
            qp.PhaseShift(theta, wires=0)
            return qp.expval(qp.Z(0))

    >>> circuit(np.pi, np.pi / 2)
    1.0
    >>> qp.qjit(circuit)(np.pi, np.pi / 2)
    -1.0

    """

    eps = AvailableCompilers.names_entrypoints.get("catalyst")
    if eps is not None and "catalyst" in sys.modules:
        tracer_loader = eps["context"].load()
        if tracer_loader.is_tracing():
            return "catalyst"

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

        dev = qp.device("lightning.qubit", wires=2)

        @qp.qnode(dev)
        def circuit(phi, theta):
            if qp.compiler.active():
                qp.RX(phi, wires=0)
            qp.CNOT(wires=[0, 1])
            qp.PhaseShift(theta, wires=0)
            return qp.expval(qp.Z(0))

    >>> circuit(np.pi, np.pi / 2)
    1.0
    >>> qp.qjit(circuit)(np.pi, np.pi / 2)
    -1.0
    """
    return active_compiler() is not None
