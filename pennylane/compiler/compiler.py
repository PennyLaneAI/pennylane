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
import dataclasses
import pkg_resources


@dataclasses.dataclass
class AvailableCompilers:
    """This contains data of installed PennyLane compiler packages"""

    names_entrypoints = {}


def _refresh_compilers():
    """Scan installed PennyLane compiler pacakges to refresh the compilers
    names and entry points
    """

    reload(pkg_resources)

    # Refresh the list of compilers
    AvailableCompilers.names_entrypoints = {}

    # Iterator packages entry-points with the 'pennylane.compilers' group name
    for entry in pkg_resources.iter_entry_points("pennylane.compilers"):
        module_name = entry.module_name
        # Only need name of the parent module
        module_name = module_name.split(".")[0]

        if module_name not in AvailableCompilers.names_entrypoints:
            AvailableCompilers.names_entrypoints[module_name] = {}
        AvailableCompilers.names_entrypoints[module_name][entry.name] = entry


def available_compilers() -> List[str]:
    """Loads and returns a list of available compilers that are
    installed and compatible with the :func:`~.qjit` decorator.

    **Example**

    This method returns the name of installed compiler packages supported in
    PennyLane. For example, after installing the `Catalyst <https://github.com/pennylaneai/catalyst>`__
    compiler, this will now appear as an available compiler:

    >>> qml.compiler.available_compilers()
    ['catalyst']
    """

    # This class updates the class variable names_entrypoints
    _refresh_compilers()

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
        # This class updates the class variable names_entrypoints
        _refresh_compilers()

    return compiler in AvailableCompilers.names_entrypoints


def active(compiler="catalyst") -> bool:
    """Check whether the caller is inside a :func:`~.qjit` evaluation context.

    This helper function may be used during implementation
    to allow differing logic for circuits or operations that are
    just-in-time compiled versus those that are not.

    Args:
        compiler (str): name of the compiler package (default value is ``catalyst``)

    Return:
        bool: True if the caller is inside a QJIT evaluation context

    **Example**

    In the JIT compilation of PennyLane programs, this helper method checks
    the status of the compilation. For example, Catalyst captures Python
    programs by tracing and we can tell if the caller is inside the Catalyst
    tracer context manager or not but calling this method.

    This method is practically useful in implementing quantum operations to
    correctly call the interpreter or compiler equivalent functions.
    """

    compilers = AvailableCompilers.names_entrypoints

    if not compilers:
        return False

    try:
        tracer_loader = compilers[compiler]["context"].load()
        return tracer_loader.is_tracing()
    except KeyError:
        return False
