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
"""This subpackage provides the JIT compilation support for PennyLane.


This subpackage exclusively functions as a wrapper for PennyLane's JIT compiler packages, without independently
implementing any compiler itself. Currently, it supports the ``pennylane-catalyst`` package, with plans to
incorporate additional packages in the near future.

For any compiler packages seeking to be registered, it is imperative that they expose the 'entry_points' metadata
under the designated group name: ``pennylane.compilers``.
"""

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


def available(name="catalyst") -> bool:
    """Check the availability of the given compiler package.

    It only refreshes the compilers names and entry points if the name
    is not already stored. This reduces the number of re-importing
    ``pkg_resources`` as it can be a very slow operation on systems
    with a large number of installed packages.

    Args:
        name (str): name of the compiler package (Default is ``catalyst``)

    Return:
        bool: ``True`` if the compiler package is installed on the system

    **Example**

    Before installing the ``pennylane-catalyst`` package:

    .. code-block:: python

    >>> qml.compiler.available("catalyst")
    False

    After installing the ``pennylane-catalyst`` package:

    >>> qml.compiler.available("catalyst")
    True
    """

    if name not in AvailableCompilers.names_entrypoints:
        # This class updates the class variable names_entrypoints
        _refresh_compilers()

    return name in AvailableCompilers.names_entrypoints


def available_compilers() -> List[str]:
    """Return the name of available compilers by refreshing the compilers
    names and entry points.

    **Example**

    This method returns the name of installed compiler packages supported in
    PennyLane. For example, after installing the ``pennylane-catalyst`` pacakge,

    .. code-block:: python

    >>> qml.compiler.available_compilers()
    ['catalyst']
    """

    # This class updates the class variable names_entrypoints
    _refresh_compilers()

    return list(AvailableCompilers.names_entrypoints.keys())


def active(name="catalyst") -> bool:
    """Check whether the caller is inside a QJIT evaluation context.

    Args:
        name (str): name of the compiler package (Default is ``catalyst``)

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
        tracer_loader = compilers[name]["context"].load()
        return tracer_loader.is_tracing()
    except KeyError:
        return False


def qjit(
    fn=None, *args, compiler_name="catalyst", **kwargs
):  # pylint:disable=keyword-arg-before-vararg
    """A just-in-time decorator for PennyLane and JAX programs.

    This decorator enables both just-in-time and ahead-of-time compilation,
    depending on the compiler package and whether function argument type hints
    are provided.

    Args:
        compiler_name(str): name of the compiler package (Default is ``catalyst``)
        fn (Callable): the quantum or classical function
        autograph (bool): Experimental support for automatically converting Python control
            flow statements to Catalyst-compatible control flow. Currently supports Python ``if``,
            ``elif``, ``else``, and ``for`` statements. Note that this feature requires an
            available TensorFlow installation. Please see the
            `AutoGraph guide <https://docs.pennylane.ai/projects/catalyst/en/latest/dev/autograph.html>`__
            for more information.
        target (str): the compilation target
        keep_intermediate (bool): Whether or not to store the intermediate files throughout the
            compilation. If ``True``, intermediate representations are available via the
            :attr:`~.QJIT.mlir`, :attr:`~.QJIT.jaxpr`, and :attr:`~.QJIT.qir`, representing
            different stages in the optimization process.
        verbosity (bool): If ``True``, the tools and flags used by Catalyst behind the scenes are
            printed out.
        logfile (Optional[TextIOWrapper]): File object to write verbose messages to (default -
            ``sys.stderr``).
        pipelines (Optional(List[Tuple[str,List[str]]])): A list of pipelines to be executed. The
            elements of this list are named sequences of MLIR passes to be executed. A ``None``
            value (the default) results in the execution of the default pipeline. This option is
            considered to be used by advanced users for low-level debugging purposes.

    Returns:
        QJIT object.

    Raises:
        FileExistsError: Unable to create temporary directory
        PermissionError: Problems creating temporary directory
        OSError: Problems while creating folder for intermediate files
        AutoGraphError: Raised if there was an issue converting the given the function(s).
        ImportError: Raised if AutoGraph is turned on and TensorFlow could not be found.

    **Example**

    In just-in-time (JIT) mode, the compilation is triggered at the call site the
    first time the quantum function is executed. For example, ``circuit`` is
    compiled as early as the first call.

    .. code-block:: python

        @qjit
        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def circuit(theta):
            qml.Hadamard(wires=0)
            qml.RX(theta, wires=1)
            qml.CNOT(wires=[0,1])
            return qml.expval(qml.PauliZ(wires=1))

    >>> circuit(0.5)  # the first call, compilation occurs here
    array(0.)
    >>> circuit(0.5)  # the precompiled quantum function is called
    array(0.)

    Alternatively, if argument type hints are provided, compilation
    can occur 'ahead of time' when the function is decorated.

    .. code-block:: python

        from jax.core import ShapedArray

        @qjit  # compilation happens at definition
        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def circuit(x: complex, z: ShapedArray(shape=(3,), dtype=jnp.float64)):
            theta = jnp.abs(x)
            qml.RY(theta, wires=0)
            qml.Rot(z[0], z[1], z[2], wires=0)
            return qml.state()

    >>> circuit(0.2j, jnp.array([0.3, 0.6, 0.9]))  # calls precompiled function
    array([0.75634905-0.52801002j, 0. +0.j,
        0.35962678+0.14074839j, 0. +0.j])

    Catalyst also supports capturing imperative Python control flow in compiled programs. You can
    enable this feature via the ``autograph=True`` parameter. Note that it does come with some
    restrictions, in particular whenever global state is involved. Refer to the documentation page
    for a complete discussion of the supported and unsupported use-cases.

    .. code-block:: python

        @qjit(autograph=True)
        @qml.qnode(qml.device("lightning.qubit", wires=2))
        def circuit(x: int):

            if x < 5:
                qml.Hadamard(wires=0)
            else:
                qml.T(wires=0)

            return qml.expval(qml.PauliZ(0))

    >>> circuit(3)
    array(0.)

    >>> circuit(5)
    array(1.)
    """

    if not available(compiler_name):
        raise RuntimeError(f"The {compiler_name} package is not installed.")

    compilers = AvailableCompilers.names_entrypoints
    qjit_loader = compilers[compiler_name]["qjit"].load()
    return qjit_loader(fn=fn, *args, **kwargs)
