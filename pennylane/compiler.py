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
implementing any compiler package itself. Presently, it exclusively supports the 'pennylane-catalyst package,
with plans to incorporate additional packages in the near future.

For any compiler packages seeking to be registered, it is imperative that they expose the 'entry_points' metadata
under the designated group name: 'pennylane.compilers'.
"""

from typing import List
from importlib import reload
import pkg_resources


class Compiler:
    """The JIT compiler module containing all utilities and wrapper methods for ``qml.qjit`` in PennyLane."""

    # Private class properties
    __BACKENDS = []
    __ENTRY_POINTS = {}

    @classmethod
    def available(cls, name: str = "pennylane_catalyst") -> bool:
        """Check the availability of the given compiler package.

        Args:
            name (str): the name of the compiler package

        Return:
            bool : True if the compiler package is installed on the system
        """

        reload(pkg_resources)
        pkg_name = pkg_resources.Requirement.parse(name)

        if pkg_resources.working_set.find(pkg_name):
            if name not in cls.__BACKENDS:
                cls.__ENTRY_POINTS = {
                    entry.name: entry
                    for entry in pkg_resources.iter_entry_points("pennylane.compilers")
                }
                cls.__BACKENDS.append(name)
            return True

        if name in cls.__BACKENDS:
            cls.__BACKENDS.remove(name)

        return False

    @classmethod
    def available_backends(cls) -> List[str]:
        """Return the available compiler packages."""
        return cls.__BACKENDS

    @classmethod
    def active(cls) -> bool:
        """Check whether the caller is inside a QJIT evaluation context.

        Return:
            bool : True if the caller is inside a QJIT evaluation context
        """
        if not cls.__BACKENDS:
            raise RuntimeError("There is no available compiler package.")

        if "cpl_utils" not in cls.__ENTRY_POINTS:
            raise RuntimeError("There is no available 'cpl_utils' entry point.")

        utils_loader = cls.__ENTRY_POINTS["cpl_utils"].load()
        return utils_loader.contexts.EvaluationContext.is_tracing()

    @classmethod
    def qjit(cls, fn=None, *args, **kwargs):
        """A just-in-time decorator for PennyLane and JAX programs using Catalyst.

        This decorator enables both just-in-time and ahead-of-time compilation,
        depending on whether function argument type hints are provided.

        .. note::

            Currently, ``lightning.qubit`` is the only supported backend device
            for Catalyst compilation. For a list of supported operations, observables,
            and measurements, please see the :doc:`/dev/quick_start`.

        Args:
            fn (Callable): the quantum or classical function
            autograph (bool): Experimental support for automatically converting Python control
                flow statements to Catalyst-compatible control flow. Currently supports Python ``if``,
                ``elif``, ``else``, and ``for`` statements. Note that this feature requires an
                available TensorFlow installation.
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
            catalyst.QJIT object.

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

        Note that imperative control flow will still work in Catalyst even when the AutoGraph feature is
        turned off, it just won't be captured in the compiled program and cannot involve traced values.
        The example above would then raise a tracing error, as there is no value for ``x`` yet than can
        be compared in the if statement. A loop like ``for i in range(5)`` would be unrolled during
        tracing, "copy-pasting" the body 5 times into the program rather than appearing as is.

        .. important::

            Most decomposition logic will be equivalent to PennyLane's decomposition.
            However, decomposition logic will differ in the following cases:

            1. All :class:`qml.Controlled <pennylane.ops.op_math.Controlled>` operations will decompose
                to :class:`qml.QubitUnitary <pennylane.QubitUnitary>` operations.

            2. :class:`qml.ControlledQubitUnitary <pennylane.ControlledQubitUnitary>` operations will
                decompose to :class:`qml.QubitUnitary <pennylane.QubitUnitary>` operations.

            3. The list of device-supported gates employed by Catalyst is currently different than that
                of the ``lightning.qubit`` device, as defined by the
                :class:`~.pennylane_extensions.QJITDevice`.
        """

        if not cls.__BACKENDS and not cls.available():
            raise RuntimeError("There is no available compiler package.")

        if "cpl_qjit" not in cls.__ENTRY_POINTS:
            raise RuntimeError("There is no available 'cpl_qjit' entry point.")

        qjit_loader = cls.__ENTRY_POINTS["cpl_qjit"].load()
        return qjit_loader(fn=fn, *args, **kwargs)


# Exported Methods

qjit = Compiler.qjit
