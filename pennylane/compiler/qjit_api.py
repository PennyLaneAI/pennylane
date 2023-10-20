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
"""QJIT compatible quantum and compilation operations API"""

from .compiler import AvailableCompilers, available


def qjit(fn=None, *args, compiler="catalyst", **kwargs):  # pylint:disable=keyword-arg-before-vararg
    """A just-in-time decorator for PennyLane and JAX programs.

    .. note::

        This is a wrapper around
        `catalyst.qjit <https://docs.pennylane.ai/projects/catalyst/en/latest/code/api/catalyst.qjit.html>`__.


    This decorator enables both just-in-time and ahead-of-time compilation,
    depending on the compiler package and whether function argument type hints
    are provided.

    Args:
        compiler(str): name of the compiler package (default value is ``catalyst``)
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

        dev = qml.device("lightning.qubit", wires=2)

        @qml.qjit
        @qml.qnode(dev)
        def circuit(theta):
            qml.Hadamard(wires=0)
            qml.RX(theta, wires=1)
            qml.CNOT(wires=[0,1])
            return qml.expval(qml.PauliZ(wires=1))

    >>> circuit(0.5)  # the first call, compilation occurs here
    array(0.)
    >>> circuit(0.5)  # the precompiled quantum function is called
    array(0.)

    For more details on using the :func:`~.qjit` decorator and Catalyst
    with PennyLane, please refer to the Catalyst
    `quickstart guide <https://docs.pennylane.ai/projects/catalyst/en/latest/dev/quick_start.html>`__,
    as well as the `sharp bits and debugging tips <https://docs.pennylane.ai/projects/catalyst/en/latest/dev/sharp_bits.html>`__
    page for an overview of the differences between Catalyst and PennyLane, and
    how to best structure your workflows to improve performance when
    using Catalyst.
    """

    if not available(compiler):
        raise RuntimeError(f"The {compiler} package is not installed.")

    compilers = AvailableCompilers.names_entrypoints
    qjit_loader = compilers[compiler]["qjit"].load()
    return qjit_loader(fn=fn, *args, **kwargs)
