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

from pennylane.exceptions import CompileError

from .compiler import AvailableCompilers, _check_compiler_version, available


def qjit(fn=None, *args, compiler="catalyst", **kwargs):  # pylint:disable=keyword-arg-before-vararg
    """A decorator for just-in-time compilation of hybrid quantum programs in PennyLane.

    This decorator enables both just-in-time and ahead-of-time compilation,
    depending on the compiler package and whether function argument type hints
    are provided.

    .. note::

        Currently, only two compilers are supported; the :doc:`Catalyst <catalyst:index>` hybrid
        quantum-classical compiler, which works with the JAX interface, and CUDA Quantum.

        For more details on Catalyst, see the :doc:`Catalyst documentation <catalyst:index>` and
        :func:`catalyst.qjit`.

    .. note::

        Catalyst only supports the JAX interface and selected devices.
        Supported backend devices for Catalyst include
        ``lightning.qubit``, ``lightning.kokkos``, ``lightning.gpu``, and ``braket.aws.qubit``,
        but **not** ``default.qubit``.

        For a full list of supported devices, please see :doc:`catalyst:dev/devices`.

        CUDA Quantum supports ``softwareq.qpp``, ``nvidia.custatevec``, and ``nvidia.cutensornet``.

    Args:
        fn (Callable): Hybrid (quantum-classical) function to compile
        compiler (str): Name of the compiler to use for just-in-time compilation. Available
            options include ``catalyst`` and ``cuda_quantum``.
        autograph (bool): Experimental support for automatically converting Python control
            flow statements to Catalyst-compatible control flow. Currently supports Python ``if``,
            ``elif``, ``else``, and ``for`` statements. Note that this feature requires an
            available TensorFlow installation. See the
            :doc:`AutoGraph guide <catalyst:dev/autograph>` for more information.
        keep_intermediate (bool): Whether or not to store the intermediate files throughout the
            compilation. The files are stored at the location where the Python script is called.
            If ``True``, intermediate representations are available via the
            :attr:`~.QJIT.mlir`, :attr:`~.QJIT.jaxpr`, and :attr:`~.QJIT.qir`, representing
            different stages in the optimization process.
        verbosity (bool): If ``True``, the tools and flags used by Catalyst behind the scenes are
            printed out.
        logfile (TextIOWrapper): File object to write verbose messages to (default is
            ``sys.stderr``)
        pipelines (List[Tuple[str, List[str]]]): A list of pipelines to be executed. The
            elements of this list are named sequences of MLIR passes to be executed. A ``None``
            value (the default) results in the execution of the default pipeline. This option is
            considered to be used by advanced users for low-level debugging purposes.
        static_argnums(int or Sequence[Int]): an index or a sequence of indices that specifies the
            positions of static arguments.
        abstracted_axes (Sequence[Sequence[str]] or Dict[int, str] or Sequence[Dict[int, str]]):
            An experimental option to specify dynamic tensor shapes.
            This option affects the compilation of the annotated function.
            Function arguments with ``abstracted_axes`` specified will be compiled to ranked tensors
            with dynamic shapes. For more details, please see the Dynamically-shaped Arrays section
            below.

    Returns:
        catalyst.QJIT: A class that, when executed, just-in-time compiles and executes the
        decorated function

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
            return qml.expval(qml.Z(1))

    >>> circuit(0.5)  # the first call, compilation occurs here
    array(0.)
    >>> circuit(0.5)  # the precompiled quantum function is called
    array(0.)

    :func:`~.qjit` compiled programs also support nested container types as inputs and outputs of
    compiled functions. This includes lists and dictionaries, as well as any data structure implementing
    the `JAX PyTree <https://jax.readthedocs.io/en/latest/pytrees.html>`__.

    .. code-block:: python

        dev = qml.device("lightning.qubit", wires=2)

        @qml.qjit
        @qml.qnode(dev)
        def f(x):
            qml.RX(x["rx_param"], wires=0)
            qml.RY(x["ry_param"], wires=0)
            qml.CNOT(wires=[0, 1])
            return {
                "XY": qml.expval(qml.X(0) @ qml.Y(1)),
                "X": qml.expval(qml.X(0)),
            }

    >>> x = {"rx_param": 0.5, "ry_param": 0.54}
    >>> f(x)
    {'X': array(-0.75271018), 'XY': array(1.)}

    For more details on using the :func:`~.qjit` decorator and Catalyst
    with PennyLane, please refer to the Catalyst
    :doc:`quickstart guide <catalyst:dev/quick_start>`,
    as well as the :doc:`sharp bits and debugging tips <catalyst:dev/sharp_bits>`
    page for an overview of the differences between Catalyst and PennyLane, and
    how to best structure your workflows to improve performance when
    using Catalyst.

    .. details::
        :title: Static arguments

        ``static_argnums`` defines which elements should be treated as static. If it takes an
        integer, it means the argument whose index is equal to the integer is static. If it takes
        an iterable of integers, arguments whose index is contained in the iterable are static.
        Changing static arguments will trigger re-compilation.

        A valid static argument must be hashable and its ``__hash__`` method must be able to
        reflect any changes of its attributes.

        .. code-block:: python

            @dataclass
            class MyClass:
                val: int

                def __hash__(self):
                    return hash(str(self))

            @qjit(static_argnums=1)
            def f(
                x: int,
                y: MyClass,
            ):
                return x + y.val

            f(1, MyClass(5))
            f(1, MyClass(6)) # re-compilation
            f(2, MyClass(5)) # no re-compilation

        In the example above, ``y`` is static. Note that the second function call triggers
        re-compilation since the input object is different from the previous one. However,
        the third function call directly uses the previous compiled one and does not introduce
        re-compilation.

        .. code-block:: python

            @dataclass
            class MyClass:
                val: int

                def __hash__(self):
                    return hash(str(self))

            @qjit(static_argnums=(1, 2))
            def f(
                x: int,
                y: MyClass,
                z: MyClass,
            ):
                return x + y.val + z.val

            my_obj_1 = MyClass(5)
            my_obj_2 = MyClass(6)
            f(1, my_obj_1, my_obj_2)
            my_obj_1.val = 7
            f(1, my_obj_1, my_obj_2) # re-compilation

        In the example above, ``y`` and ``z`` are static. The second function will cause
        function ``f`` to re-compile because ``my_obj_1`` is changed. This requires that
        the mutation is properly reflected in the hash value.

        Note that when ``static_argnums`` is used in conjunction with type hinting,
        ahead-of-time compilation will not be possible since the static argument values
        are not yet available. Instead, compilation will be just-in-time.


    .. details::
        :title: Dynamically-shaped arrays

        There are three ways to use ``abstracted_axes``; by passing a sequence of tuples, a
        dictionary, or a sequence of dictionaries. Passing a sequence of tuples:

        .. code-block:: python

            abstracted_axes=((), ('n',), ('m', 'n'))

        Each tuple in the sequence corresponds to one of the arguments in the annotated
        function. Empty tuples can
        be used and correspond to parameters with statically known shapes.
        Non-empty tuples correspond to parameters with dynamically known shapes.

        In this example above,

        - the first argument will have a statically known shape,

        - the second argument will have dynamic
          shape ``n``  for the zeroth axis, and

        - the third argument will have dynamic shape
          ``m`` for its zeroth axis and dynamic shape ``n`` for
          its first axis.

        Passing a dictionary:

        .. code-block:: python

            abstracted_axes={0: 'n'}

        This approach allows a concise expression of the relationships
        between axes for different function arguments. In this example,
        it specifies that for all function arguments, the zeroth axis will
        have dynamic shape ``n``.

        Passing a sequence of dictionaries:

        .. code-block:: python

            abstracted_axes=({}, {0: 'n'}, {1: 'm', 0: 'n'})

        The example here is a more verbose version of the tuple example. This convention
        allows axes to be omitted from the list of abstracted axes.

        Using ``abstracted_axes`` can help avoid the cost of recompilation.
        By using ``abstracted_axes``, a more general version of the compiled function will be
        generated. This more general version is parametrized over the abstracted axes and
        allows results to be computed over tensors independently of their axes lengths.

        For example:

        .. code-block:: python

            @qjit
            def sum(arr):
                return jnp.sum(arr)

            sum(jnp.array([1]))     # Compilation happens here.
            sum(jnp.array([1, 1]))  # And here!

        The ``sum`` function would recompile each time an array of different size is passed
        as an argument.

        .. code-block:: python

            @qjit(abstracted_axes={0: "n"})
            def sum_abstracted(arr):
                return jnp.sum(arr)

            sum(jnp.array([1]))     # Compilation happens here.
            sum(jnp.array([1, 1]))  # No need to recompile.

        The ``sum_abstracted`` function would only compile once and its definition would be
        reused for subsequent function calls.
    """

    if not available(compiler):
        raise CompileError(f"The {compiler} package is not installed.")  # pragma: no cover

    # Check the minimum version of 'compiler' if installed
    _check_compiler_version(compiler)

    compilers = AvailableCompilers.names_entrypoints
    qjit_loader = compilers[compiler]["qjit"].load()
    return qjit_loader(fn=fn, *args, **kwargs)
