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

from .compiler import (
    _check_compiler_version,
    CompileError,
    AvailableCompilers,
    available,
    active_compiler,
)


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

        Catalyst supports compiling QNodes that use ``lightning.qubit``,
        ``lightning.kokkos``, ``braket.local.qubit``, and ``braket.aws.qubit``
        devices. It does not support ``default.qubit``.

        Please see the :doc:`Catalyst documentation <catalyst:index>` for more details on
        supported devices, operations, and measurements.

        CUDA Quantum supports ``softwareq.qpp``, ``nvidida.custatevec``, and ``nvidia.cutensornet``.

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
        static_argnums(int or Seqence[Int]): an index or a sequence of indices that specifies the
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


def while_loop(cond_fn):
    """A :func:`~.qjit` compatible while-loop for PennyLane programs.

    .. note::

        This function only supports the Catalyst compiler. See
        :func:`catalyst.while_loop` for more details.

        Please see the Catalyst :doc:`quickstart guide <catalyst:dev/quick_start>`,
        as well as the :doc:`sharp bits and debugging tips <catalyst:dev/sharp_bits>`
        page for an overview of the differences between Catalyst and PennyLane.

    This decorator provides a functional version of the traditional while
    loop, similar to `jax.lax.while_loop <https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.while_loop.html>`__.
    That is, any variables that are modified across iterations need to be provided as
    inputs and outputs to the loop body function:

    - Input arguments contain the value of a variable at the start of an
      iteration

    - Output arguments contain the value at the end of the iteration. The
      outputs are then fed back as inputs to the next iteration.

    The final iteration values are also returned from the
    transformed function.

    The semantics of ``while_loop`` are given by the following Python pseudo-code:

    .. code-block:: python

        def while_loop(cond_fn, body_fn, *args):
            while cond_fn(*args):
                args = body_fn(*args)
            return args

    Args:
        cond_fn (Callable): the condition function in the while loop

    Returns:
        Callable: A wrapper around the while-loop function.

    Raises:
        CompileError: if the compiler is not installed

    .. seealso:: :func:`~.for_loop`, :func:`~.qjit`

    **Example**

    .. code-block:: python

        dev = qml.device("lightning.qubit", wires=1)

        @qml.qjit
        @qml.qnode(dev)
        def circuit(x: float):

            @qml.while_loop(lambda x: x < 2.0)
            def loop_rx(x):
                # perform some work and update (some of) the arguments
                qml.RX(x, wires=0)
                return x ** 2

            # apply the while loop
            final_x = loop_rx(x)

            return qml.expval(qml.Z(0)), final_x

    >>> circuit(1.6)
    (array(-0.02919952), array(2.56))
    """

    if active_jit := active_compiler():
        compilers = AvailableCompilers.names_entrypoints
        ops_loader = compilers[active_jit]["ops"].load()
        return ops_loader.while_loop(cond_fn)

    raise CompileError("There is no active compiler package.")  # pragma: no cover


def for_loop(lower_bound, upper_bound, step):
    """A :func:`~.qjit` compatible for-loop for PennyLane programs.

    .. note::

        This function only supports the Catalyst compiler. See
        :func:`catalyst.for_loop` for more details.

        Please see the Catalyst :doc:`quickstart guide <catalyst:dev/quick_start>`,
        as well as the :doc:`sharp bits and debugging tips <catalyst:dev/sharp_bits>`
        page for an overview of the differences between Catalyst and PennyLane.

    This decorator provides a functional version of the traditional
    for-loop, similar to `jax.cond.fori_loop <https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.fori_loop.html>`__.
    That is, any variables that are modified across iterations need to be provided
    as inputs/outputs to the loop body function:

    - Input arguments contain the value of a variable at the start of an
      iteration.

    - output arguments contain the value at the end of the iteration. The
      outputs are then fed back as inputs to the next iteration.

    The final iteration values are also returned from the transformed
    function.

    The semantics of ``for_loop`` are given by the following Python pseudo-code:

    .. code-block:: python

        def for_loop(lower_bound, upper_bound, step, loop_fn, *args):
            for i in range(lower_bound, upper_bound, step):
                args = loop_fn(i, *args)
            return args

    Unlike ``jax.cond.fori_loop``, the step can be negative if it is known at tracing time
    (i.e., constant). If a non-constant negative step is used, the loop will produce no iterations.

    Args:
        lower_bound (int): starting value of the iteration index
        upper_bound (int): (exclusive) upper bound of the iteration index
        step (int): increment applied to the iteration index at the end of each iteration

    Returns:
        Callable[[int, ...], ...]: A wrapper around the loop body function.
        Note that the loop body function must always have the iteration index as its first
        argument, which can be used arbitrarily inside the loop body. As the value of the index
        across iterations is handled automatically by the provided loop bounds, it must not be
        returned from the function.

    Raises:
        CompileError: if the compiler is not installed

    .. seealso:: :func:`~.while_loop`, :func:`~.qjit`

    **Example**


    .. code-block:: python

        dev = qml.device("lightning.qubit", wires=1)

        @qml.qjit
        @qml.qnode(dev)
        def circuit(n: int, x: float):

            @qml.for_loop(0, n, 1)
            def loop_rx(i, x):
                # perform some work and update (some of) the arguments
                qml.RX(x, wires=0)

                # update the value of x for the next iteration
                return jnp.sin(x)

            # apply the for loop
            final_x = loop_rx(x)

            return qml.expval(qml.Z(0)), final_x

    >>> circuit(7, 1.6)
    (array(0.97926626), array(0.55395718))
    """

    if active_jit := active_compiler():
        compilers = AvailableCompilers.names_entrypoints
        ops_loader = compilers[active_jit]["ops"].load()
        return ops_loader.for_loop(lower_bound, upper_bound, step)

    raise CompileError("There is no active compiler package.")  # pragma: no cover
